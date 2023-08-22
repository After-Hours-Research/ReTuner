from typing import Any, Optional, Type
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

import pytorch_lightning as pl
from transformers import AutoModel

class NoiseScheduler:
    def __init__(self, noise_min, noise_max, max_steps):
        self.noise_arr = torch.linspace(noise_min, noise_max, max_steps)

    def __getitem__(self, step):
        self.current = self.noise_arr[step]
        return self.current
        

def mean_pooling(model_output, attention_mask, dim=1):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=dim) / torch.clamp(input_mask_expanded.sum(dim), min=1e-9)

class HFEmbedder:
    def __init__(
        self,
        hf_modelname: str,
    ):
        self.model = AutoModel.from_pretrained(hf_modelname)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def embed(
            self, 
            encoded_input: str,
            pooling: bool = True,
            dim: int = 1,
        ):
        with torch.no_grad():
            model_output = self.model(**encoded_input)[0] # 0 because MPNet returns a tuple - 0 is embedding
        if pooling:
            pooled = mean_pooling(model_output, encoded_input['attention_mask'], dim=dim)
            return F.normalize(pooled, p=2, dim=dim)
        else:
            return model_output

    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.embed(*args, **kwargs)
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    @property
    def embedding_dim(self):
        return self.model.config.hidden_size
    
    def device(self, device):
        self.model.to(device)
        return self
    
class Baseline(nn.Module):
    def __init__(self, embedding_size: int):
        super(Baseline, self).__init__()
    
    def forward(self, x, mask):
        return x

class FullyConnectedHead(nn.Module):
    def __init__(self, embedding_size: int):
        super(FullyConnectedHead, self).__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)
        with torch.no_grad():
            self.fc.weight.data = torch.eye(embedding_size)
            self.fc.bias.data.fill_(0.0)
    
    def forward(self, x, mask):
        return self.fc(x)
    
class FCBottleneckHeadWithDropout(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.3, bottleneck_mult: int = 1):
        super(FCBottleneckHeadWithDropout, self).__init__()
        self.fc1 = nn.Linear(embedding_size, int(embedding_size * bottleneck_mult))
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(int(embedding_size * bottleneck_mult), embedding_size)
        self.gelu = nn.GELU()
        with torch.no_grad():
            self.fc1.weight.data = torch.eye(embedding_size, int(embedding_size * bottleneck_mult))
            self.fc1.bias.data.fill_(0.0)
            self.fc2.weight.data = torch.eye(int(embedding_size * bottleneck_mult), embedding_size)
            self.fc2.bias.data.fill_(0.0)
    
    def forward(self, x, mask):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))
    
class ResidualHeadWithDropout(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.3, bottleneck_divisor: int = 2):
        super(ResidualHeadWithDropout, self).__init__()
        self.fc1 = nn.Linear(embedding_size, embedding_size // bottleneck_divisor)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_size // bottleneck_divisor, embedding_size)
        self.gelu = nn.GELU()
        with torch.no_grad():
            self.fc1.weight.data = torch.full((embedding_size, embedding_size // bottleneck_divisor), 1e-9)
            self.fc1.bias.data.fill_(0.0)
            self.fc2.weight.data = torch.full((embedding_size // bottleneck_divisor, embedding_size), 1e-9)
            self.fc2.bias.data.fill_(0.0)
    
    def forward(self, x, mask):
        return self.fc2(self.dropout(self.gelu(self.fc1(x)))) + x
    

class AttentionHead(nn.Module):
    def __init__(self, embedding_size: int, num_heads : int =4, dropout: float = 0.3, inverse_bn_mult: int = 2):
        super(AttentionHead, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.fc1 = nn.Linear(embedding_size, int(embedding_size * inverse_bn_mult))
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(int(embedding_size * inverse_bn_mult), embedding_size)
    
    def forward(self, x, mask):
        x = self.norm1(x)
        mask = ~mask.bool()
        mask = mask.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1)).repeat(4, 1, 1)
        xt = x.transpose(0, 1)
        h, _ = self.attention(xt, xt, xt, attn_mask=mask, need_weights=False)
        h = h.transpose(0, 1)
        x = x + h
        h = self.norm2(x)
        h = self.fc2(self.gelu(self.fc1(h)))
        x = x + h
        return x
    
class ResidualAttentionHead(nn.Module):
    def __init__(self, embedding_size: int, num_heads : int =4, dropout: float = 0.3, inverse_bn_mult: int = 2):
        super(ResidualAttentionHead, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.fc1 = nn.Linear(embedding_size, int(embedding_size * inverse_bn_mult))
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(int(embedding_size * inverse_bn_mult), embedding_size)
        with torch.no_grad():
            dev = self.fc1.weight.device
            # kaiming init ith 1e-9
            self.fc2.weight.data = torch.randn((embedding_size, int(embedding_size * inverse_bn_mult)), device=dev) * 1e-9
            self.fc2.bias.data.fill_(1e-9)

    
    def forward(self, x, mask):
        x = self.norm1(x)
        mask = ~mask.bool()
        mask = mask.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1)).repeat(4, 1, 1)
        h, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        h = self.norm2(h)
        h = self.fc2(self.gelu(self.fc1(h)))
        x = x + h
        return x


class InitialValCallback(pl.Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        return trainer.validate(model=pl_module, dataloaders=trainer.val_dataloaders)
    
class TuningModule(pl.LightningModule):
    def __init__(self,
            original_embedding_model, 
            tuning_head_model: Type[nn.Module], 
            tuning_params: Optional[dict] = None,
            val_ids_per_ood: int = 5,
            loss_name: str = "mse",
            margin: float = 0.2,
            epochs: int = 100,
            adding_noise: bool = False,
            noise_max: float = 1.0,
            noise_min: float = 0.01
        ):
        super(TuningModule, self).__init__()
        self.val_ids_per_ood = val_ids_per_ood
        self.loss_name = loss_name
        self.margin = margin

        self.adding_noise = adding_noise
        self.noise_max = noise_max
        self.noise_min = noise_min


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        # Freeze the original embedding model
        self.original_embedding_model = original_embedding_model.freeze()
        self.epochs = epochs
        
        # Create the fully connected head
        original_embedding_size = self.original_embedding_model.embedding_dim
        self.tuning_params = tuning_params if tuning_params is not None else {}
        self.tuning_head = tuning_head_model(embedding_size=original_embedding_size, **self.tuning_params)

        if loss_name == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = self.triplet_loss

    def _get_noise(self, embeddings, mask):
        noise_scale = self.noise_scheduler[self.trainer.global_step]
        shuffled_indices = torch.randperm(embeddings.size(0))
        shuffled_embeddings = embeddings[shuffled_indices]
        shuffled_mask = mask[shuffled_indices]
        mixup_noise = 0.1 * (shuffled_embeddings - embeddings) * shuffled_mask.unsqueeze(-1)
        return mixup_noise
        

    def forward(self, chunk, query):
        # Embed the chunk using the original model
        chunk_embedding = self.original_embedding_model(chunk)        
        # Embed the query using the original model
        query_embedding_original = self.original_embedding_model(query, pooling=False)
        if self.adding_noise and self.trainer.training:
            query_embedding_original += self._get_noise(query_embedding_original, query['attention_mask'])
        
        # Transform the query embedding using the fully connected head
        query_embedding_transformed = self.tuning_head(query_embedding_original, query['attention_mask'])
        query_embedding_transformed = mean_pooling(query_embedding_transformed, query['attention_mask'], dim=1)
        query_embedding_transformed = F.normalize(query_embedding_transformed, p=2, dim=1)
        
        return chunk_embedding, query_embedding_transformed
    
    def on_fit_start(self) -> None:
        self.ood_epoch = self.trainer.check_val_every_n_epoch * self.val_ids_per_ood

    def on_train_start(self):
        max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        if self.adding_noise:
            self.noise_scheduler = NoiseScheduler(self.noise_min, self.noise_max, max_steps)


    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        pos_dist = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

    def infonce_loss(self, anchor, positive, negatives):
        """
        InfoNCE loss function.
        
        Args:
        - anchor (torch.Tensor): Anchor embeddings of shape (batch_size, embedding_dim).
        - positive (torch.Tensor): Positive embeddings of shape (batch_size, embedding_dim).
        - negatives (torch.Tensor): Negative embeddings of shape (batch_size, num_negatives, embedding_dim).
        
        Returns:
        - loss (torch.Tensor): Scalar tensor representing the loss.
        """
        
        # Compute similarity between anchor and positive
        pos_sim = (anchor * positive).sum(dim=-1, keepdim=True)
        
        # Compute similarity between anchor and negatives
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)
        
        # Log-sum-exp trick for numerical stability
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)  # Targets the positive samples
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def _get_negative_sample(self, chunk_embedding):
        batch_size = chunk_embedding.size(0)
        all_indices = torch.arange(batch_size).unsqueeze(0).repeat(batch_size, 1)
        mask = torch.eye(batch_size).bool()
        all_indices[mask] = -1e9
        negative_idx = torch.multinomial(F.softmax(all_indices.float(), dim=1), 1).squeeze()
        negative_chunk_embeddings = chunk_embedding[negative_idx]
        return negative_chunk_embeddings

    def _get_unique_negatives(self, embeddings):
        """
        For each embedding in the input, finds the unique set of embeddings that are different from it using tensor operations.
        
        Parameters:
        - embeddings (torch.Tensor): A tensor of shape (N, embedding_size)
        
        Returns:
        - torch.Tensor: A tensor of shape (N, unique(N) - 1, embedding_size)
        """
        
        # Get unique embeddings
        unique_embeddings = torch.unique(embeddings, dim=0)
        
        # Initialize an empty tensor to store the results
        result = torch.empty((embeddings.shape[0], unique_embeddings.shape[0]-1, embeddings.shape[1])).to(embeddings.device)
        
        # For each embedding, gather unique embeddings that are different from it
        for idx, emb in enumerate(embeddings):
            mask = (unique_embeddings != emb).any(dim=1)
            result[idx] = unique_embeddings[mask]
        
        return result
    
    def batch_loss(self, batch):
        chunk, query = batch
        chunk_embedding, query_embedding_transformed = self(chunk, query)
        if self.loss_name == "triplet":
            negative_chunk_embeddings = self._get_negative_sample(chunk_embedding)
            loss = self.triplet_loss(query_embedding_transformed, chunk_embedding, negative_chunk_embeddings)
        elif self.loss_name == "infonce":
            negative_chunk_embeddings = self._get_unique_negatives(chunk_embedding)
            loss = self.infonce_loss(query_embedding_transformed, chunk_embedding, negative_chunk_embeddings)
        else:
            loss = self.loss(chunk_embedding, query_embedding_transformed)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        self.log('train_loss', loss)
        if self.adding_noise:
            self.log('added_noise', self.noise_scheduler.current)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.tuning_head.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output = {0: [], 1: []}
        return None
    
    def check_ood_epoch(self):
        return self.current_epoch % self.ood_epoch == 0

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 1 and not self.check_ood_epoch():
            return
        loss = self.batch_loss(batch)

        # Log the loss separately for each validation set
        if dataloader_idx == 0:
            self.log('val_id_loss', loss)
        elif dataloader_idx == 1:
            self.log('val_ood_loss', loss)
        self.val_output[dataloader_idx].append(loss)


    def on_validation_epoch_end(self):
        # Separate the outputs for the ID and OOD validation sets
        id_outputs = torch.tensor(self.val_output[0])
        avg_id_loss = torch.mean(id_outputs)
        self.log('avg_val_id_loss', avg_id_loss, prog_bar=True)

        if self.check_ood_epoch():
            ood_outputs = torch.tensor(self.val_output[1])
            avg_ood_loss = torch.mean(ood_outputs)
            self.log('avg_val_ood_loss', avg_ood_loss, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output = {0: [], 1: []}
        return None
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        loss = self.batch_loss(batch)
        # Log the loss separately for each validation set
        if dataloader_idx == 0:
            self.log('test_id_loss', loss)
        elif dataloader_idx == 1:
            self.log('test_ood_loss', loss)
        self.test_output[dataloader_idx].append(loss)

    def on_test_epoch_end(self):
        # Separate the outputs for the ID and OOD validation sets
        id_outputs = torch.tensor(self.test_output[0])
        ood_outputs = torch.tensor(self.test_output[1])

        # Compute the average loss for each set
        avg_id_loss = torch.mean(id_outputs)
        avg_ood_loss = torch.mean(ood_outputs)
        # Log the results
        self.log('avg_test_id_loss', avg_id_loss, prog_bar=True)
        self.log('avg_test_ood_loss', avg_ood_loss, prog_bar=True)

    def describe(self):
        return {
            'original_embedding_model': self.original_embedding_model,
            'tuning_head': self.tuning_head,
            'loss': self.loss,
            **self.tuning_params,
        }
    
    def predict(self, query):
        query_embedding_original = self.original_embedding_model(query, pooling=False)
        query_embedding_transformed = self.tuning_head(query_embedding_original, query["attention_mask"])
        query_embedding_transformed = mean_pooling(query_embedding_transformed, query['attention_mask'], dim=1)
        query_embedding_transformed = F.normalize(query_embedding_transformed, p=2, dim=1)
        return query_embedding_transformed
