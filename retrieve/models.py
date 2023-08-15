from typing import Any, Optional, Type
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import torch.nn as nn

import pytorch_lightning as pl
from transformers import AutoModel


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
    
    @property
    def parameters(self):
        return self.model.parameters()
    
    @property
    def embedding_dim(self):
        return self.model.config.hidden_size
    
    def device(self, device):
        self.model.to(device)
        return self

class FullyConnectedHead(nn.Module):
    def __init__(self, embedding_size: int):
        super(FullyConnectedHead, self).__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, x):
        return self.fc(x)
    
class FCBottleneckHeadWithDropout(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.3, bottleneck_divisor: int = 2):
        super(FCBottleneckHeadWithDropout, self).__init__()
        self.fc1 = nn.Linear(embedding_size, embedding_size // bottleneck_divisor)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_size // bottleneck_divisor, embedding_size)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
    
class ResidualHeadWithDropout(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.3, bottleneck_divisor: int = 2):
        super(ResidualHeadWithDropout, self).__init__()
        self.fc1 = nn.Linear(embedding_size, embedding_size // bottleneck_divisor)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_size // bottleneck_divisor, embedding_size)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x)))) + x


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
            val_ids_per_ood: int = 5
        ):
        super(TuningModule, self).__init__()
        self.ood_epoch = self.trainer.check_val_every_n_epoch * val_ids_per_ood
        
        # Freeze the original embedding model
        self.original_embedding_model = original_embedding_model
        for param in self.original_embedding_model.parameters:
            param.requires_grad = False
        
        # Create the fully connected head
        original_embedding_size = self.original_embedding_model.embedding_dim
        self.tuning_params = tuning_params if tuning_params is not None else {}
        self.tuning_head = tuning_head_model(embedding_size=original_embedding_size, **self.tuning_params)
        self.loss = nn.MSELoss()

    def forward(self, chunk, query):
        # Embed the chunk using the original model
        chunk_embedding = self.original_embedding_model(chunk)
        
        # Embed the query using the original model
        query_embedding_original = self.original_embedding_model(query, pooling=False)
        
        # Transform the query embedding using the fully connected head
        query_embedding_transformed = self.tuning_head(query_embedding_original)
        query_embedding_transformed = mean_pooling(query_embedding_transformed, query['attention_mask'], dim=1)
        query_embedding_transformed = F.normalize(query_embedding_transformed, p=2, dim=1)
        
        return chunk_embedding, query_embedding_transformed

    def training_step(self, batch, batch_idx):
        chunk, query = batch
        chunk_embedding, query_embedding_transformed = self(chunk, query)
        loss = self.loss(chunk_embedding, query_embedding_transformed)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.tuning_head.parameters(), lr=1e-3)
        return optimizer
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output = {0: [], 1: []}
        return None
    
    def ood_epoch(self):
        return self.current_epoch % self.ood_epoch == 0

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 1 and not self.ood_epoch():
            return
        chunk, query = batch
        chunk_embedding, query_embedding_transformed = self(chunk, query)
        loss = self.loss(chunk_embedding, query_embedding_transformed)

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

        if self.ood_epoch():
            ood_outputs = torch.tensor(self.val_output[1])
            avg_ood_loss = torch.mean(ood_outputs)
            self.log('avg_val_ood_loss', avg_ood_loss, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output = {0: [], 1: []}
        return None
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        chunk, query = batch
        chunk_embedding, query_embedding_transformed = self(chunk, query)
        loss = self.loss(chunk_embedding, query_embedding_transformed)

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
        query_embedding_transformed = self.tuning_head(query_embedding_original)
        query_embedding_transformed = mean_pooling(query_embedding_transformed, query['attention_mask'], dim=1)
        query_embedding_transformed = F.normalize(query_embedding_transformed, p=2, dim=1)
        return query_embedding_transformed


