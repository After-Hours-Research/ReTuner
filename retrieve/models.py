from transformers import AutoModel
import torch
import torch.nn.functional as F


class HFEmbedder:
    def __init__(
        self,
        hf_modelname: str,
    ):
        self.model = AutoModel.from_pretrained(hf_modelname)

    def embed(self, encoded_input: str):
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(pooled, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)
    