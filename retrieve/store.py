from enum import Enum
from functools import partial
from pathlib import Path
from typing_extensions import Literal
from loguru import logger

from transformers import AutoModel, AutoTokenizer
import torch

import chromadb

from tqdm import tqdm

class DataSplit(Enum):
    TRAIN = "train"
    TEST_ID = "test_id"
    TEST_OOD = "test_ood"

class RetrievalStore:

    def __init__(
        self,
        dataset_path: Path,
        store_path: Path,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        name: DataSplit,
        metric: Literal['cosine', 'l2'],
    ):
        logger.info("Loading dataset")
        self.loaded_dataset = torch.load(dataset_path)[name.value]

        self.client = chromadb.PersistentClient(path=str(store_path))
        self.collection = self.client.create_collection(name=name.name, metadata={"hnsw:space": metric}, get_or_create=True)

        self.model = model
        self.metric = metric
        self.token_func = partial(tokenizer, padding=True, truncation=True, return_tensors='pt')

        self.load_documents()

    def _add_document(
        self, 
        embedding, 
        document,
        idx
    ):
        self.collection.add(
            documents = [document],
            embeddings = [embedding.squeeze(0).tolist()],
            ids = [str(idx)],
        )

    def load_documents(self):
        logger.info("Loading documents")
        for i in tqdm(range(len(self.loaded_dataset)), total=len(self.loaded_dataset), desc="Loading documents"):
            chunk_text = self.loaded_dataset.get_text(i)["chunk"]
            tokens = self.token_func(chunk_text)
            embd = self.model(tokens)
            self._add_document(embd, chunk_text, i)

    def query(self, query: str, emb_model, k=2):
        tokens = self.token_func(query)
        embd = emb_model(tokens).squeeze(0).tolist()
        results = self.collection.query(query_embeddings=embd, n_results=k)
        breakpoint()
        return results




