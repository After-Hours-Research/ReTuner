import loguru as logger
from pathlib import Path
import random
import re
from typing import List, Tuple
import lmql
import torch
from tqdm import tqdm

from retrieve.data_load import load_document
from torch.utils.data import Dataset

import transformers

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.texts = [
            (
                tokenizer(item['question'], padding=True, truncation=True, return_tensors='pt'), 
                tokenizer(item['chunk'], padding=True, truncation=True, return_tensors='pt')
            ) 
            for item in texts
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class QuestionGenerator:
    def __init__(
            self, 
            docs_id: List[Path],
            docs_ood: List[Path],
            model = "openai/gpt-3.5-turbo",
            max_id_samples: int = 1000,
            max_ood_samples: int = 100,
            chunk_size: int = 200
    ):
        if model.startswith("openai"):
            self.model = model
        else:
            logger.info("Local model specified. Make sure to run with: lmql serve-model")
            self.model = lmql.model(model)
        self.chunk_size = chunk_size
        self.docs_id = self._load_docs(docs_id)
        random.shuffle(self.docs_id)
        self.docs_ood = self._load_docs(docs_ood)
        self.max_id_samples = min(len(self.docs_id), max_id_samples)
        self.max_ood_samples = min(len(self.docs_ood), max_ood_samples)

    def _load_docs(self, docs):
        loaded_docs = []
        for doc in docs:
            loaded_docs.extend(load_document(doc, min_chunk_size=self.chunk_size))
        return loaded_docs
        
    @lmql.query
    def _data_gen(
        self,
        text: str,
        n: int, 
        temperature: float = 0.8
    ):
        '''lmql
        sample(temperature=temperature)
            """
            {text}

            Come up with a list of orthogonal questions that could reasonably be answered from the above text. \n
            """
            for i in range(int(n)):
                "<QUESTION> [QUESTION] </QUESTION>"
        from
            self.model
        where
            STOPS_AT(QUESTION, "?") or STOPS_AT(QUESTION, "\n")
        '''

    def extract_questions(self, text):
        questions = re.findall('<QUESTION>(.*?)</QUESTION>', text, re.DOTALL)
        questions = [re.sub('^\s*\d+\.\s*', '', q).strip() for q in questions]
        return questions

    def generate_questions(
            self,
            text_chunk: str, 
            n_train_questions: int,
            n_test_questions: int,
            temperature: float = 0.8
        ) -> Tuple[List[dict], List[dict]]:
        results = self._data_gen(
            text=text_chunk, 
            n=n_train_questions + n_test_questions, 
            temperature=temperature
        )
        results = self.extract_questions(results[0].prompt)
        qc_pairs = [
            {"question": result, "chunk": text_chunk}
            for result in results
        ]
        return qc_pairs[:n_train_questions], qc_pairs[n_train_questions:]
    
    def _create_dict_dataset(
        self,
        n_train_per_doc: int,
        n_test_id_per_doc: int,
        n_test_ood_per_doc: int,
        temperature: float = 0.8
    ):
        train = []
        test_id = []
        test_ood = []
        for doc in tqdm(self.docs_id[:self.max_id_samples], desc="Generating in-distribution questions"):
            train_questions, test_questions = self.generate_questions(
                doc, n_train_per_doc, n_test_id_per_doc, temperature
            )
            train.extend(train_questions)
            test_id.extend(test_questions)
        for doc in tqdm(self.docs_ood[:self.max_ood_samples], desc="Generating out-of-distribution questions"):
            _, test_questions = self.generate_questions(
                doc, 0, n_test_ood_per_doc, temperature
            )
            test_ood.extend(test_questions)
        return train, test_id, test_ood
    
    def create_dataset(
        self,
        tokenizer: transformers.AutoTokenizer,
        n_train_per_doc: int,
        n_test_id_per_doc: int,
        n_test_ood_per_doc: int,
        temperature: float = 0.8
    ):
        train, test_id, test_ood = self._create_dict_dataset(
            n_train_per_doc, n_test_id_per_doc, n_test_ood_per_doc, temperature
        )
        train_dataset = TextDataset(tokenizer, train)
        test_id_dataset = TextDataset(tokenizer, test_id)
        test_ood_dataset = TextDataset(tokenizer, test_ood)
        return train_dataset, test_id_dataset, test_ood_dataset

    @classmethod
    def from_directory(
        cls,
        id_directory: Path,
        ood_directory: Path,
        model: str = "openai/gpt-3.5-turbo",
        max_id_samples: int = 1000,
        max_ood_samples: int = 100,
        chunk_size: int = 200,
    ):
        docs_id = [f for f in id_directory.iterdir() if not f.is_dir()]
        docs_ood = [f for f in ood_directory.iterdir() if not f.is_dir()]
        return cls(docs_id, docs_ood, model, max_id_samples, max_ood_samples, chunk_size)
    
    def save_dataset(
        self,
        file_path: Path,
        tokenizer: transformers.AutoTokenizer,
        n_train_per_doc: int,
        n_test_id_per_doc: int,
        n_test_ood_per_doc: int,
        temperature: float = 0.8,
    ):
        train_dataset, test_id_dataset, test_ood_dataset = self.create_dataset(
            tokenizer, n_train_per_doc, n_test_id_per_doc, n_test_ood_per_doc, temperature
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train": train_dataset,
            "test_id": test_id_dataset,
            "test_ood": test_ood_dataset
        }, str(file_path))