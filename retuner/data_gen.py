from collections import Counter
from functools import partial
from loguru import logger
from pathlib import Path
import random
import re
from typing import List, Tuple, Union
import lmql
import numpy as np
import torch
from tqdm import tqdm

from retuner.data_load import process_document
from torch.utils.data import Dataset

import transformers
import datasets
import pandas as pd


class QCGenerationCallback:
    def __init__(self) -> None:
        pass

    def on_load(self, loaded):
        raise NotImplementedError

    def on_api_end(self, questions, n_train, n_val, n_test):
        raise NotImplementedError

class LengthFilterCallback(QCGenerationCallback):
    def __init__(self, min_length: int = 100) -> None:
        self.min_length = min_length

    def on_load(self, loaded: List[str]):
        return [text for text in loaded if self.min_length <= len(text)]

class TrainTestValCallback(QCGenerationCallback):
    def __init__(self) -> None:
        super().__init__()

    def _rouge_1(self, question_1, question_2):
        gen_tokens = Counter(question_1.split())
        ref_tokens = Counter(question_2.split())
        
        gen_total = sum(gen_tokens.values())
        ref_total = sum(ref_tokens.values())
        
        # Directly compute overlapping words count
        overlap_count = sum(min(gen_tokens[word], ref_tokens[word]) for word in gen_tokens)
        
        if gen_total == 0 or ref_total == 0:
            return 0.0
        
        precision = overlap_count / gen_total
        recall = overlap_count / ref_total
        
        if precision + recall == 0:
            return 0.0
        else:
            return 2 * (precision * recall) / (precision + recall)
    
    def on_api_end(self, questions, n_train, n_val, n_test):
        if n_train <= 0:
            return None, questions[:n_val], questions[n_val:]
        rogue_matrix = np.zeros((len(questions), len(questions)))
        for i in range(len(questions)):
            for j in range(len(questions)):
                if i == j:
                    rogue_matrix[i, j] = 0
                    continue
                rogue_matrix[i, j] = self._rouge_1(questions[i], questions[j])
        rouge_max = rogue_matrix.max(axis=0)
        sorted_rogue = np.argsort(rouge_max)
        training_set = np.array(questions)[sorted_rogue[n_val + n_test:]].tolist()
        other_set = np.array(questions)[sorted_rogue[:n_val + n_test]].tolist()
        np.random.shuffle(other_set)
        return training_set, other_set[:n_val], other_set[n_val:]




        


class RetrievalDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tok_func = partial(tokenizer, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        self.texts = self.remove_nans(texts)
        logger.info(f"Dataset initialized with {len(self.texts)} texts")
    
    def remove_nans(self, texts):
        logger.info(f"Dataset has {len(texts)} texts")
        for i, text in enumerate(texts):
            if isinstance(text["question"], float) or isinstance(text["chunk"], float):
                texts.pop(i)
        logger.info(f"Filtered dataset - now has {len(texts)} texts")
        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (self.tok_func(self.texts[idx]['question']), self.tok_func(self.texts[idx]['chunk']))

    def get_text(self, idx):
        return self.texts[idx]

    def copy(self):
        return RetrievalDataset(self.tok_func.func, self.texts)

    def get_unique_chunks(self):
        self_copy = self.copy()
        self_copy.texts = list(set([text['chunk'] for text in self.texts]))
        return self_copy


class QuestionGenerator:
    def __init__(
            self, 
            docs_id: Union[List[str], List[Path]],
            docs_ood: Union[List[str], List[Path]],
            model = "openai/gpt-3.5-turbo",
            max_id_samples: int = 1000,
            max_ood_samples: int = 100,
            chunk_size: int = 600,
            chunk_ceiling: int = 1000,
            seed: int = 42,
            intermittent_save_path: Path = None,
            filter_callbacks: List[QCGenerationCallback] = [QCGenerationCallback()]
    ):
        self.seed = seed
        self.chunk_ceiling = chunk_ceiling
        random.seed(self.seed)
        if model.startswith("openai"):
            self.model = model
        else:
            logger.info("Local model specified. Make sure to run with: lmql serve-model")
            self.model = lmql.model(model)
        self.chunk_size = chunk_size
        self.filter_callbacks = filter_callbacks
        self.docs_id = self._load_docs(docs_id)
        random.shuffle(self.docs_id)  
        self.docs_ood = self._load_docs(docs_ood)          
        self.max_id_samples = min(len(self.docs_id), max_id_samples)
        self.max_ood_samples = min(len(self.docs_ood), max_ood_samples)
        self.intermittent_save_path = intermittent_save_path

    def _load_docs(self, docs):
        loaded_docs = []
        for doc in docs:
            loaded_docs.extend(process_document(doc, min_chunk_size=self.chunk_size, max_chunk_size=self.chunk_ceiling))
        for callback in self.filter_callbacks:
            try:
                loaded_docs = callback.on_load(loaded_docs)
            except NotImplementedError:
                continue
        return loaded_docs
        
    @lmql.query
    def _data_gen(
        self,
        text: str,
        test_n: int,
        train_n: int,
        temperature: float = 0.8
    ):
        '''lmql
        sample(temperature=temperature)
            """
            {text}

            Without referring to the text specifically, come up with a list of {test_n} orthogonal questions that could reasonably be answered from the information provided in the above text. \n
            The question should have no foresight of the above text. \n
            """
            for i in range(int(test_n)):
                "<TESTQUESTION> [QUESTION] </TESTQUESTION>"
            """
            Now, come up with a list of {train_n} questions that could reasonably be answered from the information provided in the above text. \n
            IT CANNOT MATCH ANY OF THE PREVIOUS <TESTQUESTION>s. \n
            """
            for i in range(int(train_n)):
                "<TRAINQUESTION> [QUESTION] </TRAINQUESTION>"
        from
            self.model
        where
            STOPS_AT(QUESTION, "?") or STOPS_AT(QUESTION, "\n")
        '''

    @lmql.query
    def _data_rank(
        self,
        question: str,
        answer: str,
        temperature: float = 0.8
    ):
        '''lmql
        sample(temperature=temperature)
            """
            On a scale of 1-10, rank how well the ANSWER answers this QUERY.\n
            QUERY: {question}\n
            ANSWER: {answer}\n
            The rank is: <rank>[RANK]</rank>
            """
        from
            self.model
        where
            RANK in set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        '''

    def extract_questions(self, text):
        train_questions = re.findall('<TRAINQUESTION>(.*?)</TRAINQUESTION>', text, re.DOTALL)
        train_questions = [re.sub('^\s*\d+\.\s*', '', q).strip() for q in train_questions]
        test_questions = re.findall('<TESTQUESTION>(.*?)</TESTQUESTION>', text, re.DOTALL)
        test_questions = [re.sub('^\s*\d+\.\s*', '', q).strip() for q in test_questions]
        return train_questions + test_questions
    
    def rank_questions(
        self,
        question: str,
        answer: str,
        temperature: float = 0.8
    ):
        results = self._data_rank(
            question=question,
            answer=answer,
            temperature=temperature
        )
        rank_raw = results[0].prompt
        try:
            rank = int(re.findall("<rank>(.*?)</rank>", rank_raw)[0])
        except:
            logger.warning(f"Could not parse rank for question: {question}")
            return 1
        return rank

    def generate_questions(
            self,
            text_chunk: str, 
            n_train_questions: int,
            n_val_questions: int,
            n_test_questions: int,
            temperature: float = 0.8
        ) -> Tuple[List[dict], List[dict]]:
        results = self._data_gen(
            text=text_chunk, 
            train_n=n_train_questions,
            test_n=n_test_questions + n_val_questions,
            temperature=temperature
        )
        results = self.extract_questions(results[0].prompt)
        for callback in self.filter_callbacks:
            try:
                train, val, test = callback.on_api_end(results, n_train_questions, n_val_questions, n_test_questions)
            except NotImplementedError:
                continue
        train = [{"question": q, "chunk": text_chunk} for q in train] if train is not None else []
        val = [{"question": q, "chunk": text_chunk} for q in val]
        test = [{"question": q, "chunk": text_chunk} for q in test]
        return train, val, test

    def _load_resume_csv(self, path: Path):
        df = pd.read_csv(path, index_col=0)
        df = df.dropna()
        return df.to_dict(orient="records")
    
    def _create_dict_dataset(
        self,
        n_train_per_doc: int,
        n_val_id_per_doc: int,
        n_val_ood_per_doc: int,
        n_test_id_per_doc: int,
        n_test_ood_per_doc: int,
        temperature: float = 0.8,
        resume: bool = False,
    ):
        train = []
        val_id = []
        val_ood = []
        test_id = []
        test_ood = []
        try:
            if resume:
                train = self._load_resume_csv(self.intermittent_save_path / f"train_temp.csv")
                val_id = self._load_resume_csv(self.intermittent_save_path / f"val_id_temp.csv")
                val_ood = self._load_resume_csv(self.intermittent_save_path / f"val_ood_temp.csv")
                test_id = self._load_resume_csv(self.intermittent_save_path / f"test_id_temp.csv")
                test_ood = self._load_resume_csv(self.intermittent_save_path / f"test_ood_temp.csv")
                logger.info(f"Resuming from {self.intermittent_save_path} - Records loaded:")
                logger.info(f"Train: {len(train)}")
                logger.info(f"Val ID: {len(val_id)}")
                logger.info(f"Val OOD: {len(val_ood)}")
                logger.info(f"Test ID: {len(test_id)}")
                logger.info(f"Test OOD: {len(test_ood)}")
        except FileNotFoundError:
            raise FileNotFoundError("No saved data found. Set resume=False to generate new data.")
        cn_id = int(len(train) / n_train_per_doc)
        cn_ood = int(len(val_ood) / n_val_ood_per_doc)

        if self.intermittent_save_path is not None:
            self.intermittent_save_path.mkdir(parents=True, exist_ok=True)
        for doc in tqdm(self.docs_id[cn_id:self.max_id_samples], desc="Generating in-distribution questions"):
            train_questions, val_questions, test_questions = self.generate_questions(
                doc, n_train_per_doc, n_val_id_per_doc, n_test_id_per_doc, temperature
            )
            train.extend(train_questions)
            val_id.extend(val_questions)
            test_id.extend(test_questions)
            if self.intermittent_save_path is not None:
                pd.DataFrame(train).to_csv(self.intermittent_save_path / f"train_temp.csv")
                pd.DataFrame(val_id).to_csv(self.intermittent_save_path / f"val_id_temp.csv")
                pd.DataFrame(test_id).to_csv(self.intermittent_save_path / f"test_id_temp.csv")
        for doc in tqdm(self.docs_ood[cn_ood:self.max_ood_samples], desc="Generating out-of-distribution questions"):
            _, val_questions, test_questions = self.generate_questions(
                doc, 0, n_val_ood_per_doc, n_test_ood_per_doc, temperature
            )
            try:
                val_ood.extend(test_questions)
                test_ood.extend(test_questions)
                if self.intermittent_save_path is not None:
                    pd.DataFrame(val_ood).to_csv(self.intermittent_save_path / f"val_ood_temp.csv")
                    pd.DataFrame(test_ood).to_csv(self.intermittent_save_path / f"test_ood_temp.csv")
            except:
                val_ood = val_ood[:-(n_val_ood_per_doc+1)]
                test_ood = test_ood[:-(n_test_ood_per_doc+1)]
                pd.DataFrame(val_ood).to_csv(self.intermittent_save_path / f"val_ood_temp.csv")
                pd.DataFrame(test_ood).to_csv(self.intermittent_save_path / f"test_ood_temp.csv")
                logger.warning(f"Issue with last doc, popping {doc}")
                continue
        return train, val_id, val_ood, test_id, test_ood
    
    def create_dataset(
        self,
        tokenizer: transformers.AutoTokenizer,
        n_train_per_doc: int,
        n_val_id_per_doc: int,
        n_val_ood_per_doc: int,
        n_test_id_per_doc: int,
        n_test_ood_per_doc: int,
        temperature: float = 0.8,
        resume: bool = False,
    ):
        train, val_id, val_ood, test_id, test_ood = self._create_dict_dataset(
            n_train_per_doc, n_val_id_per_doc, n_val_ood_per_doc, 
            n_test_id_per_doc, n_test_ood_per_doc, temperature, resume
        )
        train_dataset = RetrievalDataset(tokenizer, train)
        val_id_dataset = RetrievalDataset(tokenizer, val_id)
        val_ood_dataset = RetrievalDataset(tokenizer, val_ood)
        test_id_dataset = RetrievalDataset(tokenizer, test_id)
        test_ood_dataset = RetrievalDataset(tokenizer, test_ood)
        return train_dataset, val_id_dataset, val_ood_dataset, test_id_dataset, test_ood_dataset

    @classmethod
    def from_directory(
        cls,
        id_directory: Path,
        ood_directory: Path,
        model: str = "openai/gpt-3.5-turbo",
        max_id_samples: int = 1000,
        max_ood_samples: int = 100,
        chunk_size: int = 200,
        chunk_ceiling: int = 1000,
        seed: int = 42,
        intermittent_save_path: Path = None,
        filter_callbacks: List[QCGenerationCallback] = [QCGenerationCallback()]

    ):
        docs_id = [f for f in id_directory.iterdir() if not f.is_dir()]
        docs_ood = [f for f in ood_directory.iterdir() if not f.is_dir()]
        return cls(
            docs_id=docs_id,
            docs_ood=docs_ood,
            model=model,
            max_id_samples=max_id_samples, 
            max_ood_samples=max_ood_samples,
            chunk_size=chunk_size,
            chunk_ceiling=chunk_ceiling,
            seed=seed,
            intermittent_save_path=intermittent_save_path,
            filter_callbacks = filter_callbacks
        )
    
    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        dataset_version: str,
        text_column: str = "text",
        train_split: str = "train",
        test_split: str = "test",
        model: str = "openai/gpt-3.5-turbo",
        max_id_samples: int = 1000,
        max_ood_samples: int = 100,
        chunk_size: int = 600,
        chunk_ceiling: int = 1000,
        seed: int = 42,
        intermittent_save_path: Path = None,
        filter_callbacks: List[QCGenerationCallback] = [QCGenerationCallback()]
    ):
        dataset = datasets.load_dataset(dataset_name, dataset_version)
        docs_id = dataset[train_split][text_column]
        docs_ood = dataset[test_split][text_column]
        return cls(
            docs_id=docs_id,
            docs_ood=docs_ood,
            model=model,
            max_id_samples=max_id_samples, 
            max_ood_samples=max_ood_samples,
            chunk_size=chunk_size,
            chunk_ceiling=chunk_ceiling,
            seed=seed,
            intermittent_save_path=intermittent_save_path,
            filter_callbacks = filter_callbacks
        )

    def save_dataset(
        self,
        file_path: Path,
        tokenizer: transformers.AutoTokenizer,
        n_train_per_doc: int,
        n_val_id_per_doc: int,
        n_val_ood_per_doc: int,
        n_test_id_per_doc: int,
        n_test_ood_per_doc: int,
        temperature: float = 0.8,
        resume: bool = False,
    ):
        train_dataset, val_id_dataset, val_ood_dataset, test_id_dataset, test_ood_dataset = self.create_dataset(
            tokenizer, n_train_per_doc, n_val_id_per_doc, n_val_ood_per_doc, 
            n_test_id_per_doc, n_test_ood_per_doc, temperature, resume
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train": train_dataset,
            "val_id": val_id_dataset,
            "val_ood": val_ood_dataset,
            "test_id": test_id_dataset,
            "test_ood": test_ood_dataset
        }, str(file_path))