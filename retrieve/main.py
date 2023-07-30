from retrieve.data_gen import QuestionGenerator

from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from retrieve.models import HFEmbedder

from retrieve.store import DataSplit, RetrievalStore

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = HFEmbedder('sentence-transformers/all-mpnet-base-v2')

base_path = Path(__file__).parents[1] / "test_data"

# qg = QuestionGenerator.from_directory(
#     id_directory=base_path / "id",
#     ood_directory=base_path / "ood",
#     max_id_samples=3, 
#     max_ood_samples=2,
# )

# qg = QuestionGenerator.from_hf_dataset(
#     dataset_name="cnn_dailymail",
#     dataset_version="3.0.0",
#     text_column="article",
#     train_split="train",
#     test_split="test",
#     max_id_samples=3,
#     max_ood_samples=2,
# )

# qg.save_dataset(
#     file_path=base_path / "test_save.pt",
#     tokenizer=tokenizer, 
#     n_train_per_doc=4, 
#     n_test_id_per_doc=1, 
#     n_test_ood_per_doc=1,
# )

rs_ood = RetrievalStore(
    dataset_path = base_path / "test_save.pt",
    store_path= base_path / "test_store",
    model=model,
    tokenizer=tokenizer,
    name=DataSplit.TEST_OOD,
    metric="l2"
)


rez = rs_ood.query("Who is Hussian?", emb_model=model)

# 1 - Create a Dataset of Q/C using LLM
# 2 - Finetune new head for emb model on Q/Cs
# 3 - Store chunks in vector store (EVALUATION)
# 4 - Question -> New Embedding
# 5 - Retrieve K-Chunks from vector store (EVALUATION)


# BASELINE
# 1 - Create a Dataset of Q/C using LLM
# 2 - Store chunks in vector store (EVALUATION)
# 3 - Question -> Embedding
# 4 - Retrieve K-Chunks from vector store (EVALUATION)


# rs_ood = RetrievalStore(dataset_path, model, store_path, name, metric)
# evaler = Eval(dataset_path, rs_ood, model)
# evalar.evaluate()


# tuner = Tuner(qg, model)
# new_model = tuner.tune()

# rs_test = RetrievalStore.from_questiongen(qg, new_model, path, name, metric)
# rs_ood = RetrievalStore.from_questiongen(qg, new_model, path, name, metric)
# test_evaler = Eval(qg, rs_test, new_model)
# ood_evaler = Eval(qg, rs_ood, new_model)
# test_evaler.evaluate()
# ood_evaler.evaluate()

