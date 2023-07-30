from retrieve.data_gen import QuestionGenerator

from pathlib import Path

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
base_path = Path(__file__).parents[1] / "test_data"

# qg = QuestionGenerator.from_directory(
#     id_directory=base_path / "id",
#     ood_directory=base_path / "ood",
#     max_id_samples=3, 
#     max_ood_samples=2,
# )

qg = QuestionGenerator.from_hf_dataset(
    dataset_name="cnn_dailymail",
    dataset_version="3.0.0",
    text_column="article",
    train_split="train",
    test_split="test",
    max_id_samples=3,
    max_ood_samples=2,
)

qg.save_dataset(
    file_path=base_path / "test_save.pt",
    tokenizer=tokenizer, 
    n_train_per_doc=4, 
    n_test_id_per_doc=1, 
    n_test_ood_per_doc=1,
)

breakpoint()