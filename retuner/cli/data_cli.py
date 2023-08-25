from omegaconf import DictConfig
from retuner.data_gen import LengthFilterCallback, QuestionGenerator, TrainTestValCallback
from pathlib import Path
from transformers import AutoTokenizer

import hydra

def _get_callbacks():
    return [
        TrainTestValCallback(),
        LengthFilterCallback()
    ]

def _huggingface_datagen(cfg: DictConfig):
    qg = QuestionGenerator.from_hf_dataset(
        dataset_name=cfg.dataset_name,
        dataset_version=cfg.dataset_version,
        model=cfg.model,
        text_column=cfg.text_column,
        train_split=cfg.train_split,
        test_split=cfg.test_split,
        max_id_samples=cfg.max_id_samples,
        max_ood_samples=cfg.max_ood_samples,
        intermittent_save_path=cfg.base_path / cfg.intermittent_save_path,
        filter_callbacks = _get_callbacks()
    )
    return qg

def _directory_datagen(cfg: DictConfig):
    qg = QuestionGenerator.from_directory(
        id_directory=Path(cfg.id_directory),
        ood_directory=Path(cfg.ood_directory),
        model=cfg.model,
        max_id_samples=cfg.max_id_samples,
        max_ood_samples=cfg.max_ood_samples,
        intermittent_save_path=cfg.base_path / cfg.intermittent_save_path,
        filter_callbacks = _get_callbacks()
    )
    return qg


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def data(cfg: DictConfig):
    cfg = cfg["data"]
    cfg.base_path = Path(cfg.base_path)
    if cfg.from_hf:
        qg = _huggingface_datagen(cfg)
    else:
        qg = _directory_datagen(cfg)
    tokenizer  = AutoTokenizer.from_pretrained(cfg.tokenizer)
    qg.save_dataset(
        file_path=cfg.base_path / cfg.ds_file_path,
        tokenizer=tokenizer, 
        n_train_per_doc=cfg.n_train_per_doc, 
        n_val_id_per_doc=cfg.n_val_id_per_doc,
        n_val_ood_per_doc=cfg.n_val_ood_per_doc,
        n_test_id_per_doc=cfg.n_test_id_per_doc,
        n_test_ood_per_doc=cfg.n_test_ood_per_doc,
        resume=cfg.resume
    )

if __name__ == "__main__":
    data()