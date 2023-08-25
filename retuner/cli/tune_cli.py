from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from retuner.models import AttentionHead, Baseline, FCBottleneckHeadWithDropout, FullyConnectedHead, HFEmbedder, ResidualAttentionHead, ResidualHeadWithDropout, TuningModule
from retuner.store import DataSplit, RetrievalStore
from retuner.train import RetrievalEvalCallback, TunerTrainer

from transformers import AutoTokenizer
import torch

HEAD_DICT = {
    "fully-connected": FullyConnectedHead,
    "bottleneck-dropout": FCBottleneckHeadWithDropout,
    "residual-dropout": ResidualHeadWithDropout,
    "attention": AttentionHead,
    "baseline": Baseline,
    'residual-attention': ResidualAttentionHead
}


def get_store(store_id: DataSplit, cfg: DictConfig, base_model: HFEmbedder, tokenizer: AutoTokenizer):
    return RetrievalStore(
            dataset_path=Path(cfg.data.base_path) / cfg.data.ds_file_path,
            store_path=Path(cfg.data.base_path) / cfg.experiment.store_path / store_id.value,
            model=base_model,
            tokenizer=tokenizer,
            name=store_id,
            metric=cfg.experiment.retrieval_metric,
    )

    

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def tune(cfg: DictConfig):
    base_model = HFEmbedder(cfg.model.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    tuner_model = TuningModule(
        base_model, HEAD_DICT[cfg.model.head], 
        loss_name=cfg.model.loss,
        margin=cfg.model.margin,
        epochs=cfg.training.max_epochs,
        adding_noise=cfg.model.adding_noise,
        noise_max=cfg.model.noise_max,
        noise_min=cfg.model.noise_min
    )
    
    tuner_trainer = TunerTrainer(
        project_name=cfg.experiment.project_name,
        entity_name=cfg.experiment.entity_name,
        dataset_path=Path(cfg.data.base_path) / cfg.data.ds_file_path,
        tuning_module=tuner_model,
        batch_size=cfg.training.batch_size,
    )
    friendly_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.experiment.project_name, 
        entity=cfg.experiment.entity_name,
        config=friendly_config
    )
    if cfg.model.head != "baseline":
        id_store = get_store(DataSplit.VAL_ID, cfg, base_model, tokenizer)
        ood_store = get_store(DataSplit.VAL_OOD, cfg, base_model, tokenizer)
        trained_model = tuner_trainer.fit(
            log_every_n_steps=cfg.training.log_every_n_steps, 
            check_val_every_n_epoch=cfg.training.check_val_every_n_epoch, 
            max_epochs=cfg.training.max_epochs,
            es_patience=cfg.training.es_patience,
            es_min_delta=cfg.training.es_min_delta,
            callbacks=[
                RetrievalEvalCallback(
                    id_store, 
                    "id_retrieval", 
                    every_n_epochs=cfg.training.retrieve_every_n_epochs, 
                    dataset=tuner_trainer.datasets[DataSplit.VAL_ID.value]
                ), 
                RetrievalEvalCallback(
                    ood_store, 
                    "ood_retrieval", 
                    every_n_epochs=cfg.training.retrieve_every_n_epochs, 
                    dataset=tuner_trainer.datasets[DataSplit.VAL_OOD.value]
                ),
            ]
        )
        tuner_trainer.test()
        trained_model.to("cuda")
    tuner_trainer.evaluate_retrieval(
        dataset_path=Path(cfg.data.base_path) / cfg.data.ds_file_path,
        store_path=Path(cfg.data.base_path) / cfg.experiment.store_path / "id",
        base_model=base_model,
        tuned_model=trained_model if cfg.model.head != "baseline" else base_model,
        tokenizer=tokenizer,
        name=DataSplit.TEST_ID,
        metric=cfg.experiment.retrieval_metric,
        k=cfg.experiment.k_retrievals,
        figure_name="test_id"
    )
    tuner_trainer.evaluate_retrieval(
        dataset_path=Path(cfg.data.base_path) / cfg.data.ds_file_path,
        store_path=Path(cfg.data.base_path) / cfg.experiment.store_path / "ood",
        base_model=base_model,
        tuned_model=trained_model if cfg.model.head != "baseline" else base_model,
        tokenizer=tokenizer,
        name=DataSplit.TEST_OOD,
        metric=cfg.experiment.retrieval_metric,
        k=cfg.experiment.k_retrievals,
        figure_name="test_ood"
    )
    tuner_trainer.close_wandb()


if __name__ == "__main__":
    tune()