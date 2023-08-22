from pathlib import Path
from pytorch_lightning import Trainer
import torch

from torch.utils.data import DataLoader
from retuner.data_gen import RetrievalDataset
from retuner.eval import RetrievalEvaluator

from transformers import AutoModel, AutoTokenizer

from retuner.models import  TuningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback

import wandb

from retuner.store import DataSplit, RetrievalStore

def collate_fn(batch):
    questions, chunks = zip(*batch)

    # Concatenate along the batch dimension and squeeze the second dimension
    question_inputs = {key: torch.cat([q[key] for q in questions], dim=0) for key in questions[0]}
    chunk_inputs = {key: torch.cat([c[key] for c in chunks], dim=0) for key in chunks[0]}

    return chunk_inputs, question_inputs

class TunerTrainer:
    def __init__(
            self,
            project_name: str,
            entity_name: str,
            dataset_path: Path,
            tuning_module: TuningModule,
            batch_size: int = 32,
        ):
        self.tuning_module = tuning_module
        self.batch_size = batch_size

        train_ds, id_val_ds, ood_val_ds, id_test_ds, ood_test_ds = self.get_dataset(dataset_path)

        self.train_data_loader = self.get_dataloader(train_ds)

        self.id_val_data_loader = self.get_dataloader(id_val_ds, shuffle=False)
        self.ood_val_data_loader = self.get_dataloader(ood_val_ds, shuffle=False)

        self.id_test_data_loader = self.get_dataloader(id_test_ds, shuffle=False)
        self.ood_test_data_loader = self.get_dataloader(ood_test_ds, shuffle=False)

        self.project_name = project_name
        self.entity_name = entity_name

    def get_dataset(self, dataset_path):
        datasets = torch.load(dataset_path)
        self.datasets = datasets
        id_train_ds = datasets[DataSplit.TRAIN.value]
        id_val_ds = datasets[DataSplit.VAL_ID.value]
        ood_val_ds = datasets[DataSplit.VAL_OOD.value]
        id_test_ds = datasets[DataSplit.TEST_ID.value]
        ood_test_ds = datasets[DataSplit.TEST_OOD.value]
        return id_train_ds, id_val_ds, ood_val_ds, id_test_ds, ood_test_ds



    def get_dataloader(self, dataset: RetrievalDataset, shuffle: bool = True):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=4
        )

    def fit(self, *args, **kwargs):
        wandb_logger = WandbLogger(
            project=self.project_name, 
            entity=self.entity_name,
        )
        wandb_logger.watch(self.tuning_module, log="all")
        es_callback = EarlyStopping(monitor="avg_val_id_loss", min_delta=kwargs.pop("es_min_delta", 0.0001), patience=kwargs.pop("es_patience", 3), verbose=False, mode="min")
        trainer = Trainer(logger=wandb_logger, callbacks=[es_callback, *kwargs.pop("callbacks", [])], *args, **kwargs)
        trainer.fit(
            self.tuning_module, 
            self.train_data_loader,
            val_dataloaders=[self.id_val_data_loader, self.ood_val_data_loader],
        )
        self.trainer = trainer
        self.model = trainer.model
        return trainer.model
    
    def test(self) -> dict:
        loss_results = self.trainer.test(self.model, [self.id_test_data_loader, self.ood_test_data_loader])
        result = {
            "id_loss": loss_results[0],
            "ood_loss": loss_results[1]
        }
        wandb.log(result)
        return result
    
    def evaluate_retrieval(
        self,
        dataset_path: Path,
        store_path: Path,
        base_model: AutoModel,
        tuned_model: AutoModel,
        tokenizer: AutoTokenizer,
        name: DataSplit,
        figure_name: str,
        metric: str = "l2",
        k: int = 5,
    ):
        store = RetrievalStore(
            dataset_path = dataset_path,
            store_path= store_path,
            model=base_model,
            tokenizer=tokenizer,
            name=name,
            metric=metric
        )
        evalr = RetrievalEvaluator(store, tuned_model)
        evalr.evaluate(self.datasets[name.value], k_retrievals=k)
        fig = evalr.plot_performance()
        wandb.log({figure_name: wandb.Image(fig)})

    def close_wandb(self):
        wandb.finish()


class RetrievalEvalCallback(Callback):
    def __init__(
        self,
        retrieval_store: RetrievalStore,
        figure_name: str,
        k: int = 5,
        every_n_epochs: int = 1,
        dataset: RetrievalDataset = None
    ):
        self.retrieval_store = retrieval_store
        self.figure_name = figure_name
        self.k = k
        self.every_n_epochs = every_n_epochs
        self.dataset = dataset

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            evalr = RetrievalEvaluator(self.retrieval_store, pl_module)
            results = evalr.evaluate(self.dataset, k_retrievals=self.k)
            if wandb.run is not None:
                wandb.log({self.figure_name: wandb.Image(evalr.plot_performance())})
                for k, v in results.items():
                    wandb.log({f"{self.figure_name}_{k}": v})
            