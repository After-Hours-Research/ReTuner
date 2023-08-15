from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
from retrieve.data_gen import RetrievalDataset

from retrieve.models import HFEmbedder, InitialValCallback, TuningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

def collate_fn(batch):
    questions, chunks = zip(*batch)

    # Concatenate along the batch dimension and squeeze the second dimension
    question_inputs = {key: torch.cat([q[key] for q in questions], dim=0) for key in questions[0]}
    chunk_inputs = {key: torch.cat([c[key] for c in chunks], dim=0) for key in chunks[0]}

    return question_inputs, chunk_inputs

class TunerTrainer:
    def __init__(
            self,
            project_name: str,
            entity_name: str,
            train_ds: RetrievalDataset,
            id_val_ds: RetrievalDataset,
            ood_val_ds: RetrievalDataset,
            id_test_ds: RetrievalDataset,
            ood_test_ds: RetrievalDataset,
            tuning_module: TuningModule,
            batch_size: int = 32,
        ):
        self.tuning_module = tuning_module
        self.batch_size = batch_size

        self.train_data_loader = self.get_dataloader(train_ds)

        self.id_val_data_loader = self.get_dataloader(id_val_ds, shuffle=False)
        self.ood_val_data_loader = self.get_dataloader(ood_val_ds, shuffle=False)

        self.id_test_data_loader = self.get_dataloader(id_test_ds, shuffle=False)
        self.ood_test_data_loader = self.get_dataloader(ood_test_ds, shuffle=False)

        self.project_name = project_name
        self.entity_name = entity_name

    def get_dataloader(self, dataset: RetrievalDataset, shuffle: bool = True):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=4
        )

    def fit(self, *args, **kwargs):
        wandb_logger = WandbLogger(project=self.project_name, entity=self.entity_name)
        wandb.init(project=self.project_name, config=self.tuning_module.describe())
        wandb.watch(self.tuning_module, log="all")
        es_callback = EarlyStopping(monitor="avg_val_id_loss", min_delta=0.0001, patience=3, verbose=False, mode="min")
        trainer = Trainer(logger=wandb_logger, callbacks=[es_callback], *args, **kwargs)
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

    def close_wandb(self):
        wandb.finish()
