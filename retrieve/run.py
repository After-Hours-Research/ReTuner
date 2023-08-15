import wandb
from retrieve.data_gen import QuestionGenerator

from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from retrieve.eval import RetrievalEvaluator
from retrieve.models import FullyConnectedHead, HFEmbedder, ResidualHeadWithDropout, TuningModule

from retrieve.store import DataSplit, RetrievalStore
import torch

from retrieve.train import TunerTrainer

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = HFEmbedder('sentence-transformers/all-mpnet-base-v2')

base_path = Path(__file__).parents[1] / "test_data"

qg = QuestionGenerator.from_hf_dataset(
    dataset_name="cnn_dailymail",
    dataset_version="3.0.0",
    text_column="article",
    train_split="train",
    test_split="test",
    max_id_samples=500,
    max_ood_samples=300,
    intermittent_save_path=base_path / "int_temp"
)

qg.save_dataset(
    file_path=base_path / "test_save.pt",
    tokenizer=tokenizer, 
    n_train_per_doc=4, 
    n_val_id_per_doc=2,
    n_val_ood_per_doc=3,
    n_test_id_per_doc=1,
    n_test_ood_per_doc=2,
    resume=False
)

datasets = torch.load(base_path / "test_save.pt")

id_train_ds = datasets[DataSplit.TRAIN.value]

id_val_ds = datasets[DataSplit.VAL_ID.value]
ood_val_ds = datasets[DataSplit.VAL_OOD.value]

id_test_ds = datasets[DataSplit.TEST_ID.value]
ood_test_ds = datasets[DataSplit.TEST_OOD.value]

tuner_model = TuningModule(model, FullyConnectedHead)
tuner_trainer = TunerTrainer(
    project_name="ReTuner_Experiments",
    entity_name="after-hours",
    train_ds=id_train_ds, 
    id_val_ds=id_val_ds,
    ood_val_ds=ood_val_ds,
    id_test_ds=id_test_ds, 
    ood_test_ds=ood_test_ds, 
    tuning_module=tuner_model
)

trained_model = tuner_trainer.fit(log_every_n_steps=1, check_val_every_n_epoch=1, max_epochs=5)

tuner_trainer.test()

# trained_model = TuningModule.load_from_checkpoint("test_project/cc568gi2/checkpoints/epoch=99-step=700.ckpt", original_embedding_model=model, tuning_head_model=FullyConnectedHead)
trained_model.cuda()

rs_ood = RetrievalStore(
    dataset_path = base_path / "test_save.pt",
    store_path= base_path / "test_store",
    model=model,
    tokenizer=tokenizer,
    name=DataSplit.TEST_OOD,
    metric="l2"
)

evalr = RetrievalEvaluator(rs_ood, trained_model)

evalr.evaluate(ood_test_ds, k_retrievals=5)
fig_ood = evalr.plot_performance()

rs_id = RetrievalStore(
    dataset_path = base_path / "test_save.pt",
    store_path= base_path / "test_store",
    model=model,
    tokenizer=tokenizer,
    name=DataSplit.TEST_ID,
    metric="l2"
)

evalr = RetrievalEvaluator(rs_id, trained_model)

evalr.evaluate(id_test_ds, k_retrievals=5)
fig_id = evalr.plot_performance()

wandb.log({"test_id": wandb.Image(fig_id), "test_ood": wandb.Image(fig_ood)})

tuner_trainer.close_wandb()