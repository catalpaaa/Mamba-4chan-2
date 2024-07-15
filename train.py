import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, random_split
from torchdata.stateful_dataloader import StatefulDataLoader

from model import mamba_4chan_2
from model_config import ssm_780m_config

# load base mamba model
model = mamba_4chan_2(ssm_780m_config())
state_dict = torch.load("pytorch_model.bin")
model.load_state_dict(state_dict)

# load mamba 4chan 2 model
# model = mamba_4chan_2.load_from_checkpoint("path_to.ckpt")

model.learning_rate = 1e-7


class pol_dataset(Dataset):
    def __init__(
        self,
        memmap_path: str,
        context_size: int = 2048,
        eos_token: int = 0,
        stride: int = 2047,
    ):
        self.memmap = np.memmap(memmap_path, dtype="uint16")
        self.context_size = context_size
        self.eos_token = eos_token
        self.stride = stride

    def __len__(self):
        return ((len(self.memmap) - self.context_size) // self.stride) + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_size

        if end > len(self.memmap):
            padding_size = end - len(self.memmap)
            data = np.concatenate(
                (
                    self.memmap[start:],
                    np.full(padding_size, self.eos_token, dtype="uint16"),
                )
            )
        else:
            data = self.memmap[start:end]

        return torch.tensor(data, dtype=torch.long)


class pol_data_module(pl.LightningDataModule):
    def __init__(
        self,
        memmap_path: str,
        batch_size: int,
        num_workers: int,
        context_size: int = 2048,
        eos_token: int = 0,
        stride: int = 2047,
        train_val_ratio: float = 0.95,
    ):
        super().__init__()
        self.memmap_path = memmap_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_size = context_size
        self.eos_token = eos_token
        self.stride = stride
        self.train_val_ratio = train_val_ratio

    def setup(self, stage: str = None):
        dataset = pol_dataset(
            self.memmap_path, self.context_size, self.eos_token, self.stride
        )

        train_size = int(len(dataset) * self.train_val_ratio)
        val_size = len(dataset) - train_size

        self.train_set, self.val_set = random_split(
            dataset, [train_size, val_size], torch.Generator().manual_seed(943264)
        )

    def train_dataloader(self):
        return StatefulDataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return StatefulDataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


data = pol_data_module(memmap_path="dataset.dat", batch_size=2, num_workers=16)

trainer = pl.Trainer(
    callbacks=[
        ModelCheckpoint(
            dirpath="models/",
            save_top_k=-1,
            every_n_train_steps=500,
        ),
    ],
    logger=pl_loggers.WandbLogger(
        project="Mamba 4chan 2 780m",
        name="Fine-tuning",
    ),
    precision="bf16-mixed",
    max_epochs=1,
    accumulate_grad_batches=128,
    num_sanity_val_steps=0,
)

trainer.fit(model, data)
# trainer.fit(model, data, ckpt_path="ckpt to resume training")

trainer.save_checkpoint("mamba_4chan_2_780m.ckpt")
trainer.save_checkpoint(
    "mamba_4chan_2_780m_weights_only.ckpt",
    weights_only=True,
)
