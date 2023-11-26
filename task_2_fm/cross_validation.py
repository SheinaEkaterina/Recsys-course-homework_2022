from pathlib import Path
from typing import Sequence, Union
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch import optim, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from fm import FactorizationMachineTokenized
from dataset import ClickDatasetTokenized


def main(csv_file: Union[Path, str], emb_dim: int, n_splits: int,
         batch_size: int, max_epochs: int):
    ds = ClickDatasetTokenized(
        file=csv_file,
        categorical=("oaid_hash", "banner_id", "country_id",
                     "os_id", "zone_id"),
        numerical=("weekday", "hour", "log_campaign_clicks"),
        target="clicks"
    )

    skf = StratifiedKFold(n_splits=n_splits)
    for train_idx, val_idx in skf.split(np.zeros(len(ds)), ds.y):
        ds_train = Subset(ds, train_idx)
        ds_val = Subset(ds, val_idx)
        dl_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
        dl_val = DataLoader(ds_val, max(batch_size, 4096), shuffle=False,
                            num_workers=4)

        model = FM(ds.cat_sizes, len(ds.numerical), emb_dim=emb_dim)
        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(model, dl_train, dl_val)


class FM(pl.LightningModule):
    """
    Pytorch Lightning wrapper for FM model.
    """
    def __init__(self, categorical_sizes: Sequence[int],
                 n_numerical: int, emb_dim: int = 10):
        super().__init__()
        self.model = FactorizationMachineTokenized(
            categorical_sizes, n_numerical, emb_dim)

    def forward(self, Xc: Tensor, Xn: Tensor) -> Tensor:
        return self.model(Xc, Xn)

    def predict(self, dl: DataLoader) -> np.ndarray:
        return self.model.predict(dl)

    @staticmethod
    def log_loss(logits: Tensor, target: Tensor):
        return F.binary_cross_entropy_with_logits(logits, target)

    def _forward_step(self, batch):
        Xc, Xn, y = batch
        logits = self.forward(Xc, Xn)
        loss = self.log_loss(logits.squeeze(), y)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._forward_step(batch)

    def training_epoch_end(self, outputs):
        avg_loss = np.mean([d["loss"].item() for d in outputs])
        self.log("train_loss", avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([d["loss"] for d in outputs])
        self.log("val_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("src", type=str, help="Path to processed dataset file")
    parser.add_argument("dim", type=int, help="Embedding dimension")
    parser.add_argument("-k", "--n_splits", type=int, help="Number of splits",
                        default=4)
    parser.add_argument("--batch_size", type=int, help="Batch size",
                        default=2048)
    parser.add_argument("--max_epochs", type=int, help="Number of epochs",
                        default=5)
    args = parser.parse_args()

    main(csv_file=args.src, emb_dim=args.dim, n_splits=args.n_splits,
         batch_size=args.batch_size, max_epochs=args.max_epochs)