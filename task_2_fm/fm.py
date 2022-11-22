from typing import Sequence
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader


class FactorizationMachineOneHot(nn.Module):
    def __init__(self, num_features: int, emb_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(num_features, 1)
        self.w = nn.Parameter(torch.randn((num_features, emb_dim)) * 0.02)

    def forward(self, X: Tensor) -> Tensor:
        linear = self.fc(X)
        sq_sum = torch.mm(X, self.w)**2
        sum_sq = torch.mm(X**2, self.w**2)
        logits = linear + 0.5 * torch.sum(sq_sum - sum_sq, 1, keepdim=True)
        return logits

    @torch.no_grad()
    def predict(self, dl: DataLoader) -> np.ndarray:
        predictions = []
        for batch in tqdm(dl):
            X = batch[0]
            output = torch.sigmoid(self.forward(X)).cpu().numpy()
            predictions += list(output.ravel())
        return np.array(predictions, dtype=np.float32)


class FactorizationMachineTokenized(nn.Module):
    def __init__(self, categorical_sizes: Sequence[int],
                 n_numerical: int, emb_dim: int = 10):
        super().__init__()
        self.linear_c = CategoricalLinear(categorical_sizes)
        self.linear_n = nn.Linear(n_numerical, 1)
        self.emb_c = Embeddings(categorical_sizes, emb_dim)
        self.emb_n = nn.Parameter(torch.randn((n_numerical, emb_dim)))

    def forward(self, Xc: Tensor, Xn: Tensor) -> Tensor:
        linear = self.linear_c(Xc) + self.linear_n(Xn)

        Xn = Xn.unsqueeze(2)
        vx_c = self.emb_c(Xc)  # one-hot => no multiplication
        vx_n = Xn * self.emb_n
        vx = torch.cat([vx_c, vx_n], 1)
        sq_sum = vx.mean(1)**2
        sum_sq = (vx**2).mean(1)

        logits = linear + 0.5 * torch.sum(sq_sum - sum_sq, 1, keepdim=True)
        return logits

    @torch.no_grad()
    def predict(self, dl: DataLoader) -> np.ndarray:
        predictions = []
        for batch in tqdm(dl):
            Xc, Xn = batch[:2]
            output = torch.sigmoid(self.forward(Xc, Xn)).cpu().numpy()
            predictions += list(output.ravel())
        return np.array(predictions, dtype=np.float32)


class Embeddings(nn.Module):
    def __init__(self, emb_numbers: Sequence[int], emb_dim: int):
        super().__init__()
        self.offsets = torch.tensor(np.cumsum(emb_numbers) - emb_numbers[0],
                                    dtype=torch.int32)
        self.emb = nn.Embedding(np.sum(emb_numbers), emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.offsets
        return self.emb(x)


class CategoricalLinear(Embeddings):
    def __init__(self, categorical_sizes: Sequence[int]):
        super().__init__(categorical_sizes, 1)

    def forward(self, x: Tensor) -> Tensor:
        output = super().forward(x)
        return output.mean(1)


if __name__ == "__main__":
    from tqdm import tqdm
    from sklearn.metrics import log_loss, roc_auc_score
    from dataset import ClickDatasetTokenized

    ds_train = ClickDatasetTokenized("data/processed/train.csv")
    dl_train = DataLoader(ds_train, batch_size=2048, shuffle=True)
    ds_test = ClickDatasetTokenized("data/processed/test.csv")
    dl_test = DataLoader(ds_test, batch_size=4096, shuffle=False)

    model = FactorizationMachineTokenized(
        ds_train.cat_sizes, len(ds_train.numerical), 10)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    for _ in range(10):
        num_iter = 0
        avg_loss = 0
        for Xc, Xn, y in tqdm(dl_train):
            logits = model(Xc, Xn)
            loss = loss_fn(logits, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iter += 1
            avg_loss *= (num_iter - 1) / num_iter
            avg_loss += loss.item() / num_iter
        print(f"Train loss: {avg_loss:.4f}")

        pred = model.predict(dl_test)
        test_loss = log_loss(ds_test.y, pred, eps=1e-7)
        test_auc = roc_auc_score(ds_test.y, pred)
        print(f"Test loss: {test_loss:.4f}, test ROC-AUC: {test_auc: .3f}")
