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
        Xd = X.to_dense()  # :'(
        sq_sum = (Xd.unsqueeze(2) * self.w).sum(1)**2
        sum_sq = (Xd.unsqueeze(2)**2 * self.w**2).sum(1)
        logits = linear + 0.5 * torch.sum(sq_sum - sum_sq, 1, keepdim=True)
        return torch.sigmoid(logits)


class FactorizationMachineTokenized(nn.Module):
    def __init__(self, categorical_sizes: Sequence[int],
                 n_numerical: int, emb_dim: int = 10):
        super().__init__()
        self.linear_c = CategoricalLinear(categorical_sizes)
        self.linear_n = nn.Linear(n_numerical, 1)

    def forward(self, Xc: Tensor, Xn: Tensor) -> Tensor:
        out_c = self.linear_c(Xc)
        out_n = self.linear_n(Xn)
        return torch.sigmoid(out_n + out_c)

    @torch.no_grad()
    def predict(self, dl: DataLoader) -> np.ndarray:
        predictions = []
        for batch in tqdm(dl):
            Xc, Xn = batch[:2]
            output = self.forward(Xc, Xn).cpu().numpy()
            predictions += list(output.ravel())
        return np.array(predictions, dtype=np.float32)


class CategoricalLinear(nn.Module):
    def __init__(self, categorical_sizes: Sequence[int]):
        super().__init__()
        self.embeddings = [nn.Embedding(sz, 1) for sz in categorical_sizes]

    def forward(self, x: Tensor) -> Tensor:
        output = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        output = torch.cat(output, 1)
        return output.sum(1, keepdims=True)


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from sklearn.metrics import log_loss, roc_auc_score
    from dataset import ClickDatasetTokenized

    ds_train = ClickDatasetTokenized("data/processed/train.csv")
    dl_train = DataLoader(ds_train, batch_size=4096, shuffle=True)
    ds_test = ClickDatasetTokenized("data/processed/test.csv")
    dl_test = DataLoader(ds_test, batch_size=4096, shuffle=False)

    model = FactorizationMachineTokenized(ds_train.cat_sizes, len(ds_train.numerical), 10)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)

    for _ in range(10):
        num_iter = 0
        avg_loss = 0
        for Xc, Xn, y in tqdm(dl_train):
            pred = model(Xc, Xn)
            loss = loss_fn(pred, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iter += 1
            avg_loss *= (num_iter - 1) / num_iter
            avg_loss += loss.item() / num_iter
        print(f"Train loss: {avg_loss:.4f}")

        pred = model.predict(dl_test)
        test_loss = log_loss(ds_test.y, pred)
        test_auc = roc_auc_score(ds_test.y, pred)
        print(f"Test loss: {test_loss:.4f}, test ROC-AUC: {test_auc: .3f}")
