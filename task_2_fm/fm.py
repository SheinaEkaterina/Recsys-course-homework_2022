from typing import Sequence
import torch
from torch import nn, optim, Tensor


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
    from dataset import ClickDatasetTokenized

    ds = ClickDatasetTokenized("data/processed/train.csv")
    dl = DataLoader(ds, batch_size=4096, shuffle=True)

    model = FactorizationMachineTokenized(ds.cat_sizes, len(ds.numerical), 10)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)

    for _ in range(3):
        num_iter = 0
        avg_loss = 0
        for Xc, Xn, y in tqdm(dl):
            pred = model(Xc, Xn)
            loss = loss_fn(pred, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iter += 1
            avg_loss *= (num_iter - 1) / num_iter
            avg_loss += loss.item() / num_iter
        print(avg_loss)
