import torch
from torch import nn, optim, Tensor


class FactorizationMachine(nn.Module):
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import ClickDataset, collate_fn
    from sklearn.metrics import log_loss

    ds = ClickDataset("data/processed/train")
    dl = DataLoader(ds, batch_size=1024, collate_fn=collate_fn, shuffle=True)

    model = FactorizationMachine(ds.num_features, 10)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    num_iter = 0
    for _ in range(3):
        for X, y in dl:
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iter += 1
            print(num_iter, loss.item())
            # if num_iter % 10 == 0:
            #     print(loss.item())
            # else:
            #     print('.', end='')
