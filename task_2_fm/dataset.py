from typing import Union, Tuple, List
from pathlib import Path
import pickle
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset


class ClickDataset(Dataset):
    def __init__(self, data_dir: Union[Path, str],
                 features: tuple[str] = (
                    "oaid_hash", "banner_id", "country_id", "date_time",
                    "os_id", "log_campaign_clicks", "zone_id"),
                 target: str = "clicks"):
        # load feature arrays
        data_dir = Path(data_dir)
        arrays = []
        columns = []
        for name in features:
            arr = load_array(data_dir / f"{name}.pkl")
            if not sparse.issparse(arr):
                arr = sparse.coo_matrix(arr)
            arrays.append(arr)
            columns.append((name, arr.shape[1]))

        # construct X and y sparse arrays
        self.X = sparse.hstack(arrays, format="csr", dtype=np.float32)
        self.columns = tuple(columns)
        self.y = load_array(data_dir / f"{target}.pkl").astype(np.float32)

    def __getitem__(self, index) -> Tuple[sparse.spmatrix, float]:
       return self.X[index].tocoo(), float(self.y[index].todense())

    def __len__(self) -> int:
        return self.y.shape[0]


def load_array(file: Union[Path, str]):
    with open(file, "rb") as fin:
        return pickle.load(fin)


def collate_fn(data: List[Tuple[sparse.spmatrix]]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = zip(*data)
    X = scipy_to_torch(X)
    y = torch.tensor(np.array(y, dtype=np.float32))
    return X, y


def scipy_to_torch(x: Tuple[sparse.coo_matrix]) -> torch.sparse_csr_tensor:
    coo = sparse.vstack(x)
    coo = torch.sparse_coo_tensor(
        indices=np.vstack([coo.row, coo.col]),
        values=coo.data,
        size=coo.shape
    )
    return coo.to_sparse_csr()


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    train_dset = ClickDataset("data/processed/train")
    train_dloader = DataLoader(train_dset, 1024, collate_fn=collate_fn, shuffle=True)
    for batch in tqdm(train_dloader):
        X, y = batch
