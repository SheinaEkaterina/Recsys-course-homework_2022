from typing import Union, Tuple, List, Iterable
from pathlib import Path
import pickle
import numpy as np
from scipy import sparse
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClickDatasetOneHot(Dataset):
    """
    Dataset that uses one-hot encoded data stored as sparse (CSR) arrays.
    """
    def __init__(self, data_dir: Union[Path, str],
                 features: Tuple[str] = (
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
        self.num_features = self.X.shape[1]

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


def scipy_to_torch(x: Iterable[sparse.coo_matrix]) -> torch.sparse_coo_tensor:
    coo = sparse.vstack(x)
    coo = torch.sparse_coo_tensor(
        indices=np.vstack([coo.row, coo.col]),
        values=coo.data,
        size=coo.shape
    )
    return coo


class ClickDatasetTokenized(Dataset):
    """
    Dataset that uses categorical data encoded as tokens {0, 1, .., n}.
    """
    def __init__(
        self,
        file: Union[Path, str],
        categorical: Tuple[str] = \
            ("oaid_hash", "banner_id", "country_id", "os_id", "zone_id"),
        numerical: Tuple[str] = \
            ("weekday", "hour", "log_campaign_clicks"),
        target: str = "clicks"
    ):
        # load dataframe
        df = pd.read_csv(file)
        # categorical features
        self.Xc = np.zeros(shape=(len(df), len(categorical)), dtype=np.int32)
        self.Xc[:, :] = df.loc[:, categorical]
        self.cat_sizes = tuple([len(df[col].unique()) for col in categorical])
        # numerical features
        self.Xn = np.zeros(shape=(len(df), len(numerical)), dtype=np.float32)
        self.Xn[:, :] = df.loc[:, numerical]
        # target
        self.y = df[target].to_numpy().astype(np.float32)

        # save feature names
        self.categorical = categorical
        self.numerical = numerical

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index) -> Tuple[np.ndarray]:
        return self.Xc[index], self.Xn[index], self.y[index]


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    train_dset = ClickDatasetTokenized("data/processed/train.csv")
    train_dloader = DataLoader(train_dset, 1024, shuffle=True)
    for batch in tqdm(train_dloader):
        Xc, Xn, y = batch
