from typing import Union, Tuple
from pathlib import Path
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")

ONE_HOT_KWARGS = {
    "zone_id": dict(min_frequency=100),
    "banner_id": dict(min_frequency=20),
    "oaid_hash": dict(min_frequency=20),
    "os_id": dict(max_categories=8),
    "country_id": {}
}

def main():
    df_train = pd.read_csv("data/interim/train.csv")
    df_test = pd.read_csv("data/interim/test.csv")
    dataframes = (df_train, df_test)
    paths = (TRAIN_DIR, TEST_DIR)

    # numerical data
    for df, path in zip(dataframes, paths):
        process_datetime(df["date_time"], path / "date_time.npy")
        process_campaign_clicks(df["campaign_clicks"],
                                path / "log_campaign_clicks.npy")
        process_clicks(df["clicks"], path / "clicks.npy")

    # one-hot encoded features
    for name, kwargs in ONE_HOT_KWARGS.items():
        xtrain, xtest = one_hot_wrapper(df_train[name], df_test[name], **kwargs)
        fname = name + ".npy"
        print(name, xtrain.shape, xtest.shape)
        np.save(TRAIN_DIR / fname, xtrain)
        np.save(TEST_DIR / fname, xtest)


def process_datetime(dt_col: pd.Series, save_to: Union[Path, str]):
    dt = pd.to_datetime(dt_col)
    out = pd.DataFrame(index=dt_col.index, columns=("weekday", "hour"))
    out["weekday"] = dt.dt.weekday
    out["hour"] = dt.dt.hour
    np.save(save_to, out.to_numpy())


def process_campaign_clicks(cc_col: pd.Series, save_to: Union[Path, str]):
    log_cc = np.log(cc_col.to_numpy() + 1).reshape(-1, 1)
    np.save(save_to, log_cc)


def process_clicks(clicks: pd.Series, save_to: Union[Path, str]):
    clicks = clicks.to_numpy().reshape(-1, 1)
    clicks = sparse.coo_matrix(clicks)
    np.save(save_to, clicks)


def one_hot_wrapper(train_data: pd.Series, test_data: pd.Series, **kwargs
                    ) -> Tuple[Union[np.ndarray, sparse.spmatrix]]:
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", **kwargs)
    train_1hot = enc.fit_transform(train_data.to_numpy().reshape(-1, 1))
    test_1hot = enc.transform(test_data.to_numpy().reshape(-1, 1))
    return train_1hot, test_1hot


if __name__ == "__main__":
    main()
