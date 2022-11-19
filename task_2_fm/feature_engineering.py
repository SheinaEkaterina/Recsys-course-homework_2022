from collections import defaultdict
import pickle
from typing import Optional, Union, Tuple
from pathlib import Path
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")

ONE_HOT_KWARGS = {
    "zone_id": dict(min_frequency=100),
    "banner_id": dict(min_frequency=20),
    "oaid_hash": dict(min_frequency=20),
    "os_id": dict(max_categories=8),
    "country_id": {}
}

def feature_engineering_sparse(input_dir: Union[Path, str],
                               output_dir: Union[Path, str]):
    df_train = pd.read_csv(Path(input_dir) / "train.csv")
    df_test = pd.read_csv(Path(input_dir) / "test.csv")
    dataframes = (df_train, df_test)
    output_dir = Path(output_dir)
    paths = (output_dir / "train", output_dir / "test")
    for path in paths:
        if not path.exists():
            path.mkdir()

    # numerical data
    for df, path in zip(dataframes, paths):
        process_datetime(df["date_time"], path / "date_time.pkl")
        process_campaign_clicks(df["campaign_clicks"],
                                path / "log_campaign_clicks.pkl")
        process_clicks(df["clicks"], path / "clicks.pkl")

    # one-hot encoded features
    for name, kwargs in ONE_HOT_KWARGS.items():
        xtrain, xtest = one_hot_wrapper(df_train[name], df_test[name], **kwargs)
        fname = name + ".pkl"
        print(name, xtrain.shape, xtest.shape)
        save_array(paths[0] / fname, xtrain)
        save_array(paths[1] / fname, xtest)


def feature_engineering_dense(input_dir: Union[Path, str],
                              output_dir: Union[Path, str]):
    output_dir = Path(output_dir)
    assert output_dir.exists()

    print("Reading data...")
    df_train = pd.read_csv(Path(input_dir) / "train.csv")
    df_test = pd.read_csv(Path(input_dir) / "test.csv")
    dataframes = (df_train, df_test)
 
    # numerical features
    print("Transforming numerical features...")
    for df in dataframes:
        dt = pd.to_datetime(df["date_time"])
        df["weekday"] = dt.dt.weekday
        df["hour"] = dt.dt.hour
        df["log_campaign_clicks"] = np.log(df["campaign_clicks"] + 1)
        df.drop(columns=["date_time", "campaign_clicks", "impressions"],
                inplace=True)

    # categorical features
    print("Transforming categorical features...")
    for name, kwargs in ONE_HOT_KWARGS.items():
        tok = get_tokenizer(df_train[name], **kwargs)
        for df in dataframes:
            transformed = np.zeros(len(df), dtype=np.bool8)
            for value, token in tqdm(tok.items()):
                mask = df[name] == value
                df.loc[mask, name] = token
                transformed[mask] = True
            df.loc[~transformed, name] = tok.default_factory()

    # export results
    df_train.to_csv(output_dir / "train.csv")
    df_test.to_csv(output_dir / "test.csv")


def save_array(save_to: Union[Path, str], arr: Union[np.ndarray, sparse.spmatrix]):
    with open(save_to, "wb") as fout:
        pickle.dump(arr, fout)


def process_datetime(dt_col: pd.Series, save_to: Union[Path, str]):
    dt = pd.to_datetime(dt_col)
    out = pd.DataFrame(index=dt_col.index, columns=("weekday", "hour"))
    out["weekday"] = dt.dt.weekday
    out["hour"] = dt.dt.hour
    save_array(save_to, out.to_numpy())


def process_campaign_clicks(cc_col: pd.Series, save_to: Union[Path, str]):
    log_cc = np.log(cc_col.to_numpy() + 1).reshape(-1, 1)
    save_array(save_to, log_cc)


def process_clicks(clicks: pd.Series, save_to: Union[Path, str]):
    clicks = clicks.to_numpy().reshape(-1, 1)
    clicks = sparse.csr_matrix(clicks)
    save_array(save_to, clicks)


def one_hot_wrapper(train_data: pd.Series, test_data: pd.Series, **kwargs
                    ) -> Tuple[Union[np.ndarray, sparse.spmatrix]]:
    "Use one-hot encoder from sklearn to transform train and test data."
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", **kwargs)
    train_1hot = enc.fit_transform(train_data.to_numpy().reshape(-1, 1))
    test_1hot = enc.transform(test_data.to_numpy().reshape(-1, 1))
    return train_1hot, test_1hot


def get_tokenizer(data: pd.Series,
                  min_frequency: Optional[int] = None,
                  max_categories: Optional[int] = None
                  ) -> defaultdict: 
    "One-hot encoding without one-hot encoding"
    vc = data.value_counts(sort=True)  # series: value -> count
    if min_frequency is not None:
        if 0 < min_frequency < 1:
            min_frequency *= len(data)
        vc = vc[vc >= min_frequency]
    if max_categories is not None:
        vc = vc.iloc[:max_categories - 1]  # + default value
    n = len(vc)
    return defaultdict(lambda: n, zip(vc.index, range(n)))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Transform train and test data stored in csv files")
    parser.add_argument("src", type=str,
                        help="source directory with train.csv and test.csv")
    parser.add_argument("dst", type=str,
                        help="output directory")
    parser.add_argument("--onehot", action="store_true", default=False,
                        help="whether to perform one-hot encoding")
    args = parser.parse_args()

    fe_func = feature_engineering_sparse
    if args.onehot:
        feature_engineering_sparse(args.src, args.dst)
    else:
        feature_engineering_dense(args.src, args.dst)
