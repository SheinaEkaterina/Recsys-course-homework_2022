from pathlib import Path
import pickle
from typing import Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


Array = Union[pd.Series, np.ndarray]
ONE_HOT_KWARGS = {
    "zone_id": dict(min_frequency=100),
    "banner_id": dict(min_frequency=20),
    "os_id": dict(max_categories=8),
    "country_id": {}
}


def main(data_dir: str | Path):
    # check data_dir
    data_dir = Path(data_dir)
    save_to = data_dir / "processed"

    # read and preprocess data
    data = pd.read_csv(data_dir / "raw/data.csv")
    train, test = preprocess_and_split(data)

    # get final features
    train_data, test_data = feature_engineering(train, test)
    save_binary(train_data, save_to / "train.pkl")
    save_binary(test_data, save_to / "test.pkl")


def preprocess_and_split(df: pd.DataFrame) -> pd.DataFrame:
    # drop useless data
    df = df.drop(columns=["oaid_hash", "campaign_clicks", "impressions"])
    mask = df["banner_id"] == df["banner_id0"]
    df = df[mask]

    # perform train-test split by last date
    dt_col = pd.to_datetime(df["date_time"])
    last_date = dt_col.max().date()
    mask = dt_col.dt.date == last_date
    train = df[~mask]
    test = df[mask]

    return train, test


def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame
                        ) -> tuple[dict]:
    # output dictionaries: column_name -> array (possibly sparse)
    train_data, test_data = {}, {}

    # extract weekday and hour
    for df, dct in zip((df_train, df_test), [train_data, test_data]):
        dt = pd.to_datetime(df["date_time"])
        dct["weekday"] = dt.dt.weekday.to_numpy()
        dct["hour"] = dt.dt.hour.to_numpy()

    # encode categorical features
    for col, kwargs in ONE_HOT_KWARGS.items():
        kwargs["handle_unknown"] = "infrequent_if_exist"
        enc = OneHotEncoder(**kwargs)
        train_data[col] = enc.fit_transform(
            df_train[col].to_numpy().reshape(-1, 1))
        test_data[col] = enc.transform(df_test[col].to_numpy().reshape(-1, 1))

        if col == "banner_id":
            for df, dct in zip((df_train, df_test), [train_data, test_data]):
                dct["banner_id1"] = enc.transform(
                    df["banner_id1"].to_numpy().reshape(-1, 1))

    # keep some columns unchanged
    for df, dct in zip((df_train, df_test), [train_data, test_data]):
        for col in ("coeff_sum0", "coeff_sum1", "g0", "g1"):
            dct[col] = df[col].to_numpy().reshape(-1, 1)

    return train_data, test_data


def save_binary(obj: Any, path: Path | str) -> None:
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Process data from `src` directory")
    parser.add_argument("src", type=str, help="location of `data` directory")
    args = parser.parse_args()

    main(args.src)
