from copy import deepcopy
import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame, drop_ids: bool = True
                        ) -> pd.DataFrame:
    df_new = deepcopy(df)
    df_new.drop("impressions", axis=1, inplace=True)
    process_date_time(df_new)
    process_os_id(df_new)
    process_country_id(df_new)
    process_campaign_clicks(df_new)
    if drop_ids:
        df_new.drop(columns=["zone_id", "banner_id"], inplace=True)
    return df_new


# all following functions modify dataframe inplace

def process_date_time(df: pd.DataFrame) -> None:
    df["weekday"] = df["date_time"].dt.weekday
    df["hour"] = df["date_time"].dt.hour
    df.drop("date_time", axis=1, inplace=True)


def process_os_id(df: pd.DataFrame) -> None:
    one_hot_n_first(df, "os_id", 6)


def process_country_id(df: pd.DataFrame) -> None:
    one_hot_n_first(df, "country_id", None)


def one_hot_n_first(df: pd.DataFrame, col_name: str, n: None | int) -> None:
    if n is not None:
        mask = df[col_name] > n
        df[col_name][mask] = n + 1
    dummies = pd.get_dummies(df[col_name])
    if n is None:
        n = dummies.shape[1]
    else:
        n += 2
    col_names = [f"{col_name}_{i}" for i in range(n)]
    df[col_names] = dummies
    df.drop(col_name, axis=1, inplace=True)


def process_campaign_clicks(df: pd.DataFrame) -> None:
    df["log_campaign_clicks"] = np.log(df["campaign_clicks"] + 1)
    df.drop("campaign_clicks", axis=1, inplace=True)
