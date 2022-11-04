from pathlib import Path
import pandas as pd


def load_raw(data_dir: Path | str = "../data") -> pd.DataFrame:
    "Load dataset for HW1"
    path = Path(data_dir) / "data.csv"
    df = pd.read_csv(path).drop(columns=[
        "oaid_hash", "banner_id0", "banner_id1", "rate0", "rate1",
        "g0", "g1", "coeff_sum0", "coeff_sum1"])
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df


def last_day_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    "Move data from the last day to a separate dataframe"
    last_day = df["date_time"].max().date()
    mask = df["date_time"].dt.date == last_day
    assert mask.any()
    return df[~mask].copy(), df[mask].copy()  # train, test
