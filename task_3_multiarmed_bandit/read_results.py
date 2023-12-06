import os
import json
import pandas as pd


DATA_DIR = "./results"

def main():
    # find files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    files.sort(key=get_index)

    # read data
    columns = ("regret", "rounds", "total_banners")
    columns_ucb = columns + ("ni_min", "gamma")
    columns_thompson = columns + ("alpha", "beta")
    data_ucb = {col: [] for col in columns_ucb}
    data_thompson = {col: [] for col in columns_thompson}

    for fname in files:
        with open(os.path.join(DATA_DIR, fname), "r") as fin:
            dct = json.load(fin)

        if dct["bandit_name"] == "ucb":
            for col in columns_ucb:
                data_ucb[col].append(dct[col])
        else:
            assert dct["bandit_name"] == "thompson"
            for col in columns_thompson:
                data_thompson[col].append(dct[col])
  
    # create dataframes
    df_ucb = pd.DataFrame(data=data_ucb, columns=columns_ucb)
    df_thompson = pd.DataFrame(data=data_thompson, columns=columns_thompson)

    print(df_ucb)
    print(df_thompson)


def get_index(fname: str) -> int:
    name = fname.split(".")[0]
    num = name.split("_")[1]
    return int(num)


if __name__ == "__main__":
    main()
