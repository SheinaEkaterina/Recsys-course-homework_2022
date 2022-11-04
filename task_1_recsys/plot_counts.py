from pathlib import Path
import matplotlib.pyplot as plt
from load_data import load_raw, last_day_split


PLOT_DIR = Path("./plots")


# load train data
df, _ = last_day_split(load_raw("../data"))
columns = ("zone_id", "banner_id", "campaign_clicks", "os_id", "country_id")
assert all((col in df.columns for col in columns))
for col_name in columns:
    grouped = df[[col_name, "clicks"]].groupby(col_name)
    x = list(grouped.groups.keys())
    counts = grouped.size()
    means = grouped.mean()

    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    fig.set_size_inches(6, 6)
    ax1.semilogy(counts.index, counts.values, color="b")
    ax1.set_ylabel("Counts")
    ax2.semilogy(means.index, means.values, color="r")
    ax2.set_ylabel("Avg. clicks")
    ax2.set_xlabel(col_name)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / (col_name + ".png"), dpi=150)
