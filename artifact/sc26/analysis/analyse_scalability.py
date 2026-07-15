from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str) 
parser.add_argument('--output', type=str) 
args = parser.parse_args()

def merge_duplicates(df):
    time_cols = [c for c in df.columns if "time" in c]
    static_cols = [c for c in df.columns if "time" not in c and c not in ("graph", "gpus")]
    df_clean = (
        df
        .groupby(["graph", "gpus"], as_index=False)
        .agg(
            {**{c: "mean" for c in time_cols},
            **{c: "first" for c in static_cols}}
        )
    )
    return df_clean

# Ideal
fontsize = 15
x = [1,2,3,4,8,12,16,20,24,28,32]
plt.figure(figsize=(15,7))
plt.plot(x, x, marker="v", color="black", label="Linear", linestyle="-.")

# TriBIT (larger graphs)
df = pd.read_csv(args.input)
df = merge_duplicates(df)

speedups = {}
for graph in np.unique(df["graph"]):
    df_filtered = df[df["graph"] == graph]
    if 1 not in df_filtered["gpus"].values:  # Cannot normalize
        continue
    base = df_filtered.loc[df_filtered["gpus"] == 1, "counting_time"].iloc[0]
    speedup = base / df_filtered.set_index("gpus")["counting_time"]
    speedups[graph] = speedup.reindex(x)  # align to the fixed x-axis, NaN where missing

speedup_df = pd.DataFrame(speedups)  # rows = gpus (from x), columns = graphs

y = np.exp(np.nanmean(np.log(speedup_df.values), axis=1))
plt.plot(x, y, marker="o", label="TriBIT (larger graphs)", color = "green")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Num. GPUs", fontsize=fontsize)
plt.ylabel("Geom. Mean Speedup", fontsize=fontsize)
plt.legend(fontsize=fontsize)

# Grid lines through every plotted point (x = gpu counts, y = linear-reference values)
plt.xticks(x, labels=x, fontsize=12)
plt.minorticks_off()
plt.grid(True, which="major", linestyle=":")

plt.savefig(args.output)
