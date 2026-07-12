import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str) 
parser.add_argument('--output', type=str) 
args = parser.parse_args()

def merge_duplicates(df):
    time_cols = [c for c in df.columns if "time" in c]
    static_cols = [c for c in df.columns if "time" not in c]
    df_clean = (
        df
        .groupby("graph", as_index=False)
        .agg(
            {**{c: "mean" for c in time_cols},
            **{c: "first" for c in static_cols}}
        )
    )
    return df_clean

def print_number_of_matrices(ax, num_matrices):
    ax.text(
        0.98, 0.02,
        f"# matrices = {num_matrices}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
    )

title_size = 16
label_size = 14
fontsize = 15
marker_size = 40
legend_fontsize = 14
ticks_fontsize = 15
markers = [".","x"]
sorting_magnitude = "nnz"

fig, axs = plt.subplots(2, 2, figsize=(17, 17), sharex=True)
axs[0, 1].sharey(axs[0, 0])

# General properties
for ax in axs.flat:
    ax.grid(linestyle = "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_fontsize)

rescale_factor = 1000
df = pd.read_csv(args.file)
metrics = [metric for metric in df.columns if "time" in metric and "gpu_total_s_time" not in metric]
df = merge_duplicates(df)

################# TIME METRICS #################
ax = axs[0, 0]
ax.set_title("Time (s)", fontsize = title_size)

df["auxiliary_time"] = df[[m for m in metrics if m != "kernel_time"]].sum(axis=1)
df["total_time"] = df[metrics].sum(axis=1)

for j, metric in enumerate(["auxiliary_time", "kernel_time"]):
    label_display = "Pre/postprocessing time" if metric == "auxiliary_time" else "Kernel time"
    scatter = ax.scatter(
        df[sorting_magnitude],
        df[metric] / rescale_factor,
        label=label_display,
        marker=markers[j % len(markers)],
        s = marker_size
    )
    scatter.set_zorder(100 if "kernel" in metric else 101)

print_number_of_matrices(ax, len(df))
legend = ax.legend(fontsize = legend_fontsize, loc='upper left')

################# TOTAL TIME #################
ax = axs[0, 1]
ax.set_title("Total time (s)", fontsize = title_size)

ax.scatter(df[sorting_magnitude], df["total_time"] / rescale_factor, marker=markers[0], color = "red", s = marker_size)
print_number_of_matrices(ax, len(df))

################# NUMBER OF TRIANGLES #################
ax = axs[1, 0]
ax.set_title("# Triangles", fontsize = title_size)
ax.set_xlabel("Number NNZs", fontsize = label_size)
ax.set_yscale("symlog", linthresh=1)

ax.scatter(df[sorting_magnitude], df["triangles"], s = marker_size, marker=".", color = "green")
ticks = [0] + [10*(1000**i) for i in range(7)]
ax.set_yticks(ticks)
ax.minorticks_off()
ax.set_ylim(-1, 10**12)
print_number_of_matrices(ax, len(df))

################# MEMORY CONSUMPTION #################
ax = axs[1,1]
ax.set_title("Memory usage (GiB)", fontsize = title_size)
ax.set_xlabel("Number NNZs", fontsize = label_size)

xmin = round(df[sorting_magnitude].min())
xmax = round(df[sorting_magnitude].max())
ax.hlines(64, xmin, xmax, color = "red", linestyle="-.")
ax.text(
    xmin,
    40,
    "64 GiB limit",
    fontsize = 15,
    color="red",
    verticalalignment="top"
)
ax.scatter(df[sorting_magnitude], df["max_memory_consumption"]/ 1024**3, s = marker_size, marker=markers[0], color = "green")
print_number_of_matrices(ax, len(df))

plt.tight_layout()
plt.savefig(args.output)
