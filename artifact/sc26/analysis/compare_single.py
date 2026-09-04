import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os


def merge_duplicates(df):
    time_cols = [c for c in df.columns if "time" in c]
    static_cols = [c for c in df.columns if "time" not in c]
    df_clean = df.groupby("graph", as_index=False).agg(
        {**{c: "mean" for c in time_cols}, **{c: "first" for c in static_cols}}
    )
    return df_clean


def print_number_of_matrices(ax, num_matrices):
    ax.text(
        0.98,
        0.02,
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

rescale_factor = 1
markers = [".", "x", "^", "v"]
sorting_magnitude = "nnz"
pathbase = "results/raw/"
listdir = [
    "results_single_tot.csv",
    "results_single_bbtc.csv",
    "results_single_tc.csv",
    "results_single.csv",
]
baseline = listdir[-1]

subplot_dict = {
    "time": plt.subplots(
        3,
        len(listdir),
        figsize=(19, 7),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [1.2, 1, 1]},
    ),
    "correctness": plt.subplots(
        1, len(listdir), figsize=(19, 3), sharex=True, sharey="row"
    ),
    "memory": plt.subplots(2, len(listdir), figsize=(19, 5), sharex=True, sharey="row"),
}

df_baseline = pd.read_csv(pathbase + baseline)
df_baseline[[metric for metric in df_baseline.columns if "time" in metric]] /= 1000
df_baseline["total_time"] = (
    df_baseline["preprocessing_time"] + df_baseline["kernel_time"]
)
df_baseline = merge_duplicates(df_baseline)

# General properties
for stage in ["time", "correctness", "memory"]:
    fig, axs = subplot_dict[stage]
    for ax in axs.flat:
        ax.grid(linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=ticks_fontsize)


filename2pretty = {
    "results_single.csv": "TriBIT (this work)",
    "results_single_tot.csv": "ToT",
    "results_single_bbtc.csv": "bbTC",
    "results_single_tc.csv": "WeTriC",
}

ylabels = ["Time (s)", "Total time (s)", "Total speedup (x)", "# Triangles", "Memory usage (GB)", "Memory ratio (x)"]

for i, file_name in enumerate(listdir):
    counter = 0
    title = filename2pretty[file_name]
    for stage in ["time", "correctness", "memory"]:
        fig, axs = subplot_dict[stage]
        n_rows = axs.shape[0] if axs.ndim == 2 else 1
        for row in range(n_rows):
            ax = axs[i] if axs.ndim == 1 else axs[row, i]
            if row == 0:
                ax.set_title(title, fontsize=title_size)
            if i == 0:
                ax.set_ylabel(ylabels[counter], fontsize=label_size)
            counter += 1

    if not os.path.exists(pathbase + file_name):
        continue

    title = filename2pretty[file_name]
    rescale_factor = (
        1000 if file_name in ["results_single.csv", "results_single_tot.csv"] else 1
    )
    df = pd.read_csv(pathbase + file_name)
    df = df.sort_values(by=sorting_magnitude, ascending=True, ignore_index=True)
    metrics = [
        metric
        for metric in df.columns
        if "time" in metric and "gpu_total_s_time" not in metric
    ]
    df = merge_duplicates(df)
    df["total_time"] = np.zeros(len(df))
    df["auxiliary_time"] = np.zeros(len(df))

    ax_counter = 0
    ################# TIME METRICS #################
    fig, axs = subplot_dict["time"]
    ax = axs[ax_counter, i]
    ax_counter += 1
    for j, metric in enumerate(metrics):
        df["total_time"] += df[metric]
        if metric != "kernel_time":
            df["auxiliary_time"] += df[metric]

    for j, metric in enumerate(["auxiliary_time", "kernel_time"]):
        label_display = (
            "Pre/postprocessing time" if metric == "auxiliary_time" else "Kernel time"
        )
        scatter = ax.scatter(
            df[sorting_magnitude],
            df[metric] / rescale_factor,
            label=label_display,
            marker=markers[j % len(markers)],
            s=marker_size,
        )
        scatter.set_zorder(100 if "kernel" in metric else 101)
        if j == 0:
            ax.text(
                0.98,
                0.02,
                f"# matrices = {len(df)}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=fontsize,
            )
    legend = ax.legend(fontsize=legend_fontsize, loc="upper left")

    ################# TOTAL TIME #################
    ax = axs[ax_counter, i]
    ax_counter += 1
    if i == 0:
        ax.set_ylabel("Total time (s)", fontsize=label_size)
    ax.scatter(
        df[sorting_magnitude],
        df["total_time"] / rescale_factor,
        marker=markers[0],
        color="red",
        s=marker_size,
    )
    print_number_of_matrices(ax, len(df))

    df_merged = df.merge(
        df_baseline[
            [
                "graph",
                "kernel_time",
                "total_time",
                "triangles",
                "max_memory_consumption",
            ]
        ],
        on="graph",
        how="inner",
        suffixes=("", "_bsc"),
    )

    ################# TOTAL SPEEDUP #################
    ax = axs[ax_counter, i]
    ax_counter += 1
    # ax.set_yscale("linear")
    df_merged["speedup"] = (df_merged["total_time"] / rescale_factor) / df_merged[
        "total_time_bsc"
    ]
    # print(df_merged[["graph", "speedup"]].iloc[df_merged["speedup"].argmin()])
    ax.scatter(
        df_merged[sorting_magnitude],
        df_merged["speedup"],
        marker=markers[0],
        s=marker_size,
    )
    ax.hlines(
        1,
        round(df_merged[sorting_magnitude].min()),
        round(df_merged[sorting_magnitude].max()),
        color="red",
        linestyle="-.",
    )
    print_number_of_matrices(ax, len(df_merged))
    ax.set_xlabel("Number NNZs", fontsize=label_size)

    ################# NUMBER OF TRIANGLES #################
    fig, axs = subplot_dict["correctness"]
    ax = axs[i]

    if "triangles" in df.columns:
        df_matches = df_merged[df_merged["triangles"] == df_merged["triangles_bsc"]]
        df_mismatches = df_merged[df_merged["triangles"] != df_merged["triangles_bsc"]]
        ax.scatter(
            df_matches[sorting_magnitude],
            df_matches["triangles"],
            s=marker_size,
            marker=".",
            label="Match",
            color="green",
        )
        ax.scatter(
            df_mismatches[sorting_magnitude],
            df_mismatches["triangles"],
            s=marker_size,
            marker="x",
            label="Mismatch",
            color="red",
        )

    ax.set_yscale("symlog", linthresh=1)
    ticks = [0] + [10 * (1000**i) for i in range(7)]
    ax.set_yticks(ticks)
    ax.minorticks_off()
    ax.set_ylim(-1, 10**12)
    ax.legend(loc="upper left", fontsize=legend_fontsize)
    ax.set_xlabel("Number NNZs", fontsize=label_size)

    ax_counter = 0
    ################# MEMORY CONSUMPTION #################
    fig, axs = subplot_dict["memory"]
    ax = axs[ax_counter, i]
    ax_counter += 1

    # ax.set_yscale("linear")
    xmin = round(df[sorting_magnitude].min())
    xmax = round(df[sorting_magnitude].max())
    ax.hlines(64, xmin, xmax, color="red", linestyle="-.")
    ax.text(xmin, 20, "64 GB limit", fontsize=15, color="red", verticalalignment="top")
    ax.scatter(
        df[sorting_magnitude],
        (df["max_memory_consumption"]) / 1024**3,
        s=marker_size,
        marker=markers[0],
        label="Match",
        color="green",
    )

    ################# MEMORY RATIO #################
    ax = axs[ax_counter, i]
    ax_counter += 1
    ax.set_yscale("linear")
    df_to_plot = (
        df_merged["max_memory_consumption_bsc"] / df_merged["max_memory_consumption"]
    )
    ax.scatter(
        df_merged[sorting_magnitude], df_to_plot, marker=markers[0], s=marker_size
    )
    ax.hlines(
        1,
        round(df_merged[sorting_magnitude].min()),
        round(df_merged[sorting_magnitude].max()),
        color="red",
        linestyle="-.",
    )
    ax.set_xlabel("Number NNZs", fontsize=label_size)


for stage in ["time", "correctness", "memory"]:
    fig, axs = subplot_dict[stage]
    fig.tight_layout()
    fig.savefig(f"results/plot_compare_{stage}.png")
