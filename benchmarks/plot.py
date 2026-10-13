"""Plot sweep results (throughput / TTFT / TPOT vs concurrency) and print a markdown table.

Usage (from the repo root):
    python -m benchmarks.plot results/*.json --out assets/benchmark.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated default palette), light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

PANELS = [
    ("output_throughput", "Output throughput (tok/s)", "higher is better"),
    ("ttft_ms_mean", "Mean TTFT (ms)", "lower is better"),
    ("tpot_ms_mean", "Mean TPOT (ms/token)", "lower is better"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", default="assets/benchmark.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    runs = []
    for path in args.files:
        with open(path) as f:
            runs.append(json.load(f))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, (key, label, hint) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for color, run in zip(SERIES, runs):
            xs = [r["concurrency"] for r in run["results"]]
            ys = [r[key] for r in run["results"]]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=run["meta"]["label"])
        ax.set_xscale("log", base=2)
        xs = sorted({r["concurrency"] for run in runs for r in run["results"]})
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Concurrent requests")
        ax.set_title(f"{label}\n", loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    meta = runs[0]["meta"]
    caption = args.title or (f"RTX 4070 12 GB · closed loop · input {meta['input_len']} / output "
                             f"{meta['output_len']} tokens · random prompts, ignore_eos")
    fig.text(0.5, 0.015, caption, ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}\n")

    print("| Run | Concurrency | Output tok/s | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    for run in runs:
        for r in run["results"]:
            print(f"| {run['meta']['label']} | {r['concurrency']} | {r['output_throughput']:.0f} | "
                  f"{r['ttft_ms_mean']:.0f} | {r['ttft_ms_p99']:.0f} | {r['tpot_ms_mean']:.1f} |")


if __name__ == "__main__":
    main()
