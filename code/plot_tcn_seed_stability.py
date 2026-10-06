"""Plot development-scale RouteCast versus TCN seed stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "qwen3": "Qwen3",
    "deepseek_r1": "DeepSeek-R1",
    "llama4_maverick": "Llama4 Maverick",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data["runs"]
    models = list(LABELS)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)
    colors = {"RouteCast": "#D95F4A", "TCN-only": "#4C78A8"}
    markers = {"RouteCast": "o", "TCN-only": "s"}
    x = np.arange(len(models), dtype=float)
    offsets = {"RouteCast": -0.09, "TCN-only": 0.09}
    for method, key in (("RouteCast", "routecast_recall"), ("TCN-only", "tcn_recall")):
        for i, model in enumerate(models):
            values = [r[key] for r in rows if r["model"] == model]
            jitter = np.linspace(-0.025, 0.025, len(values))
            ax.scatter(
                x[i] + offsets[method] + jitter,
                values,
                s=16,
                marker=markers[method],
                color=colors[method],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            ax.errorbar(
                x[i] + offsets[method], mean, yerr=sd,
                fmt=markers[method], color=colors[method],
                markersize=4.2, capsize=2.3, linewidth=1.1,
                label=method if i == 0 else None, zorder=4,
            )

    ax.set_xticks(x, [LABELS[m] for m in models])
    ax.set_ylabel("Native-budget recall")
    ax.set_xlabel("Development-scale screening (three seeds)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"figure_saved={args.output}")


if __name__ == "__main__":
    main()
