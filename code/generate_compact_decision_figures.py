"""Generate compact main-text decision and cache figures from recorded outputs.

The full per-model reliability and mass curves remain in the supplementary
figure bundle. These main-text composites retain only evidence needed for the
two Results claims: calibration enables model-specific operating points, and
cache-aware scheduling has model- and capacity-dependent Pareto behaviour.
"""
from pathlib import Path
import csv
import json
import sys

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "python_packages" / "figure"))
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = PROJECT_ROOT / "record"
OUT = ROOT / "paper" / "figures" / "compact"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen3", "deepseek_r1", "llama4_maverick"]
DISPLAY = {
    "qwen3": "Qwen3",
    "deepseek_r1": "DeepSeek-R1",
    "llama4_maverick": "Llama4 Maverick",
}
MODEL_COLORS = {
    "qwen3": "#3B78B5",
    "deepseek_r1": "#D98242",
    "llama4_maverick": "#4F9D69",
}
POLICY_COLORS = {
    "LRU": "#9AA1A7",
    "fixed_native": "#6C8EBF",
    "adaptive_mass": "#D6A34A",
    "adaptive_cache": "#C94C4C",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#E7E9EB", linewidth=0.6)
    ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(-0.13, 1.06, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")


def save_bundle(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(stem, flush=True)


def calibration_and_budget():
    calibration = read_json(
        ROOT / "calibration" / "routecast_v3_limit100_calibration.json"
    )["models"]
    cache = read_json(
        ROOT / "adaptive_cache" / "routecast_v3_limit100_adaptive_cache.json"
    )["models"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.15, 2.45))
    x = np.arange(len(MODELS))
    before = [calibration[m]["test_before"]["ece"] for m in MODELS]
    after = [calibration[m]["test_after"]["ece"] for m in MODELS]
    width = 0.34
    ax0.bar(x - width / 2, before, width, color="#A7B0B8", label="Before")
    ax0.bar(x + width / 2, after, width, color="#D55E5E", label="After")
    ax0.set_xticks(x, ["Qwen3", "DeepSeek-R1", "Llama4\nMaverick"])
    ax0.set_ylabel("Expected calibration error")
    ax0.set_title("Calibration effect is model dependent")
    ax0.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax0)
    panel_label(ax0, "a")

    label_offsets = {
        "qwen3": (7, 0, "left"),
        "deepseek_r1": (-7, 7, "right"),
        "llama4_maverick": (7, 7, "left"),
    }
    for model in MODELS:
        row = cache[model]
        mass = row["selected_mass"]
        curve = row["val_curve"]
        point = curve[str(mass)]
        ax1.scatter(point["avg_k"], point["recall"], s=48,
                    color=MODEL_COLORS[model], zorder=3)
        dx, dy, align = label_offsets[model]
        ax1.annotate(
            f"{DISPLAY[model]}, m={mass:g}",
            (point["avg_k"], point["recall"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=6,
            ha=align,
        )
    ax1.set_xlabel("Average candidate width")
    ax1.set_ylabel("Validation Recall")
    ax1.set_title("Selected operating points are model specific")
    style_axis(ax1)
    panel_label(ax1, "b")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.82, wspace=0.34)
    save_bundle(fig, "Fig5_compact_calibration_budget")


def calibration_and_budget_column():
    """Single-column vertical layout that can sit beside Results prose."""
    calibration = read_json(
        ROOT / "calibration" / "routecast_v3_limit100_calibration.json"
    )["models"]
    cache = read_json(
        ROOT / "adaptive_cache" / "routecast_v3_limit100_adaptive_cache.json"
    )["models"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.45, 4.35))
    x = np.arange(len(MODELS))
    before = [calibration[m]["test_before"]["ece"] for m in MODELS]
    after = [calibration[m]["test_after"]["ece"] for m in MODELS]
    width = 0.34
    ax0.bar(x - width / 2, before, width, color="#A7B0B8", label="Before")
    ax0.bar(x + width / 2, after, width, color="#D55E5E", label="After")
    ax0.set_xticks(x, ["Qwen3", "DeepSeek-R1", "Llama4\nMaverick"])
    ax0.set_ylabel("Expected calibration error")
    ax0.set_title("Calibration effect is model dependent")
    ax0.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax0)
    panel_label(ax0, "a")

    label_offsets = {
        "qwen3": (6, -2, "left"),
        "deepseek_r1": (-6, 8, "right"),
        "llama4_maverick": (6, -13, "left"),
    }
    for model in MODELS:
        row = cache[model]
        mass = row["selected_mass"]
        point = row["val_curve"][str(mass)]
        ax1.scatter(point["avg_k"], point["recall"], s=42,
                    color=MODEL_COLORS[model], zorder=3)
        dx, dy, align = label_offsets[model]
        ax1.annotate(
            f"{DISPLAY[model]}, m={mass:g}",
            (point["avg_k"], point["recall"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=6,
            ha=align,
        )
    ax1.set_xlabel("Average candidate width")
    ax1.set_ylabel("Validation Recall")
    ax1.set_title("Selected operating points are model specific", pad=9)
    style_axis(ax1)
    panel_label(ax1, "b")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.12, top=0.95, hspace=0.55)
    save_bundle(fig, "Fig5_column_calibration_budget")


def cache_pareto():
    rows = read_csv(
        ROOT / "adaptive_cache" / "routecast_v3_limit100_adaptive_cache.csv"
    )
    policies = ["LRU", "fixed_native", "adaptive_mass", "adaptive_cache"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    for index, (ax, model) in enumerate(zip(axes, MODELS)):
        subset = [row for row in rows if row["model"] == model]
        for policy in policies:
            values = sorted(
                [row for row in subset if row["policy"] == policy],
                key=lambda row: int(row["capacity"]),
            )
            transfers = [float(row["total_transfers"]) / 1e6 for row in values]
            hits = [float(row["hit_rate"]) for row in values]
            ax.plot(transfers, hits, "o-", markersize=3.0, linewidth=1.1,
                    color=POLICY_COLORS[policy],
                    label={
                        "LRU": "LRU",
                        "fixed_native": "Fixed Native",
                        "adaptive_mass": "Adaptive Mass",
                        "adaptive_cache": "Adaptive Cache",
                    }[policy])
            if policy == "adaptive_cache":
                for tx, hit, row in zip(transfers, hits, values):
                    ax.annotate(f"{row['capacity']}x", (tx, hit), xytext=(3, 2),
                                textcoords="offset points", fontsize=5.5)
        ax.set_title(DISPLAY[model])
        ax.set_xlabel("Transfers (million)")
        if index == 0:
            ax.set_ylabel("Cache hit rate")
        style_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.52, 0.995))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.78, wspace=0.30)
    save_bundle(fig, "Fig6_compact_cache_pareto")


def cache_pareto_column():
    """Single-column vertical cache layout to avoid a full-width figure page."""
    rows = read_csv(
        ROOT / "adaptive_cache" / "routecast_v3_limit100_adaptive_cache.csv"
    )
    policies = ["LRU", "fixed_native", "adaptive_mass", "adaptive_cache"]
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.45))
    for index, (ax, model) in enumerate(zip(axes, MODELS)):
        subset = [row for row in rows if row["model"] == model]
        for policy in policies:
            values = sorted(
                [row for row in subset if row["policy"] == policy],
                key=lambda row: int(row["capacity"]),
            )
            transfers = [float(row["total_transfers"]) / 1e6 for row in values]
            hits = [float(row["hit_rate"]) for row in values]
            ax.plot(transfers, hits, "o-", markersize=2.8, linewidth=1.0,
                    color=POLICY_COLORS[policy],
                    label={
                        "LRU": "LRU",
                        "fixed_native": "Fixed Native",
                        "adaptive_mass": "Adaptive Mass",
                        "adaptive_cache": "Adaptive Cache",
                    }[policy])
            if policy == "adaptive_cache":
                for tx, hit, row in zip(transfers, hits, values):
                    ax.annotate(f"{row['capacity']}x", (tx, hit), xytext=(3, 2),
                                textcoords="offset points", fontsize=5.5)
        ax.set_title(DISPLAY[model])
        ax.set_xlabel("Transfers (million)")
        ax.set_ylabel("Cache hit rate")
        style_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.53, 0.995))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.08, top=0.91, hspace=0.63)
    save_bundle(fig, "Fig6_column_cache_pareto")


if __name__ == "__main__":
    calibration_and_budget()
    cache_pareto()
    calibration_and_budget_column()
    cache_pareto_column()
