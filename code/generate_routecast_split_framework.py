"""Generate three manuscript figures that explain RouteCast at increasing depth."""
from __future__ import annotations

from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
PKG = PROJECT_ROOT / "python_packages" / "figure"
sys.path.insert(0, str(PKG))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = PROJECT_ROOT / "record" / "paper" / "figures" / "routecast_split"
FINAL = PROJECT_ROOT / "record" / "paper" / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

INK = "#26323D"
MUTED = "#66727D"
LINE = "#52606B"
GRID = "#D8DEE4"
BLUE = "#4D82B8"
BLUE_SOFT = "#E5F0FA"
TEAL = "#4E9A93"
TEAL_SOFT = "#E3F2EF"
PURPLE = "#8067A8"
PURPLE_SOFT = "#EEE9F5"
GOLD = "#D59A3A"
GOLD_SOFT = "#F8EDD9"
RED = "#C75B5B"
RED_SOFT = "#F8E4E4"
GREEN = "#5C9A68"
GREEN_SOFT = "#E6F2E8"
GREY_SOFT = "#F3F5F7"


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rounded(ax, x, y, w, h, fc="white", ec=LINE, lw=0.9, radius=0.025, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.0, z=5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z,
    ))


def label(ax, x, y, letter, color):
    ax.add_patch(Circle((x, y), 0.017, fc=color, ec="none", zorder=9))
    ax.text(x, y, letter, ha="center", va="center", color="white",
            fontsize=6.5, fontweight="bold", zorder=10)


def bars(ax, x, y, w, h, color, values=(.35, .72, .48, .90, .58)):
    bw = w / (len(values) * 1.55)
    for i, v in enumerate(values):
        ax.add_patch(Rectangle((x + i * bw * 1.55, y), bw, h * v,
                               fc=color, ec="none", zorder=5))


def matrix(ax, x, y, w, h, color):
    values = np.array([[.15, .75, .25, .45], [.55, .20, .80, .30],
                       [.30, .50, .15, .70], [.78, .25, .48, .20]])
    for r in range(4):
        for c in range(4):
            ax.add_patch(Rectangle(
                (x + c*w/4, y + (3-r)*h/4), w/4, h/4,
                fc=mpl.colors.to_rgba(color, .18 + .72*values[r, c]),
                ec="white", lw=.35, zorder=5,
            ))


def network(ax, x, y, w, h, color, layers=(4, 3, 2)):
    xs = np.linspace(x, x+w, len(layers))
    pts = []
    for xx, n in zip(xs, layers):
        ys = np.linspace(y+.12*h, y+.88*h, n)
        pts.append([(xx, yy) for yy in ys])
    for left, right in zip(pts[:-1], pts[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                        alpha=.25, lw=.45, zorder=4)
    for layer in pts:
        for xx, yy in layer:
            ax.add_patch(Circle((xx, yy), .005, fc="white", ec=color,
                                lw=.8, zorder=6))


def save(fig, stem: str, final_name: str):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FINAL / final_name, bbox_inches="tight")
    plt.close(fig)


def overview():
    """Figure-level claim: RouteCast separates forecasting from selective action."""
    fig, ax = canvas(7.15, 2.35)
    stages = [
        ("A", "Route-only observation", "past expert IDs\n+ structural metadata", BLUE, BLUE_SOFT),
        ("B", "Multi-source forecast", "six evidence branches\n+ conditional fusion", TEAL, TEAL_SOFT),
        ("C", "Calibrated selection", "marginal calibration\n+ score-mass width", GOLD, GOLD_SOFT),
        ("D", "Selective admission", "weight + resident filters\n+ cache-value test", RED, RED_SOFT),
        ("E", "Native execution", "router reveals demand\n+ hit or demand load", GREEN, GREEN_SOFT),
    ]
    xs = np.linspace(.035, .805, len(stages))
    w, y, h = .155, .36, .45
    for i, ((letter, title, body, color, soft), x) in enumerate(zip(stages, xs)):
        rounded(ax, x, y, w, h, fc=soft, ec=color, radius=.035)
        label(ax, x+.022, y+h-.045, letter, color)
        ax.text(x+.048, y+h-.045, title, ha="left", va="center",
                fontsize=7.1, fontweight="bold", color=INK)
        ax.text(x+w/2, y+.15, body, ha="center", va="center",
                fontsize=6.3, color=MUTED, linespacing=1.35)
        if i == 0:
            matrix(ax, x+.045, y+.235, .066, .105, color)
        elif i == 1:
            bars(ax, x+.042, y+.235, .075, .10, color)
            network(ax, x+.105, y+.235, .035, .10, color, (3, 2, 1))
        elif i == 2:
            bars(ax, x+.042, y+.235, .085, .10, color)
        elif i == 3:
            for j, fc in enumerate((GREEN_SOFT, GREEN_SOFT, "white")):
                ax.add_patch(Rectangle((x+.043+j*.031, y+.25), .024, .07,
                                       fc=fc, ec=GREEN if j < 2 else GRID, lw=.7))
        else:
            network(ax, x+.045, y+.235, .065, .10, color, (3, 2, 1))
        if i < len(stages)-1:
            arrow(ax, x+w+.007, y+h/2, xs[i+1]-.009, y+h/2)

    ax.text(.50, .91, "RouteCast proposes transfers without replacing the native MoE router",
            ha="center", va="center", fontsize=8.2, color=INK, fontweight="bold")
    arrow(ax, .10, .22, .29, .22, color=LINE, lw=.9)
    arrow(ax, .29, .22, .48, .22, color=LINE, lw=.9)
    arrow(ax, .48, .22, .67, .22, color=LINE, lw=.9)
    arrow(ax, .67, .22, .86, .22, color=LINE, lw=.9)
    for x, text in zip((.10, .29, .48, .67, .86),
                       ("observe through t", "forecast t+1", "rank candidates",
                        "prefetch window", "native demand")):
        ax.add_patch(Circle((x, .22), .009, fc=LINE, ec="white", lw=.5, zorder=7))
        ax.text(x, .14, text, ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.text(.50, .055, "Causal inference order; cache replay assumes admitted transfers complete within the prefetch window.",
            ha="center", va="center", fontsize=5.9, color=MUTED)
    fig.subplots_adjust(left=.012, right=.992, bottom=.035, top=.98)
    save(fig, "Fig1_routecast_overview", "Fig1_RouteCastOverview.pdf")


def forecaster():
    """Figure-level claim: conditional fusion combines complementary route evidence."""
    fig, ax = canvas(7.15, 3.45)
    ax.text(.50, .965, "Context-gated multi-source route forecaster",
            ha="center", va="center", fontsize=8.3, color=INK, fontweight="bold")
    cards = [
        ("Layer popularity", "training prior", TEAL, "bar"),
        ("One-hop transition", "current token", BLUE, "mat"),
        ("Two-hop transition", "token t-1", PURPLE, "mat"),
        ("Request-prefill", "request-local prior", GOLD, "bar"),
        ("GRU history", "H = 16 route sequence", BLUE, "net"),
        ("Multiscale evidence", "frequency + recency", RED, "bar"),
    ]
    positions = [(.035,.64),(.225,.64),(.035,.36),(.225,.36),(.035,.08),(.225,.08)]
    for (title, sub, color, kind), (x, y) in zip(cards, positions):
        rounded(ax, x, y, .165, .22, fc="white", ec=color, radius=.022)
        ax.add_patch(Rectangle((x, y+.17), .165, .05,
                               fc=mpl.colors.to_rgba(color,.16), ec="none"))
        ax.text(x+.012, y+.195, title, ha="left", va="center",
                fontsize=6.4, color=INK, fontweight="bold")
        ax.text(x+.012, y+.025, sub, ha="left", va="bottom",
                fontsize=5.6, color=MUTED)
        if kind == "mat": matrix(ax, x+.088, y+.065, .058, .078, color)
        elif kind == "net": network(ax, x+.085, y+.065, .062, .078, color, (4,3,1))
        else: bars(ax, x+.088, y+.065, .060, .078, color)
        ax.plot([x+.165, .415], [y+.11, y+.11], color=color, lw=.75, zorder=3)

    ax.plot([.415, .415], [.19, .75], color=LINE, lw=.7, zorder=3)
    arrow(ax, .415, .70, .44, .70, color=PURPLE, lw=.85)
    ax.text(.407, .49, "normalize", ha="right", va="center",
            fontsize=5.4, color=MUTED, rotation=90, rotation_mode="anchor")

    rounded(ax, .44, .55, .20, .30, fc=PURPLE_SOFT, ec=PURPLE, radius=.026)
    label(ax, .465, .815, "A", PURPLE)
    ax.text(.49, .815, "Branch descriptors", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    ax.text(.54, .74, "[maximum, margin, entropy] × 6",
            ha="center", va="center", fontsize=6.2, color=MUTED)
    network(ax, .485, .625, .105, .08, PURPLE, (4,3,2))
    ax.text(.54, .59, "Gate MLP: 23 → 64 → 6",
            ha="center", va="center", fontsize=6.0, color=PURPLE)

    rounded(ax, .44, .15, .20, .29, fc=GREY_SOFT, ec=LINE, radius=.026)
    label(ax, .465, .405, "B", LINE)
    ax.text(.49, .405, "Structural context", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    context = ["layer position", "expert-pool size", "native sparsity",
               "prefill entropy", "token position"]
    for i, text in enumerate(context):
        ax.add_patch(Circle((.47, .35-i*.045), .004, fc=BLUE, ec="none"))
        ax.text(.482, .35-i*.045, text, ha="left", va="center",
                fontsize=5.8, color=MUTED)
    arrow(ax, .54, .47, .54, .55, color=LINE)

    rounded(ax, .69, .55, .27, .30, fc=TEAL_SOFT, ec=TEAL, radius=.026)
    label(ax, .715, .815, "C", TEAL)
    ax.text(.74, .815, "Probability-mixture fusion", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    colors = (TEAL, BLUE, PURPLE, GOLD, BLUE, RED)
    weights = (.12,.18,.14,.15,.25,.16)
    x0 = .72
    for c, wt in zip(colors, weights):
        ax.add_patch(Rectangle((x0, .685), .026, .10*wt/.25, fc=c, ec="none"))
        x0 += .034
    ax.text(.825, .625, "p_mix = sum_b omega_b p_b",
            ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(.825, .585, "dynamic weights for each token-layer",
            ha="center", va="center", fontsize=5.7, color=MUTED)
    arrow(ax, .64, .70, .69, .70, color=PURPLE)

    rounded(ax, .69, .15, .27, .29, fc=RED_SOFT, ec=RED, radius=.026)
    label(ax, .715, .405, "D", RED)
    ax.text(.74, .405, "Residual refinement", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    network(ax, .735, .265, .10, .08, RED, (4,3,1))
    ax.text(.865, .305, "10 → 32 → 1", ha="center", va="center",
            fontsize=6.2, color=INK)
    ax.text(.825, .205, "sₑ = log p_mix,e + 0.1 Δₑ",
            ha="center", va="center", fontsize=6.5, color=INK)
    arrow(ax, .825, .55, .825, .45, color=TEAL)
    ax.text(.50, .035,
            "All branch transforms share parameters across experts; the fixed descriptor dimension supports heterogeneous expert pools.",
            ha="center", va="center", fontsize=5.9, color=MUTED)
    fig.subplots_adjust(left=.012, right=.992, bottom=.035, top=.98)
    save(fig, "Fig2_multisource_forecaster", "Fig2_MultiSourceForecaster.pdf")


def selection_admission():
    """Figure-level claim: calibrated scores become selective cache actions."""
    fig, ax = canvas(7.15, 3.15)
    ax.text(.50, .965, "From calibrated expert scores to selective cache action",
            ha="center", va="center", fontsize=8.3, color=INK, fontweight="bold")

    rounded(ax, .035, .55, .14, .27, fc=GREY_SOFT, ec=LINE, radius=.025)
    label(ax, .06, .785, "A", LINE)
    ax.text(.085, .785, "Fused scores", ha="left", va="center",
            fontsize=7.0, color=INK, fontweight="bold")
    bars(ax, .065, .625, .085, .10, LINE)
    ax.text(.105, .585, "sₑ / T*", ha="center", va="center", fontsize=6.5, color=INK)

    rounded(ax, .225, .60, .18, .22, fc=BLUE_SOFT, ec=BLUE, radius=.025)
    label(ax, .25, .785, "B", BLUE)
    ax.text(.275, .785, "Marginal calibration", ha="left", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    ax.text(.315, .705, "qₑ = sigmoid(sₑ / T*)", ha="center", va="center",
            fontsize=6.4, color=INK)
    ax.text(.315, .645, "BCE · Brier · ECE", ha="center", va="center",
            fontsize=5.8, color=MUTED)

    rounded(ax, .225, .30, .18, .22, fc=GOLD_SOFT, ec=GOLD, radius=.025)
    label(ax, .25, .485, "C", GOLD)
    ax.text(.275, .485, "Ranking weights", ha="left", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    ax.text(.315, .405, "wₑ = softmax(sₑ / T*)", ha="center", va="center",
            fontsize=6.4, color=INK)
    ax.text(.315, .345, "Σₑ wₑ = 1; ranking only", ha="center", va="center",
            fontsize=5.8, color=MUTED)
    arrow(ax, .175, .685, .225, .71, color=BLUE)
    arrow(ax, .175, .655, .225, .41, color=GOLD)

    rounded(ax, .455, .30, .17, .52, fc=GOLD_SOFT, ec=GOLD, radius=.025)
    label(ax, .48, .785, "D", GOLD)
    ax.text(.505, .785, "Score-mass width", ha="left", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    bars(ax, .50, .625, .085, .11, GOLD, (.95,.75,.58,.42,.30))
    ax.plot([.49,.59], [.60,.60], color=RED, lw=.9, ls=(0,(3,2)))
    ax.text(.60, .60, "m*", ha="left", va="center", fontsize=6.0, color=RED)
    ax.text(.54, .525, "smallest prefix K(m*)", ha="center", va="center",
            fontsize=6.1, color=INK)
    ax.text(.54, .43, "validation constraint:\nretain 98% native Recall",
            ha="center", va="center", fontsize=5.8, color=MUTED, linespacing=1.35)
    arrow(ax, .405, .41, .455, .50, color=GOLD)

    rounded(ax, .675, .30, .29, .52, fc=RED_SOFT, ec=RED, radius=.025)
    label(ax, .70, .785, "E", RED)
    ax.text(.725, .785, "Selective cache admission", ha="left", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    steps = [
        ("Weight filter", "w(a) ≥ τ*"),
        ("Resident filter", "remove cached experts"),
        ("Value test", "w(a) > min v(c) + δ"),
    ]
    for i, (title, sub) in enumerate(steps):
        yy = .655 - i*.12
        rounded(ax, .71, yy, .20, .075, fc="white", ec=RED, lw=.75, radius=.014)
        ax.text(.735, yy+.048, title, ha="left", va="center",
                fontsize=6.1, fontweight="bold", color=INK)
        ax.text(.895, yy+.025, sub, ha="right", va="center",
                fontsize=5.5, color=MUTED)
        if i < 2: arrow(ax, .81, yy-.002, .81, yy-.04, color=RED, lw=.8)
    rounded(ax, .755, .335, .11, .065, fc=GREEN_SOFT, ec=GREEN, radius=.017)
    ax.text(.81, .367, "Prefetch | Abstain", ha="center", va="center",
            fontsize=6.0, color=INK, fontweight="bold")
    arrow(ax, .625, .56, .675, .56, color=RED)

    rounded(ax, .055, .085, .89, .115, fc="#FAFBFC", ec=GRID, radius=.018)
    timeline = [
        ("Observe", "through t", BLUE), ("Forecast", "t+1", PURPLE),
        ("Select", "rank + cache", GOLD), ("Transfer", "idealized window", RED),
        ("Demand", "native route", LINE), ("Execute", "hit / load", GREEN),
    ]
    xs = np.linspace(.10, .90, len(timeline))
    for i, (title, sub, color) in enumerate(timeline):
        ax.add_patch(Circle((xs[i], .14), .010, fc=color, ec="white", lw=.5, zorder=7))
        ax.text(xs[i], .175, title, ha="center", va="center",
                fontsize=5.9, color=INK, fontweight="bold")
        ax.text(xs[i], .105, sub, ha="center", va="center", fontsize=5.3, color=MUTED)
        if i < len(timeline)-1:
            arrow(ax, xs[i]+.014, .14, xs[i+1]-.014, .14, color=LINE, lw=.75)
    ax.text(.50, .035,
            "RouteCast affects speculative movement only; the frozen native router remains authoritative over execution.",
            ha="center", va="center", fontsize=5.9, color=MUTED)
    fig.subplots_adjust(left=.012, right=.992, bottom=.035, top=.98)
    save(fig, "Fig3_selection_admission", "Fig3_SelectionAdmission.pdf")


def main():
    overview()
    forecaster()
    selection_admission()
    print(f"source_exports={OUT}", flush=True)
    print(f"submission_exports={FINAL}", flush=True)


if __name__ == "__main__":
    main()
