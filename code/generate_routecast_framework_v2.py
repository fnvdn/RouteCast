"""Generate the detailed RouteCast architecture and causal timeline.

The diagram mirrors the implemented computation graph in UniversalRouteCastV3
and the validated calibration / adaptive-cache evaluation pipeline.  It does
not imply that RouteCast replaces the native MoE router.
"""
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


OUT = PROJECT_ROOT / "record" / "paper" / "figures"
FINAL = PROJECT_ROOT / "record" / "paper" / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.2,
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


def rounded(ax, x, y, w, h, fc="white", ec=LINE, lw=0.8, radius=0.012, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=0.85, style="-|>", ls="-", z=5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7.5,
        linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0", zorder=z,
    ))


def stage_header(ax, x, w, number, title, color):
    ax.add_patch(Circle((x + 0.012, 0.942), 0.012, facecolor=color,
                        edgecolor="none", zorder=8))
    ax.text(x + 0.012, 0.942, str(number), ha="center", va="center",
            color="white", fontsize=6.0, fontweight="bold", zorder=9)
    ax.text(x + 0.030, 0.942, title, ha="left", va="center",
            fontsize=6.7, fontweight="bold", color=INK)
    ax.plot([x, x + w], [0.917, 0.917], color=color, lw=1.5,
            solid_capstyle="round")


def mini_hist(ax, x, y, w, h, color):
    vals = np.array([0.25, 0.62, 0.43, 0.88, 0.52])
    bw = w / 7
    for i, val in enumerate(vals):
        ax.add_patch(Rectangle((x + (i + 0.6) * bw, y), bw * 0.62, h * val,
                               facecolor=color, edgecolor="none", zorder=5))


def mini_matrix(ax, x, y, w, h, color, skip=False):
    mat = np.array([[.15, .75, .25, .45], [.55, .2, .8, .3],
                    [.3, .5, .15, .7], [.78, .25, .48, .2]])
    if skip:
        mat = np.flipud(mat)
    for r in range(4):
        for c in range(4):
            alpha = 0.18 + 0.72 * mat[r, c]
            ax.add_patch(Rectangle((x + c*w/4, y + (3-r)*h/4), w/4, h/4,
                                   facecolor=mpl.colors.to_rgba(color, alpha),
                                   edgecolor="white", linewidth=0.25, zorder=5))


def mini_network(ax, x, y, w, h, color, layers=(4, 3, 1)):
    xs = np.linspace(x, x + w, len(layers))
    coords = []
    for xx, n in zip(xs, layers):
        ys = np.linspace(y + 0.12*h, y + 0.88*h, n)
        coords.append([(xx, yy) for yy in ys])
    for left, right in zip(coords[:-1], coords[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                        alpha=0.28, lw=0.35, zorder=4)
    for layer in coords:
        for xx, yy in layer:
            ax.add_patch(Circle((xx, yy), 0.0042, facecolor="white",
                                edgecolor=color, linewidth=0.65, zorder=6))


def branch_card(ax, x, y, w, h, title, subtitle, color, kind):
    rounded(ax, x, y, w, h, fc="white", ec=color, lw=0.75, radius=0.008, z=3)
    ax.add_patch(Rectangle((x, y+h-0.026), w, 0.026, facecolor=color,
                           edgecolor="none", alpha=0.16, zorder=3.5))
    ax.text(x + 0.008, y+h-0.013, title, ha="left", va="center",
            fontsize=5.7, fontweight="bold", color=INK, zorder=7)
    ax.text(x + 0.008, y+0.013, subtitle, ha="left", va="bottom",
            fontsize=5.1, color=MUTED, zorder=7)
    vx, vy, vw, vh = x + w*0.52, y + h*0.27, w*0.42, h*0.40
    if kind == "hist":
        mini_hist(ax, vx, vy, vw, vh, color)
    elif kind == "matrix":
        mini_matrix(ax, vx, vy, vw, vh, color, False)
    elif kind == "matrix2":
        mini_matrix(ax, vx, vy, vw, vh, color, True)
    elif kind == "gru":
        for i in range(4):
            cx = vx + i * vw/4.4
            ax.add_patch(Circle((cx, vy+vh*.55), 0.006, fc=BLUE_SOFT,
                                ec=color, lw=.55, zorder=5))
            if i:
                arrow(ax, cx-vw/4.4+.007, vy+vh*.55, cx-.007, vy+vh*.55,
                      color=color, lw=.45)
        rounded(ax, vx+vw*.74, vy+vh*.25, vw*.24, vh*.60, fc=BLUE_SOFT,
                ec=color, lw=.55, radius=.004, z=5)
        ax.text(vx+vw*.86, vy+vh*.55, "GRU", ha="center", va="center",
                fontsize=5.1, color=INK, fontweight="bold", zorder=7)
    elif kind == "multi":
        widths = [1.0, .78, .58, .38]
        for i, frac in enumerate(widths):
            yy = vy + i*vh*.22
            ax.plot([vx, vx+vw*frac*.62], [yy, yy], color=color,
                    lw=1.0, solid_capstyle="round", zorder=5)
        mini_network(ax, vx+vw*.68, vy, vw*.28, vh, color, layers=(3,2,1))


def route_tensor(ax, x, y, w, h):
    pattern = np.array([
        [0,0,1,0,0,1,0,0], [0,1,0,0,1,0,0,0], [1,0,0,1,0,0,0,1],
        [0,0,1,0,1,0,0,0], [0,1,0,0,0,1,0,0], [1,0,0,1,0,0,1,0],
    ])
    rows, cols = pattern.shape
    for r in range(rows):
        for c in range(cols):
            fc = BLUE if pattern[r,c] else "#EDF1F4"
            ax.add_patch(Rectangle((x+c*w/cols, y+(rows-1-r)*h/rows),
                                   w/cols, h/rows, facecolor=fc,
                                   edgecolor="white", linewidth=.35, zorder=5))
    ax.text(x+w/2, y-0.012, "history tensor [H x E]", ha="center", va="top",
            fontsize=5.0, color=MUTED)


def probability_bars(ax, x, y, w, h, color, sorted_values=False):
    vals = np.array([.22,.78,.34,.56,.18,.46,.30,.66])
    if sorted_values:
        vals = np.sort(vals)[::-1]
    bw = w/(len(vals)*1.35)
    for i,val in enumerate(vals):
        ax.add_patch(Rectangle((x+i*w/len(vals)+bw*.18, y), bw, h*val,
                               facecolor=color, edgecolor="none", zorder=5))


def build_figure():
    fig, ax = plt.subplots(figsize=(7.2, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.14, 1)
    ax.axis("off")

    # The labels follow the A--E method subsections in the manuscript.  Section
    # B deliberately receives most of the canvas because it contains the core
    # learned forecasting mechanism; Section C is the training-only band below.
    stage_header(ax, .015, .135, "A", "Route-only input", BLUE)
    stage_header(ax, .175, .515, "B", "Multi-source route forecasting", TEAL)
    stage_header(ax, .715, .135, "D", "Calibrated selection", GOLD)
    stage_header(ax, .875, .110, "E", "Cache admission", RED)
    ax.text(.327, .895, "B1  Six evidence branches", ha="center", va="center",
            fontsize=5.4, fontweight="bold", color=TEAL)
    ax.text(.595, .895, "B2  Context gate and residual", ha="center", va="center",
            fontsize=5.4, fontweight="bold", color=PURPLE)

    # Stage 1: route-only inputs.
    rounded(ax, .018, .545, .130, .335, fc=BLUE_SOFT, ec=BLUE, lw=.9)
    ax.text(.083, .852, "Observed route tensor", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=INK)
    route_tensor(ax, .036, .635, .094, .166)
    ax.text(.083, .576, "Previous token routes\n(no hidden states or logits)",
            ha="center", va="center", fontsize=5.1, color=INK, linespacing=1.25)

    rounded(ax, .018, .255, .130, .240, fc=GREY_SOFT, ec=BLUE, lw=.8)
    ax.text(.083, .463, "Structural context [5]", ha="center", va="center",
            fontsize=5.6, fontweight="bold", color=INK)
    context = ["layer / L", "expert pool E", "native K / E",
               "prefill entropy", "token position"]
    for i, lab in enumerate(context):
        yy = .423 - i*.034
        ax.add_patch(Circle((.035, yy), .0042, fc=BLUE, ec="none", zorder=6))
        ax.text(.045, yy, lab, ha="left", va="center", fontsize=5.1, color=INK)

    # Stage 2: exact six branches from UniversalRouteCastV3.
    card_w, card_h = .137, .155
    xs = [.180, .328]
    ys = [.710, .525, .340]
    specs = [
        ("Layer popularity", "layer frequency", TEAL, "hist"),
        ("One-hop transition", "P(next | current)", BLUE, "matrix"),
        ("Two-hop transition", "P(next | t-1)", PURPLE, "matrix2"),
        ("Request-prefill", "request frequency", GOLD, "hist"),
        ("GRU history", "route sequence", BLUE, "gru"),
        ("Multiscale freq./recency", "pool {1,2,4,8,H}", RED, "multi"),
    ]
    positions = []
    for idx, spec in enumerate(specs):
        col, row = idx % 2, idx // 2
        x, y = xs[col], ys[row]
        branch_card(ax, x, y, card_w, card_h, *spec)
        positions.append((x, y))

    # Input split and six branch outputs join a vertical score bus.
    arrow(ax, .148, .690, .170, .690, color=BLUE)
    ax.plot([.170,.170], [.405,.787], color=GRID, lw=.7, zorder=1)
    for x, y in positions:
        ax.plot([.170, x], [y+card_h/2, y+card_h/2], color=GRID, lw=.55, zorder=1)
    bus_x = .482
    ax.plot([bus_x,bus_x], [.405,.787], color=TEAL, lw=.85, zorder=2)
    for x, y in positions:
        arrow(ax, x+card_w, y+card_h/2, bus_x, y+card_h/2,
              color=TEAL, lw=.55, style="-")

    # Section B2: summaries -> MLP gate -> probability mixture + residual refinement.
    rounded(ax, .505, .702, .180, .175, fc=PURPLE_SOFT, ec=PURPLE, lw=.85)
    ax.text(.595, .848, "Branch summaries", ha="center", va="center",
            fontsize=5.9, fontweight="bold", color=INK)
    ax.text(.595, .817, "[max probability, margin, entropy] × 6",
            ha="center", va="center", fontsize=5.1, color=MUTED)
    mini_network(ax, .528, .736, .090, .050, PURPLE, layers=(4,3,2))
    ax.text(.642, .760, "MLP\n23 → 64 → 6",
            ha="center", va="center", fontsize=5.1, color=INK)
    ax.text(.595, .718, "g = softmax(MLP([summary, context]))",
            ha="center", va="center", fontsize=5.1, color=PURPLE)

    rounded(ax, .505, .478, .180, .180, fc="white", ec=PURPLE, lw=.85)
    ax.text(.595, .628, "Probability-mixture fusion", ha="center", va="center",
            fontsize=5.9, fontweight="bold", color=INK)
    colors = [TEAL, BLUE, PURPLE, GOLD, "#5A8FC7", RED]
    weights = [.12,.24,.14,.10,.27,.13]
    start = .527
    for i,(c,wgt) in enumerate(zip(colors,weights)):
        ax.add_patch(Rectangle((start+i*.023, .579), .018, .025+wgt*.09,
                               fc=c, ec="none", zorder=5))
    ax.text(.595, .555, "p_mix = weighted sum of branch probabilities",
            ha="center", va="center", fontsize=5.1, color=INK)
    ax.text(.595, .515, "+ 0.1 × residual refinement MLP", ha="center",
            va="center", fontsize=5.1, color=MUTED)

    # Context skip connection to the gate.
    ax.plot([.148,.490,.490,.505], [.375,.375,.790,.790], color=BLUE,
            lw=.65, ls=(0,(3,2)), zorder=2)
    arrow(ax, .497, .790, .505, .790, color=BLUE, lw=.65)
    arrow(ax, bus_x, .743, .505, .743, color=TEAL, lw=.75)
    arrow(ax, bus_x, .568, .505, .568, color=TEAL, lw=.75)
    arrow(ax, .595, .702, .595, .658, color=PURPLE, lw=.75)

    # Section D: calibration and cumulative-mass selection.  Confidence
    # rejection belongs to Section E and is intentionally moved out of here.
    rounded(ax, .720, .675, .125, .202, fc=GOLD_SOFT, ec=GOLD, lw=.85)
    ax.text(.7825, .846, "Marginal calibration", ha="center", va="center",
            fontsize=5.7, fontweight="bold", color=INK)
    probability_bars(ax, .738, .752, .088, .058, GOLD, sorted_values=False)
    ax.text(.7825, .710, "model-specific  T*", ha="center", va="center",
            fontsize=5.1, color=INK)

    rounded(ax, .720, .445, .125, .185, fc="white", ec=GOLD, lw=.85)
    ax.text(.7825, .602, "Score-mass budget", ha="center", va="center",
            fontsize=5.7, fontweight="bold", color=INK)
    probability_bars(ax, .738, .523, .088, .050, GOLD, sorted_values=True)
    ax.plot([.735,.829], [.512,.512], color=RED, lw=.7, ls="--")
    ax.text(.831, .512, "m*", ha="left", va="center", fontsize=5.1, color=RED)
    ax.text(.7825, .475, "smallest K reaching score mass m*", ha="center", va="center",
            fontsize=5.1, color=INK)
    arrow(ax, .685, .568, .706, .568, color=PURPLE)
    ax.plot([.706,.706], [.568,.771], color=LINE, lw=.75)
    arrow(ax, .706, .771, .720, .771, color=LINE)
    arrow(ax, .7825, .675, .7825, .630, color=GOLD)

    # Section E: four explicit decisions, matching the manuscript equations.
    rounded(ax, .880, .790, .103, .087, fc=RED_SOFT, ec=RED, lw=.8)
    ax.text(.9315, .848, "Weight filter", ha="center", va="center",
            fontsize=5.5, fontweight="bold", color=INK)
    ax.text(.9315, .817, "w(a) ≥ τ*", ha="center", va="center",
            fontsize=5.1, color=RED)

    rounded(ax, .880, .680, .103, .078, fc="white", ec=RED, lw=.75)
    ax.text(.9315, .729, "Resident filter", ha="center", va="center",
            fontsize=5.4, fontweight="bold", color=INK)
    ax.text(.9315, .700, "remove cached experts", ha="center", va="center",
            fontsize=5.0, color=MUTED)
    arrow(ax, .9315, .790, .9315, .758, color=RED)

    rounded(ax, .880, .535, .103, .112, fc="white", ec=RED, lw=.8)
    ax.text(.9315, .621, "Admission test", ha="center", va="center",
            fontsize=5.4, fontweight="bold", color=INK)
    for i in range(3):
        fc = GREEN_SOFT if i < 2 else "white"
        ec = GREEN if i < 2 else GRID
        ax.add_patch(Rectangle((.897+i*.024, .575), .019, .027,
                               fc=fc, ec=ec, lw=.5, zorder=5))
    ax.text(.9315, .551, "w(a) > min v(c) + δ", ha="center", va="center",
            fontsize=5.0, color=INK)
    arrow(ax, .9315, .680, .9315, .647, color=RED)

    rounded(ax, .880, .442, .103, .060, fc=GREEN_SOFT, ec=GREEN, lw=.8)
    ax.text(.9315, .472, "Prefetch  |  Abstain", ha="center", va="center",
            fontsize=5.2, color=INK, fontweight="bold")
    arrow(ax, .9315, .535, .9315, .502, color=RED)
    arrow(ax, .845, .548, .867, .548, color=GOLD)
    ax.plot([.867,.867], [.548,.833], color=LINE, lw=.75)
    arrow(ax, .867, .833, .880, .833, color=LINE)

    # The native execution path is visually separate from speculative policy.
    rounded(ax, .720, .275, .263, .102, fc=GREY_SOFT, ec=LINE, lw=.8)
    ax.text(.738, .345, "Frozen native router", ha="left", va="center",
            fontsize=5.6, fontweight="bold", color=INK)
    mini_network(ax, .748, .293, .054, .030, LINE, layers=(3,2,2))
    arrow(ax, .817, .322, .870, .322, color=LINE)
    ax.text(.915, .345, "Native Top-K", ha="center", va="center",
            fontsize=5.0, color=MUTED)
    ax.text(.915, .310, "Executed experts", ha="center", va="center",
            fontsize=5.2, color=GREEN, fontweight="bold")

    # Training-only supervision band.
    rounded(ax, .176, .065, .510, .118, fc="#FAFBFC", ec=GRID,
            lw=.75, radius=.008, z=2)
    ax.add_patch(Circle((.193, .158), .010, facecolor=RED,
                        edgecolor="none", zorder=8))
    ax.text(.193, .158, "C", ha="center", va="center", color="white",
            fontsize=5.5, fontweight="bold", zorder=9)
    ax.text(.209, .158, "CROSS-MODEL LEARNING (TRAINING ONLY)",
            ha="left", va="center", fontsize=5.1, color=RED,
            fontweight="bold")
    ax.text(.190, .118, "Target route  Y", ha="left", va="center",
            fontsize=5.2, color=INK, fontweight="bold")
    arrow(ax, .285, .118, .320, .118, color=RED, lw=.7)
    ax.text(.500, .130,
            "L = 0.20 BCE + 0.25 listwise + 0.35 rank + 0.20 budget",
            ha="center", va="center", fontsize=5.1, color=INK)
    ax.text(.500, .096, "Group-robust optimization across heterogeneous MoE traces",
            ha="center", va="center", fontsize=5.1, color=MUTED)
    ax.plot([.620,.620,.400,.400], [.183,.205,.205,.340], color=RED,
            lw=.65, ls=(0,(3,2)), zorder=2)
    arrow(ax, .400, .340, .400, .348, color=RED, lw=.65)
    ax.text(.508, .210, "backpropagation", ha="center", va="bottom",
            fontsize=5.1, color=RED)

    # Causal inference timeline. It makes the available information and the
    # idealized prefetch window explicit without altering the model graph.
    rounded(ax, .055, -.112, .890, .092, fc="#FAFBFC", ec=GRID,
            lw=.75, radius=.008, z=2)
    timeline = [
        ("Observe routes", "through token t", BLUE),
        ("Forecast", "token t+1", PURPLE),
        ("Select / admit", "rank weights + cache", GOLD),
        ("Transfer window", "idealized completion", RED),
        ("Native route", "reveals demand", LINE),
        ("Execute", "hit or demand load", GREEN),
    ]
    xs = np.linspace(.105, .895, len(timeline))
    for i, (title, subtitle, color) in enumerate(timeline):
        ax.add_patch(Circle((xs[i], -.073), .010, fc=color, ec="white",
                            lw=.5, zorder=7))
        ax.text(xs[i], -.052, title, ha="center", va="bottom", fontsize=4.8,
                color=INK, fontweight="bold")
        ax.text(xs[i], -.093, subtitle, ha="center", va="top", fontsize=4.4,
                color=MUTED)
        if i < len(timeline)-1:
            arrow(ax, xs[i]+.013, -.073, xs[i+1]-.013, -.073,
                  color=LINE, lw=.65)

    ax.text(.500, -.128,
            "RouteCast proposes transfers; the frozen native router determines execution.",
            ha="center", va="center", fontsize=5.0, color=MUTED)

    fig.subplots_adjust(left=.012, right=.992, bottom=.018, top=.985)
    return fig


def main():
    fig = build_figure()
    base = OUT / "Fig1_routecast_framework"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(FINAL / "Fig1_Framework.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved={base}", flush=True)
    print(f"submission_copy={FINAL / 'Fig1_Framework.pdf'}", flush=True)


if __name__ == "__main__":
    main()
