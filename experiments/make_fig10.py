"""
make_fig10.py -- CNN architecture-check figure (fig10) for report section 3.6.

Read-only w.r.t. results/: recomputes NOTHING. Every number on the plot is read
directly from results/*.json (cnn_sanity, e6_cnn, cnn_baseline, sanity_check) or
is a plain difference of two stored numbers, computed here only for display.

This is the exploratory / negative-result figure: the frozen-swap route that works
on the feedforward MLP does NOT transfer to the pooled CNN. Left panel shows the
two p-bit orderings (naive comparator-before-pool, near chance; and the pool-fixed
binarize-after-pool, which climbs but plateaus ~11 pp below deterministic), with a
faded MLP-mid ideal curve for context. Right panel isolates where the residual bias
lives via per-stage noise injection.

Style / palette: identical to experiments/make_figures.py (dataviz skill, validated
default palette instance). Line chart (left) + magnitude bars (right).

Usage:  .venv/bin/python experiments/make_fig10.py
Output: plots/fig10_cnn_transfer.png (200 dpi)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Palette (dataviz skill, references/palette.md -- validated default instance;
# identical values to experiments/make_figures.py)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CAT = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}
DET_COLOR = INK_SECONDARY  # deterministic-reference lines: neutral ink, not a data series

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 100,
})


def style_ax(ax, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)


# ==========================================================================
# FIG 10 -- CNN transfer check (exploratory)
# ==========================================================================
def fig10():
    d_naive = load("cnn_sanity.json")        # naive order: comparator before pool
    d_e6 = load("e6_cnn.json")               # pool-fixed order + diagnostics
    d_cnn_base = load("cnn_baseline.json")   # deterministic CNN reference
    d_mlp = load("sanity_check.json")        # MLP-mid context curve

    fig = plt.figure(figsize=(14.0, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1], wspace=0.24)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1])

    # ---- LEFT: accuracy vs S ------------------------------------------------
    S_cnn = [1, 2, 4, 8, 16, 32, 64]

    # naive order (comparator before pool) -- from cnn_sanity aggregate
    na = d_naive["cnn"]["_aggregate"]
    naive_mean = [na["pdnn_acc_by_S_mean"][str(s)] for s in S_cnn]
    naive_std = [na["pdnn_acc_by_S_std"][str(s)] for s in S_cnn]

    # binarize-after-pool order -- from e6 sanity aggregate
    ap = d_e6["sanity"]["aggregate"]["acc_by_S"]
    afterpool_mean = [ap[str(s)]["mean"] for s in S_cnn]
    afterpool_std = [ap[str(s)]["std"] for s in S_cnn]

    det_cnn = d_cnn_base["cnn_mean"]                        # 0.9871
    afterpool_plateau = afterpool_mean[-1]                  # S=64
    gap_pp = 100.0 * (det_cnn - afterpool_plateau)          # ~11 pp

    # MLP-mid ideal curve (context, faded) -- from sanity_check mid aggregate
    mm = d_mlp["mid"]["_aggregate"]
    S_mlp = [1, 2, 4, 8, 16, 32]
    mlp_mean = [mm["pdnn_acc_by_S_mean"][str(s)] for s in S_mlp]
    det_mlp = mm["det_acc_mean"]                            # 0.9641

    # faded MLP context: plot first so data series sit on top
    axL.plot(S_mlp, mlp_mean, marker="^", ms=5, color=CAT["green"], linestyle=":",
             linewidth=1.6, alpha=0.42, zorder=2,
             label="MLP-mid, ideal noise (context)")
    axL.axhline(det_mlp, color=CAT["green"], linestyle=":", linewidth=1.0,
                alpha=0.30, zorder=1)
    axL.text(S_mlp[-1], det_mlp, f"  MLP-mid det ref = {det_mlp:.4f}", va="bottom",
             ha="right", fontsize=6.9, color=CAT["green"], alpha=0.75)

    # binarize-after-pool (the pool-fixed order) -- climbs but plateaus
    axL.errorbar(S_cnn, afterpool_mean, yerr=afterpool_std, marker="o", ms=5.5,
                 color=CAT["blue"], linewidth=2, capsize=2.5, mfc=CAT["blue"],
                 mec=SURFACE, mew=0.6, zorder=4,
                 label="CNN, binarize-after-pool (order-fixed)")

    # naive order -- near chance, does not climb
    axL.errorbar(S_cnn, naive_mean, yerr=naive_std, marker="s", ms=5.0,
                 color=CAT["red"], linewidth=2, linestyle="-", capsize=2.5,
                 mfc=CAT["red"], mec=SURFACE, mew=0.6, zorder=3,
                 label="CNN, naive order (comparator before pool)")

    # deterministic CNN reference
    axL.axhline(det_cnn, color=DET_COLOR, linestyle="--", linewidth=1.3, zorder=1)
    axL.text(S_cnn[0], det_cnn, f"deterministic CNN ref = {det_cnn:.4f}  ",
             va="bottom", ha="left", fontsize=7.8, color=INK_SECONDARY)

    # plateau annotation
    axL.annotate(
        f"order-fixed plateau: {100*afterpool_plateau:.1f}%\n"
        f"({gap_pp:.1f} pp below det; flat S=64->1024)",
        xy=(64, afterpool_plateau), xytext=(6.5, 0.735), fontsize=7.6,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))
    axL.annotate(
        "naive order: near chance,\ndoes not improve with S",
        xy=(8, naive_mean[3]), xytext=(2.1, 0.26), fontsize=7.6,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))

    axL.set_xscale("log", base=2)
    axL.set_xticks(S_cnn)
    axL.set_xticklabels([str(s) for s in S_cnn])
    axL.set_ylim(0.05, 1.02)
    axL.set_xlabel("# inference samples S, log2 scale")
    axL.set_ylabel("MNIST test accuracy")
    style_ax(axL)
    axL.legend(loc="center right", fontsize=8.0)
    axL.set_title("The frozen-swap route does not transfer to the pooled CNN "
                  "(2 train seeds, 3 noise seeds)", fontsize=9.4, color=INK_SECONDARY)

    # ---- RIGHT: per-stage noise-isolation bars ------------------------------
    diag = d_e6["diagnostic"]
    stage_acc = diag["acc_by_stage"]
    sat = diag["saturation_stats"]
    det_sub = diag["det_acc"]        # 0.984 (2000-image subset)
    S_diag = diag["S"]               # 64
    n_sub = diag["n_subset"]         # 2000

    # bars: which single stage gets p-bit noise (others deterministic)
    bars = [
        ("none",  "none\n(det)",           stage_acc["none"],  None),
        ("conv1", "conv1\nonly",           stage_acc["conv1"], sat["conv1_pool1"]["mean_abs_a"]),
        ("conv2", "conv2\nonly",           stage_acc["conv2"], sat["conv2_pool2"]["mean_abs_a"]),
        ("fc1",   "fc1\nonly",             stage_acc["fc1"],   sat["fc1"]["mean_abs_a"]),
        ("all",   "all\nstages",           stage_acc["all"],   None),
    ]
    xs = np.arange(len(bars))
    vals = [b[2] for b in bars]
    # highlight the damaging stages (conv1 and all) in red; near-det stages blue
    colors = [CAT["red"] if key in ("conv1", "all") else CAT["blue"]
              for key, _, _, _ in bars]

    axR.bar(xs, vals, width=0.66, color=colors, zorder=3,
            edgecolor=SURFACE, linewidth=0.8)
    for x, (_, _, v, _) in zip(xs, bars):
        axR.text(x, v + 0.004, f"{100*v:.2f}", ha="center", va="bottom",
                 fontsize=8.0, color=INK)
    # mean|tanh| saturation annotation under conv/fc bars
    for x, (_, _, v, mabs) in zip(xs, bars):
        if mabs is not None:
            axR.text(x, 0.888, f"mean|tanh|\n= {mabs:.3f}", ha="center", va="top",
                     fontsize=6.7, color=INK_SECONDARY)

    axR.axhline(det_sub, color=DET_COLOR, linestyle="--", linewidth=1.2, zorder=2)
    axR.text(len(bars) - 0.5, det_sub, f"  det = {100*det_sub:.2f}", va="bottom",
             ha="right", fontsize=7.3, color=INK_SECONDARY)

    axR.set_xticks(xs)
    axR.set_xticklabels([b[1] for b in bars], fontsize=8.2)
    axR.set_ylim(0.86, 1.0)
    axR.set_ylabel("MNIST accuracy (2k subset)")
    axR.set_xlabel("stage receiving p-bit noise (order-fixed; others deterministic)")
    style_ax(axR)
    axR.set_title(f"Residual bias is concentrated in the weakly-saturated first "
                  f"conv stage\n(S={S_diag}, {n_sub}-image subset, cnn_s0): conv1-only "
                  f"reproduces nearly the whole all-stages drop", fontsize=8.4,
                  color=INK_SECONDARY)

    fig.suptitle("Architecture check (exploratory): pooled CNN breaks the frozen-swap "
                 "route via a pool-order bias, then a first-stage Jensen bias",
                 fontsize=12.5, fontweight="bold", y=1.02)
    path = os.path.join(PLOTS, "fig10_cnn_transfer.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    print(f"  det_cnn={det_cnn:.4f}  afterpool_plateau(S=64)={afterpool_plateau:.4f}  "
          f"gap={gap_pp:.2f} pp")
    print(f"  stage bars: " + ", ".join(f"{k}={100*v:.2f}" for k, _, v, _ in bars))


if __name__ == "__main__":
    fig10()
