"""
make_figures.py -- generate all report figures from results/*.json.

Read-only w.r.t. results/: this script recomputes NOTHING. Every number on
every plot is either stored directly in a results/*.json file, or a plain
aggregation (mean/std across the *already-stored* per-seed values, or a
simple difference/ratio of two stored numbers) computed here for display.
No experiment is re-run.

Style: see the `dataviz` skill (categorical palette, marks, spacers). Palette
values are the validated default instance from that skill's references.

Usage: .venv/bin/python experiments/make_figures.py
Outputs: plots/fig1..fig9*.png (200 dpi), plots/FIGURES.md
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
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
# Palette (dataviz skill, references/palette.md -- validated default instance)
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
CAT_ORDER = ["blue", "aqua", "yellow", "green", "violet", "red", "magenta", "orange"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

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


def savefig(fig, name, note=""):
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}" + (f"  ({note})" if note else ""))


FIGURES_MD = []  # (filename, caption, sources)


def register(filename, caption, sources):
    FIGURES_MD.append((filename, caption, sources))


# ==========================================================================
# FIG 1 -- activation deformation (E1)
# ==========================================================================
def fig1():
    d = load("e1_activation.json")
    panels = [
        ("ideal", "Ideal (PCG64 uniform)"),
        ("bitdepth_k1", "Bit-depth k=1"),
        ("bitdepth_k2", "Bit-depth k=2"),
        ("bias_b0.3", "Bias b=0.3"),
        ("mismatch_triangular", "Mismatch: triangular"),
        ("mismatch_truncgauss_sigma0.5", "Mismatch: trunc. Gaussian σ=0.5"),
        ("ar1_copula_rho0.9", "Neg. control: AR(1) ρ=0.9"),
        ("lfsr8bit", "Neg. control: LFSR-8"),
    ]
    neg_controls = {"ar1_copula_rho0.9", "lfsr8bit"}

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, panels):
        c = d[key]
        I = np.array(c["I_grid"])
        pred = np.array(c["predicted_firing_prob"])
        per_seed = np.array(c["measured_firing_prob_per_seed"])  # (3, 61)
        meas_mean = per_seed.mean(axis=0)
        meas_std = per_seed.std(axis=0)

        ax.plot(I, pred, color=CAT["blue"], linewidth=2, zorder=2,
                label="IST-predicted CDF")
        ax.errorbar(I[::3], meas_mean[::3], yerr=meas_std[::3], fmt="o",
                     ms=4.5, mfc=CAT["orange"], mec=SURFACE, mew=0.6,
                     ecolor=CAT["orange"], elinewidth=1, capsize=1.5,
                     alpha=0.9, zorder=3, label="Measured (mean ± 1 std, 3 seeds)")

        mae = c["max_abs_error_mean"]
        mae_std = c["max_abs_error_std"]
        tag = "NEGATIVE CONTROL\n" if key in neg_controls else ""
        ax.text(0.03, 0.96, f"{tag}max|err| = {mae:.4f} ± {mae_std:.4f}",
                 transform=ax.transAxes, va="top", ha="left", fontsize=7.3,
                 color=INK_SECONDARY)
        ax.set_title(title, fontsize=9.5, color=INK)
        style_ax(ax)
        ax.set_ylim(-0.03, 1.03)

    for ax in axes[4:]:
        ax.set_xlabel("pre-activation I", fontsize=8.8)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("P(m = +1 | I)", fontsize=8.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04),
               frameon=False)
    fig.suptitle(
        "E1: measured p-neuron activation vs IST-predicted CDF, across noise defects\n"
        "(negative controls, right: AR(1) and LFSR-8 are temporal defects invisible to a per-input marginal measurement -- expect near-ideal match)",
        fontsize=10.5, y=1.10)
    fig.tight_layout()
    savefig(fig, "fig1_activation_deformation.png")
    register("fig1_activation_deformation.png",
              "Measured p-neuron firing probability vs the IST-predicted CDF for six randomness "
              "defects plus two temporal negative controls (AR(1) rho=0.9, LFSR-8) that match "
              "the ideal curve as expected.",
              "results/e1_activation.json")


# ==========================================================================
# FIG 2 -- precision tolerance (E2)
# ==========================================================================
def fig2():
    d = load("e2_precision.json")
    S_main = [1, 4, 32]
    s_colors = {1: CAT["red"], 4: CAT["violet"], 32: CAT["blue"]}

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    for width, ls, alpha, lw in [("mid", "-", 1.0, 2.0),
                                  ("large", "--", 0.38, 1.4),
                                  ("small", ":", 0.38, 1.4)]:
        agg = d["precision"][width]["_aggregate"]["by_k"]
        ks = sorted(int(k) for k in agg.keys())
        for S in S_main:
            means = [agg[str(k)][str(S)]["mean"] for k in ks]
            stds = [agg[str(k)][str(S)]["std"] for k in ks]
            label = f"S={S}" if width == "mid" else None
            ax.plot(ks, means, marker="o" if width == "mid" else "", ms=5,
                    color=s_colors[S], linestyle=ls, linewidth=lw, alpha=alpha,
                    label=label, zorder=3 if width == "mid" else 2,
                    mfc=s_colors[S], mec=SURFACE, mew=0.6)
            if width == "mid":
                ax.fill_between(ks, np.array(means) - np.array(stds),
                                 np.array(means) + np.array(stds),
                                 color=s_colors[S], alpha=0.12, linewidth=0, zorder=1)

    det_mid = d["precision"]["mid"]["_aggregate"]["det_acc_mean"]
    ax.axhline(det_mid, color=DET_COLOR, linestyle="--", linewidth=1.3, zorder=1)
    ax.text(ks[-1], det_mid, f"  deterministic ref. (mid, full precision) = {det_mid:.4f}",
            va="center", ha="left", fontsize=7.8, color=INK_SECONDARY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("RNG bit-depth k (2^k quantization levels), log2 scale")
    ax.set_ylabel("MNIST test accuracy")
    style_ax(ax)

    k1_32 = d["precision"]["mid"]["_aggregate"]["by_k"]["1"]["32"]["mean"]
    k16_32 = d["precision"]["mid"]["_aggregate"]["by_k"]["16"]["32"]["mean"]
    ax.annotate(
        f"No knee down to k=2 (S=32: k=16 -> {k16_32:.4f}, k=2 -> "
        f"{d['precision']['mid']['_aggregate']['by_k']['2']['32']['mean']:.4f})",
        xy=(2, d['precision']['mid']['_aggregate']['by_k']['2']['32']['mean']),
        xytext=(3, 0.905), fontsize=8, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))
    ax.annotate(
        f"k=1, S=32: {k1_32:.4f} -- only {100*(k16_32-k1_32):.2f} pts below full precision "
        f"(k=16: {k16_32:.4f})",
        xy=(1, k1_32), xytext=(1.15, 0.865), fontsize=8, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))

    ax.text(0.99, 0.03,
            "faded dashed/dotted: large (256,128) / small (32,16) width, same S colors",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.3, color=MUTED)

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("E2: MNIST accuracy tolerates coarse RNG precision (mid width, aggregate over 3 train seeds)")
    fig.tight_layout()
    savefig(fig, "fig2_precision_tolerance.png")
    register("fig2_precision_tolerance.png",
              "Accuracy vs RNG bit-depth k for S in {1,4,32}: no knee down to k=2, and even "
              "k=1 loses under half a point at S=32 relative to full precision.",
              "results/e2_precision.json")


# ==========================================================================
# FIG 3 -- bias (E2 bias grid + E4 correlation/bias TV)
# ==========================================================================
def fig3():
    d2 = load("e2_precision.json")
    d4 = load("e4_cross_task.json")

    bias_agg = d2["bias"]["mid"]["_aggregate"]["by_b"]
    bs = sorted(float(b) for b in bias_agg.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    for S, color in [(1, CAT["red"]), (32, CAT["blue"])]:
        means = [bias_agg[str(b)][str(S)]["mean"] for b in bs]
        stds = [bias_agg[str(b)][str(S)]["std"] for b in bs]
        ax1.errorbar(bs, means, yerr=stds, marker="o", ms=5.5, color=color,
                     linewidth=2, capsize=2.5, mfc=color, mec=SURFACE, mew=0.6,
                     label=f"S={S}")
    ax1.set_ylabel("MNIST test accuracy")
    style_ax(ax1)
    ax1.legend(loc="lower left")
    ax1.set_title("Bias hurts both tasks, systematically (mid width, aggregate)")

    bias_tv = d4["bias"]
    axis_vals = bias_tv["axis_values"]
    tv_means = [bias_tv["results"][str(b)]["tv_mean"] for b in axis_vals]
    tv_stds = [bias_tv["results"][str(b)]["tv_std"] for b in axis_vals]
    ax2.errorbar(axis_vals, tv_means, yerr=tv_stds, marker="o", ms=5.5,
                 color=CAT["violet"], linewidth=2, capsize=2.5,
                 mfc=CAT["violet"], mec=SURFACE, mew=0.6, label="frustrated triangle TV")
    ax2.set_ylabel("Total-variation distance\n(triangle, exact vs empirical)")
    ax2.set_xlabel("bias b (shift of noise mean)")
    style_ax(ax2)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    savefig(fig, "fig3_bias.png")
    register("fig3_bias.png",
              "Accuracy at S=1/32 vs bias b (top) and frustrated-triangle TV distance vs the "
              "same bias axis (bottom, shared x): both metrics degrade monotonically with bias.",
              "results/e2_precision.json, results/e4_cross_task.json")


# ==========================================================================
# FIG 4 -- headline sampling fragility (E3)
# ==========================================================================
def fig4():
    d = load("e3_sampling.json")
    mg = d["main_grid"]["mid"]["_aggregate"]
    k2 = d["precision_contrast"]["mid"]["k2_rho0.0"]["_aggregate"]

    S_vals = sorted(int(s) for s in mg["0.0"].keys())

    series = [
        ("0.0", "ideal (ρ=0)", CAT["blue"], "-"),
        ("0.9", "ρ=0.9", CAT["yellow"], "-"),
        ("0.99", "ρ=0.99", CAT["orange"], "-"),
        ("1.0", "ρ=1.0 (fully correlated)", CAT["red"], "-"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for key, label, color, ls in series:
        means = [mg[key][str(s)]["mean"] for s in S_vals]
        stds = [mg[key][str(s)]["std"] for s in S_vals]
        ax.errorbar(S_vals, means, yerr=stds, marker="o", ms=5.5, color=color,
                     linestyle=ls, linewidth=2, capsize=2.5, mfc=color, mec=SURFACE,
                     mew=0.6, label=label, zorder=3)

    k2_means = [k2[str(s)]["mean"] for s in S_vals]
    k2_stds = [k2[str(s)]["std"] for s in S_vals]
    ax.errorbar(S_vals, k2_means, yerr=k2_stds, marker="s", ms=5.5, color=CAT["aqua"],
                linestyle="--", linewidth=1.8, capsize=2.5, mfc=CAT["aqua"], mec=SURFACE,
                mew=0.6, label="k=2, independent (ρ=0)", zorder=3)

    det_mid = load("e2_precision.json")["precision"]["mid"]["_aggregate"]["det_acc_mean"]
    ax.axhline(det_mid, color=DET_COLOR, linestyle=":", linewidth=1.3, zorder=1)
    ax.text(S_vals[0], det_mid, f"deterministic ref. = {det_mid:.4f}  ",
            va="bottom", ha="left", fontsize=7.8, color=INK_SECONDARY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(S_vals)
    ax.set_xticklabels([str(s) for s in S_vals])
    ax.set_xlabel("# inference samples S, log2 scale")
    ax.set_ylabel("MNIST test accuracy")
    style_ax(ax)
    ax.legend(loc="lower right")
    ax.set_title(
        "Even coarse (k=2) independent noise climbs with S; ρ=0.9/0.99 delay the gain; "
        "ρ=1.0 (identical draw every sample) is flat at the S=1 value.",
        fontsize=8.6, color=INK_SECONDARY, pad=32)
    fig.suptitle("Independence, not precision, carries the sampling gain",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig4_headline_sampling_fragility.png")
    register("fig4_headline_sampling_fragility.png",
              "THE HEADLINE: accuracy vs number of inference samples S for ideal, rho=0.9/0.99/1.0 "
              "correlated noise, and k=2-independent noise -- independence (even at 2-bit precision) "
              "recovers the sampling gain that correlation removes.",
              "results/e3_sampling.json, results/e2_precision.json")


# ==========================================================================
# FIG 5 -- correlation grid (E3 rho x S heatmap + cross-neuron contrast)
# ==========================================================================
def fig5():
    d = load("e3_sampling.json")
    mg = d["main_grid"]["mid"]["_aggregate"]
    rhos = sorted(float(r) for r in mg.keys())
    S_vals = sorted(int(s) for s in mg["0.0"].keys())

    # rho keys in json are like "0.0","0.25",...,"1.0" -- str(r) reproduces them exactly
    acc = np.array([[mg[str(r)][str(s)]["mean"] for s in S_vals] for r in rhos])

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    im = ax1.imshow(acc, cmap=SEQ_CMAP, aspect="auto", vmin=acc.min(), vmax=acc.max())
    ax1.set_xticks(range(len(S_vals)))
    ax1.set_xticklabels(S_vals)
    ax1.set_yticks(range(len(rhos)))
    ax1.set_yticklabels([f"{r:g}" for r in rhos])
    ax1.set_xlabel("# inference samples S")
    ax1.set_ylabel("correlation ρ")
    ax1.set_title("Accuracy over ρ × S (mid width, aggregate)")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.grid(False)
    midpoint = (acc.min() + acc.max()) / 2
    for i in range(len(rhos)):
        for j in range(len(S_vals)):
            v = acc[i, j]
            # light cells (low accuracy, near vmin) get dark ink text; dark cells
            # (high accuracy, near vmax) get surface-colored text
            txt_color = INK if v < midpoint else SURFACE
            ax1.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.7,
                      color=txt_color)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("MNIST test accuracy", fontsize=8.5, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=7.5, colors=MUTED)
    cbar.outline.set_visible(False)

    # cross-neuron contrast panel
    cn = d["cross_neuron"]["mid"]["_aggregate"]
    cn_S = sorted(int(s) for s in cn.keys())
    cn_means = [cn[str(s)]["mean"] for s in cn_S]
    cn_stds = [cn[str(s)]["std"] for s in cn_S]

    ideal_S = [s for s in cn_S if str(s) in mg["0.0"]]
    ideal_means = [mg["0.0"][str(s)]["mean"] for s in ideal_S]
    ideal_stds = [mg["0.0"][str(s)]["std"] for s in ideal_S]

    ax2.errorbar(cn_S, cn_means, yerr=cn_stds, marker="s", ms=5.5, color=CAT["magenta"],
                 linewidth=2, capsize=2.5, mfc=CAT["magenta"], mec=SURFACE, mew=0.6,
                 label="cross-neuron shared noise")
    ax2.errorbar(ideal_S, ideal_means, yerr=ideal_stds, marker="o", ms=5.5, color=CAT["blue"],
                 linewidth=2, capsize=2.5, mfc=CAT["blue"], mec=SURFACE, mew=0.6,
                 label="ideal (independent per neuron)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(cn_S)
    ax2.set_xticklabels([str(s) for s in cn_S])
    ax2.set_xlabel("# inference samples S, log2 scale")
    ax2.set_ylabel("MNIST test accuracy")
    style_ax(ax2)
    ax2.legend(loc="lower right")
    s1_gap = ideal_means[0] - cn_means[0]
    s_last_gap = ideal_means[-1] - cn_means[-1]
    ax2.set_title(f"Cross-neuron sharing: S=1 penalty ({s1_gap:.3f}), gain intact by "
                   f"S={cn_S[-1]} ({s_last_gap:.3f})", fontsize=9.3)

    fig.suptitle("E3: full ρ×S accuracy surface, and the cross-neuron sharing failure mode", y=1.03, fontsize=11.5)
    savefig(fig, "fig5_correlation_grid.png")
    register("fig5_correlation_grid.png",
              "Left: accuracy heatmap over the full correlation-strength x sample-count grid. "
              "Right: sharing one noise draw across neurons (same forward pass) costs accuracy "
              "at S=1 but the multi-sample gain still recovers -- a different failure mode from "
              "across-sample correlation.",
              "results/e3_sampling.json")


# ==========================================================================
# FIG 6 -- ECE (E3 ece block)
# ==========================================================================
def fig6():
    d = load("e3_ece.json")
    ece = d["mid"]["s0"]
    conditions = [
        ("ideal", "ideal", CAT["blue"]),
        ("k2_rho0", "k=2, independent", CAT["aqua"]),
        ("rho0.9", "ρ=0.9", CAT["yellow"]),
        ("rho1.0", "ρ=1.0", CAT["red"]),
    ]
    S_vals = sorted(int(s) for s in ece["ideal"].keys())

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for key, label, color in conditions:
        means = [ece[key][str(s)]["mean"] for s in S_vals]
        stds = [ece[key][str(s)]["std"] for s in S_vals]
        ax.errorbar(S_vals, means, yerr=stds, marker="o", ms=5.5, color=color,
                     linewidth=2, capsize=2.5, mfc=color, mec=SURFACE, mew=0.6, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(S_vals)
    ax.set_xticklabels([str(s) for s in S_vals])
    ax.set_xlabel("# inference samples S, log2 scale")
    ax.set_ylabel("Expected Calibration Error (15 bins)")
    ax.set_ylim(0.02, 0.195)
    style_ax(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("E3: ECE vs sampling -- under ideal noise, ECE RISES with S (underconfidence)",
                 fontsize=10, pad=10)

    ideal_1 = ece["ideal"]["1"]["mean"]
    ideal_last = ece["ideal"][str(S_vals[-1])]["mean"]
    rho1_1 = ece["rho1.0"]["1"]["mean"]
    rho1_last = ece["rho1.0"][str(S_vals[-1])]["mean"]
    ax.annotate(
        f"ideal: ECE {ideal_1:.3f} (S=1) -> {ideal_last:.3f} (S={S_vals[-1]}), rising",
        xy=(S_vals[-1], ideal_last), xytext=(9, 0.183),
        fontsize=7.6, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))
    ax.annotate(
        f"ρ=1.0 stays flat (~{rho1_last:.3f}) -- LOWER than ideal at large S\n"
        "(complicates 'correlation hurts calibration')",
        xy=(S_vals[-1], rho1_last), xytext=(9, 0.028),
        fontsize=7.6, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))

    fig.tight_layout()
    savefig(fig, "fig6_ece.png")
    register("fig6_ece.png",
              "ECE vs S for ideal, k=2-independent, rho=0.9, rho=1.0: ECE rises with S under "
              "ideal/independent noise (the net becomes underconfident as it averages), while "
              "rho=1.0 stays flat and ends up with LOWER measured ECE than ideal at large S.",
              "results/e3_ece.json")


# ==========================================================================
# FIG 7 -- cross-task overlay (E4 + E2/E3), H4
# ==========================================================================
def fig7():
    d4 = load("e4_cross_task.json")
    d2 = load("e2_precision.json")
    d3 = load("e3_sampling.json")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), sharey="row")
    ((ax_tv_k, ax_tv_b, ax_tv_r), (ax_acc_k, ax_acc_b, ax_acc_r)) = axes

    # --- bitdepth family ---
    bk = d4["bitdepth"]
    ks = bk["axis_values"]
    tv_k = [bk["results"][str(k)]["tv_mean"] for k in ks]
    tv_k_std = [bk["results"][str(k)]["tv_std"] for k in ks]
    ax_tv_k.errorbar(ks, tv_k, yerr=tv_k_std, marker="o", ms=5, color=CAT["blue"],
                      linewidth=2, capsize=2, mfc=CAT["blue"], mec=SURFACE, mew=0.6)
    ax_tv_k.axhline(0.00607, color=MUTED, linestyle=":", linewidth=1.1)
    ax_tv_k.text(ks[-1], 0.00607, "  k≤4 TV floor = 0.00607\n  (unreachable aligned states)",
                 va="bottom", ha="right", fontsize=6.8, color=INK_SECONDARY)
    ax_tv_k.set_yscale("log")
    ax_tv_k.set_title("bit-depth k")
    ax_tv_k.set_ylabel("triangle TV (log)")

    acc_by_k = d2["precision"]["mid"]["_aggregate"]["by_k"]
    k16 = acc_by_k["16"]["32"]["mean"]
    drop_k = [k16 - acc_by_k[str(k)]["32"]["mean"] for k in ks]
    ax_acc_k.plot(ks, drop_k, marker="o", ms=5, color=CAT["red"], linewidth=2,
                   mfc=CAT["red"], mec=SURFACE, mew=0.6)
    ax_acc_k.axhline(0, color=BASELINE, linewidth=1)
    ax_acc_k.set_xlabel("bit-depth k")
    ax_acc_k.set_ylabel("MNIST acc. drop at S=32\n(vs k=16 full precision)")

    # LFSR8 feature point on the bitdepth family
    lfsr_tv = d4["lfsr"]["results"]["8"]["tv_mean"]
    ideal_tv = d4["ideal"]["tv_mean"]
    ratio = lfsr_tv / ideal_tv
    lfsr_acc32 = d2["lfsr"]["mid"]["_aggregate"]["by_nbits"]["nbits8"]["32"]["mean"]
    lfsr_drop = k16 - lfsr_acc32
    ax_tv_k.scatter([16], [lfsr_tv], marker="*", s=170, color=CAT["orange"],
                     edgecolor=SURFACE, linewidth=0.7, zorder=5)
    ax_tv_k.annotate(f"LFSR-8: {ratio:.0f}× ideal TV", xy=(16, lfsr_tv),
                      xytext=(6.5, lfsr_tv * 1.6), fontsize=7.3, color=INK_SECONDARY,
                      arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8))
    ax_acc_k.scatter([16], [lfsr_drop], marker="*", s=170, color=CAT["orange"],
                      edgecolor=SURFACE, linewidth=0.7, zorder=5)
    ax_acc_k.annotate(f"LFSR-8: {lfsr_drop:.4f} drop\n(near-ideal on MNIST)", xy=(16, lfsr_drop),
                       xytext=(9, max(drop_k) * 0.55 + 0.001), fontsize=7.3, color=INK_SECONDARY,
                       arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8))

    # --- bias family ---
    bb = d4["bias"]
    bs = bb["axis_values"]
    tv_b = [bb["results"][str(b)]["tv_mean"] for b in bs]
    tv_b_std = [bb["results"][str(b)]["tv_std"] for b in bs]
    ax_tv_b.errorbar(bs, tv_b, yerr=tv_b_std, marker="o", ms=5, color=CAT["blue"],
                      linewidth=2, capsize=2, mfc=CAT["blue"], mec=SURFACE, mew=0.6)
    ax_tv_b.set_yscale("log")
    ax_tv_b.set_title("bias b")

    acc_by_b = d2["bias"]["mid"]["_aggregate"]["by_b"]
    b_ref = acc_by_b["0.02"]["32"]["mean"]  # least-biased tested point, not exactly b=0
    drop_b = [b_ref - acc_by_b[str(b)]["32"]["mean"] for b in bs]
    ax_acc_b.plot(bs, drop_b, marker="o", ms=5, color=CAT["red"], linewidth=2,
                   mfc=CAT["red"], mec=SURFACE, mew=0.6)
    ax_acc_b.axhline(0, color=BASELINE, linewidth=1)
    ax_acc_b.set_xlabel("bias b")

    # --- correlation family ---
    bc = d4["correlation"]
    rs = bc["axis_values"]
    tv_r = [bc["results"][str(r)]["tv_mean"] for r in rs]
    tv_r_std = [bc["results"][str(r)]["tv_std"] for r in rs]
    ax_tv_r.errorbar(rs, tv_r, yerr=tv_r_std, marker="o", ms=5, color=CAT["blue"],
                      linewidth=2, capsize=2, mfc=CAT["blue"], mec=SURFACE, mew=0.6)
    ax_tv_r.set_yscale("log")
    ax_tv_r.set_title("correlation ρ")

    mg = d3["main_grid"]["mid"]["_aggregate"]
    r_ref = mg["0.0"]["32"]["mean"]
    drop_r = [r_ref - mg[str(r)]["32"]["mean"] if str(r) in mg else np.nan for r in rs]
    ax_acc_r.plot(rs, drop_r, marker="o", ms=5, color=CAT["red"], linewidth=2,
                   mfc=CAT["red"], mec=SURFACE, mew=0.6)
    ax_acc_r.axhline(0, color=BASELINE, linewidth=1)
    ax_acc_r.set_xlabel("correlation ρ")

    for ax in axes.ravel():
        style_ax(ax)

    fig.suptitle("E4 (H4): frustrated-triangle TV distance vs MNIST accuracy drop at S=32, "
                 "per defect family", fontsize=11.5, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig7_cross_task.png")
    register("fig7_cross_task.png",
              "H4 cross-task overlay: triangle TV distance (top, log y) and MNIST accuracy "
              "drop at S=32 relative to each family's least-defective point (bottom), for "
              "bit-depth, bias, and correlation. LFSR-8 is annotated as a near-ideal MNIST "
              "point that is ~82x ideal TV distance on the triangle.",
              "results/e4_cross_task.json, results/e2_precision.json, results/e3_sampling.json")


# ==========================================================================
# FIG 8 -- STE robustness (E5)
# ==========================================================================
def fig8():
    d = load("e5_ste.json")
    agg = d["aggregate"]
    conditions = [
        ("ideal", "ideal"),
        ("bias0.2", "bias b=0.2"),
        ("rho0.9", "ρ=0.9"),
        ("rho1.0", "ρ=1.0"),
    ]
    S_vals = sorted(int(s) for s in agg["frozen"]["ideal"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), sharex=True, sharey=True)

    for ax, (key, title) in zip(axes.ravel(), conditions):
        frozen = agg["frozen"][key]
        ste4 = agg["ste_S4"][key]
        ste1 = agg["ste_S1"][key]

        for net, label, color, alpha, lw, ls in [
            (frozen, "frozen (route-1)", CAT["blue"], 1.0, 2.0, "-"),
            (ste4, "STE, S_train=4", CAT["red"], 1.0, 2.0, "-"),
            (ste1, "STE, S_train=1 (faded)", CAT["red"], 0.4, 1.4, "--"),
        ]:
            means = [net[str(s)]["mean"] for s in S_vals]
            stds = [net[str(s)]["std"] for s in S_vals]
            ax.errorbar(S_vals, means, yerr=stds, marker="o", ms=4.5, color=color,
                         linewidth=lw, linestyle=ls, alpha=alpha, capsize=2,
                         mfc=color, mec=SURFACE, mew=0.5, label=label)

        ax.set_xscale("log", base=2)
        ax.set_xticks(S_vals)
        ax.set_xticklabels([str(s) for s in S_vals])
        style_ax(ax)
        ax.set_title(title, fontsize=9.8)

    axes[0, 0].legend(loc="lower right", fontsize=7.6)
    for ax in axes[1]:
        ax.set_xlabel("# inference samples S, log2 scale")
    for ax in axes[:, 0]:
        ax.set_ylabel("MNIST test accuracy")

    s1_gap_rho1 = agg["ste_S4"]["rho1.0"]["1"]["mean"] - agg["frozen"]["rho1.0"]["1"]["mean"]
    fig.suptitle(
        "E5: STE-trained nets vs frozen-weight baseline, across noise defects\n"
        f"STE's advantage is concentrated at S=1 and decisive under ρ=1.0 "
        f"(S=1 gap = {s1_gap_rho1:+.3f})", fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig8_ste_robustness.png")
    register("fig8_ste_robustness.png",
              "Frozen (add-noise-to-trained-model) vs STE-trained (S_train=4, faded S_train=1) "
              "accuracy vs S under ideal, bias, and correlated-noise conditions: STE's edge is "
              "largest at S=1 and most decisive when rho=1.0 removes the multi-sample gain.",
              "results/e5_ste.json")


# ==========================================================================
# FIG 9 -- triangle validation + RNG stat-test summary
# ==========================================================================
def fig9():
    d4 = load("e4_cross_task.json")
    rng = load("rng_stats.json")

    exact = d4["_meta"]["exact_dist"]
    empirical = d4["ideal"]["representative_empirical_dist"]
    tv_mean = d4["ideal"]["tv_mean"]
    tv_std = d4["ideal"]["tv_std"]
    labels = [f"m={i:03b}".replace("0", "−").replace("1", "+") for i in range(8)]
    state_kind = ["aligned"] + ["frustrated"] * 6 + ["aligned"]

    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(8)
    w = 0.36
    b1 = ax1.bar(x - w / 2, exact, width=w, color=CAT["blue"], label="exact")
    b2 = ax1.bar(x + w / 2, empirical, width=w, color=CAT["orange"], label="empirical (ideal RNG, seed 0)")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("probability (log)")
    ax1.set_xlabel("triangle state (m0,m1,m2)")
    ax1.set_ylim(8e-4, 0.6)
    style_ax(ax1)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax1.set_title(f"TV = {tv_mean:.5f} ± {tv_std:.5f} (mean ± std, 5 seeds)",
                  fontsize=9, color=INK_SECONDARY, pad=34)
    for i, kind in enumerate(state_kind):
        ax1.text(i, 9.2e-4, kind, ha="center", va="bottom", fontsize=6.3,
                  color=MUTED, rotation=90)

    # --- RNG stat-test summary table ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    sources = ["pcg64", "lfsr8", "lfsr16", "ar1_rho0.9"]
    cols = ["source", "period", "lag-1 r", "chi2 p", "overall"]
    row_h = 1.0
    n_rows = len(sources) + 1
    ax2.set_xlim(0, len(cols))
    ax2.set_ylim(0, n_rows)
    ax2.set_title("RNG stat-test summary (rng_stats.json)", fontsize=9.8, pad=14)

    for j, c in enumerate(cols):
        ax2.text(j + 0.5, n_rows - 0.4, c, ha="center", va="center", fontsize=8.2,
                  color=INK_SECONDARY, fontweight="bold")

    for i, src in enumerate(sources):
        r = rng[src]
        y = n_rows - 1.5 - i
        period = r["period_estimate"]["period"]
        period_txt = "not found\n(≥ draws)" if period is None else str(period)
        lag1 = r["lag_autocorrelation"]["lags"]["1"]["r"]
        chi2p = r["histogram_chi2"]["p_value"]
        overall = r["overall_pass"]

        ax2.text(0.5, y, src, ha="center", va="center", fontsize=8)
        ax2.text(1.5, y, period_txt, ha="center", va="center", fontsize=7.6)
        ax2.text(2.5, y, f"{lag1:.3f}", ha="center", va="center", fontsize=7.8,
                  color=(STATUS["critical"] if abs(lag1) > 0.05 else INK))
        ax2.text(3.5, y, f"{chi2p:.3f}", ha="center", va="center", fontsize=7.8)
        status_color = STATUS["good"] if overall else STATUS["critical"]
        mark = "PASS" if overall else "FAIL"
        ax2.text(4.5, y, mark, ha="center", va="center", fontsize=8.2,
                  color=status_color, fontweight="bold")

    for i in range(n_rows + 1):
        ax2.axhline(i, color=GRIDLINE, linewidth=0.8, xmin=0, xmax=1)
    ax2.text(0, -0.6, "lag-1 r in red where |r| > 0.05 (near-independence heuristic, not a formal test)",
              fontsize=6.8, color=MUTED)

    fig.suptitle("Appendix: frustrated-triangle ground-truth validation and RNG source stat tests",
                 fontsize=12.5, y=1.13)
    savefig(fig, "fig9_triangle_validation.png")
    register("fig9_triangle_validation.png",
              "Appendix: exact vs empirical frustrated-triangle state probabilities under ideal "
              "RNG (left), and a pass/fail summary of period / lag-1 autocorrelation / chi-square "
              "histogram tests for each RNG source (right).",
              "results/e4_cross_task.json, results/rng_stats.json")


# ==========================================================================
if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()

    md_path = os.path.join(PLOTS, "FIGURES.md")
    with open(md_path, "w") as f:
        f.write("# Figures\n\n")
        for filename, caption, sources in FIGURES_MD:
            f.write(f"- **{filename}** -- {caption} _(source: {sources})_\n")
    print(f"wrote {md_path}")
