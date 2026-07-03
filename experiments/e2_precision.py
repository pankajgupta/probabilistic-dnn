"""E2 (H2): how much RNG precision does the p-DNN need?

Three sub-experiments, all on the FULL 10k MNIST test set with frozen
weights (deterministic training, add-noise-at-inference route):

  1. PRECISION GRID -- bitdepth k in {1,2,3,4,5,6,8,12,16}, S in
     {1,2,4,8,16,32}, all 3 widths x 3 training seeds, 5 noise seeds/cell.
     cfg.rho = 0.0 (strictly independent draws per sample -- the H2
     precision-only condition, not to be confused with H3 correlation).
  2. BIAS GRID -- bias b in {0.02,0.05,0.1,0.2,0.3,0.5} (bitdepth off),
     S in {1,32}, all widths/seeds, 5 noise seeds/cell.
  3. LFSR CONDITIONS -- GaloisLFSR(nbits=8) and GaloisLFSR(nbits=16) as
     cfg.uniform_source, S in {1,2,4,8,16,32}, mid width only, x 3
     training seeds x 3 LFSR seeds (distinct initial register states,
     used in place of independent "noise seeds" here since the LFSR
     itself is the noise realization). A fresh GaloisLFSR instance is
     built per evaluate() call so state never leaks across cells.

     Axis convention: make_noise() draws a flat uniform_source(n) buffer
     and reshapes it to (S, B, H) with numpy's default C order (see
     models/pdnn.py::_draw_uniform -> .reshape(shape)). C order fills the
     LAST axis fastest, so successive LFSR words first walk across H
     (hidden units), then B (batch), then S (samples) slowest. Any
     temporal/period structure intrinsic to the LFSR is therefore
     imposed on the hidden-unit axis first, then batch, and least of all
     on the sample axis -- relevant when interpreting whether LFSR
     degradation looks more like "bias/precision-at-a-neuron" (H2-like)
     or "shared-draws-across-samples" (H3-like) defects.

Does NOT modify models/net.py, models/pdnn.py, or pneuron/rng.py -- uses
their public APIs only. Does NOT recompute the ideal-noise curves in
results/sanity_check.json; those are read back in for reference/printing.

Run: .venv/bin/python experiments/e2_precision.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.net import MLP
from models.pdnn import NoiseConfig, evaluate
from pneuron.rng import GaloisLFSR

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
BASELINE_JSON = ROOT / "results" / "baseline_acc.json"
SANITY_JSON = ROOT / "results" / "sanity_check.json"
OUT_JSON = ROOT / "results" / "e2_precision.json"

WIDTHS = ("large", "mid", "small")
TRAIN_SEEDS = (0, 1, 2)

K_VALUES = (1, 2, 3, 4, 5, 6, 8, 12, 16)
S_VALUES = (1, 2, 4, 8, 16, 32)
PRECISION_NOISE_SEEDS = (2000, 2001, 2002, 2003, 2004)  # 5 seeds/cell

BIAS_VALUES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
BIAS_S_VALUES = (1, 32)
BIAS_NOISE_SEEDS = (2000, 2001, 2002, 2003, 2004)  # 5 seeds/cell

LFSR_NBITS = (8, 16)
LFSR_S_VALUES = (1, 2, 4, 8, 16, 32)
LFSR_SEEDS = (7, 77, 777)  # 3 distinct initial LFSR register states
LFSR_WIDTH = "mid"

# Runtime governor: if a quick calibration projects total wall time over
# this many seconds, drop precision-grid noise seeds to 3 for S<=4 cells
# (per HANDOFF instructions) rather than shrinking any grid.
RUNTIME_BUDGET_SECONDS = 40 * 60
REDUCED_NOISE_SEEDS = (2000, 2001, 2002)


def load_test_set():
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    X = test_ds.data.float().div(255.0).view(len(test_ds), -1)
    y = test_ds.targets.clone()
    return X, y


def load_model(tag, seed):
    ckpt = torch.load(MODELS_DIR / f"{tag}_s{seed}.pt", map_location="cpu")
    model = MLP(hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def deterministic_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def mean_std(accs):
    return float(np.mean(accs)), float(np.std(accs))


def check_monotonicity(curve_by_x, x_values, label, tol_abs=0.003):
    """Flag decreases along x_values (ascending) bigger than the pooled
    noise-seed std between the two points (and bigger than tol_abs, to
    ignore sub-percent noise-floor wiggle). curve_by_x: {x: (mean, std)}.
    Returns a list of anomaly dicts.
    """
    anomalies = []
    xs = sorted(x_values)
    for a, b in zip(xs[:-1], xs[1:]):
        m0, s0 = curve_by_x[a]
        m1, s1 = curve_by_x[b]
        drop = m0 - m1
        thresh = max(tol_abs, 2.0 * (s0 + s1))
        if drop > thresh:
            anomalies.append({
                "label": label, "x0": a, "x1": b,
                "mean0": m0, "mean1": m1, "drop": drop, "threshold": thresh,
            })
    return anomalies


def main():
    torch.set_num_threads(8)
    t0 = time.time()

    X, y = load_test_set()
    print(f"Loaded MNIST test set: X={tuple(X.shape)} y={tuple(y.shape)}")

    with open(BASELINE_JSON) as f:
        baseline = json.load(f)
    with open(SANITY_JSON) as f:
        sanity = json.load(f)

    # ---- load all 9 checkpoints once ----
    models = {}
    det_accs = {}
    for tag in WIDTHS:
        for seed in TRAIN_SEEDS:
            model, ckpt = load_model(tag, seed)
            models[(tag, seed)] = model
            det_accs[(tag, seed)] = deterministic_accuracy(model, X, y)
            print(f"  loaded {tag}_s{seed}: hidden={ckpt['hidden']} "
                  f"det_acc={det_accs[(tag, seed)]:.4f}")

    reduced_used = False  # flips true if the runtime governor kicks in

    # =====================================================================
    # 1. PRECISION GRID
    # =====================================================================
    print("\n--- precision grid ---")
    precision = {}
    anomalies = []
    tp0 = time.time()
    n_calls = 0

    for tag in WIDTHS:
        width_entry = {}
        agg_means = {S: {k: [] for k in K_VALUES} for S in S_VALUES}  # per-seed means to aggregate

        for seed in TRAIN_SEEDS:
            model = models[(tag, seed)]
            by_k = {}
            for k in K_VALUES:
                by_S = {}
                for S in S_VALUES:
                    noise_seeds = PRECISION_NOISE_SEEDS
                    if reduced_used and S <= 4:
                        noise_seeds = REDUCED_NOISE_SEEDS
                    cfg = NoiseConfig(bitdepth=k, rho=0.0)
                    accs = [evaluate(model, (X, y), S=S, cfg=cfg, seed=ns) for ns in noise_seeds]
                    n_calls += len(accs)
                    m, s = mean_std(accs)
                    by_S[str(S)] = {"mean": m, "std": s, "raw": accs,
                                     "noise_seeds_used": list(noise_seeds)}
                    agg_means[S][k].append(m)
                by_k[str(k)] = by_S
                elapsed = time.time() - tp0
                # runtime governor: check once, early (after first model's k=1 row)
                if tag == WIDTHS[0] and seed == TRAIN_SEEDS[0] and k == K_VALUES[0] and not reduced_used:
                    projected_total = elapsed * (len(WIDTHS) * len(TRAIN_SEEDS) * len(K_VALUES))
                    print(f"  [calibration] one (tag,seed) k-row took {elapsed:.1f}s, "
                          f"projected precision-grid total ~{projected_total:.0f}s")
                    if projected_total > RUNTIME_BUDGET_SECONDS:
                        reduced_used = True
                        print("  [calibration] projected time exceeds budget -> "
                              "dropping to 3 noise seeds for S<=4 cells")
            width_entry[f"s{seed}"] = {"det_acc": det_accs[(tag, seed)], "by_k": by_k}
            print(f"  {tag} s{seed} precision grid done ({time.time()-tp0:.1f}s elapsed)")

        # aggregate over training seeds
        agg_by_k = {}
        for k in K_VALUES:
            agg_by_S = {}
            for S in S_VALUES:
                vals = agg_means[S][k]
                m, s = mean_std(vals)
                agg_by_S[str(S)] = {"mean": m, "std": s, "per_seed_means": vals}
            agg_by_k[str(k)] = agg_by_S
        width_entry["_aggregate"] = {
            "det_acc_mean": float(np.mean([det_accs[(tag, s)] for s in TRAIN_SEEDS])),
            "by_k": agg_by_k,
        }

        # anomaly checks on the aggregate curves
        for S in S_VALUES:
            curve = {k: (agg_by_k[str(k)][str(S)]["mean"], agg_by_k[str(k)][str(S)]["std"])
                     for k in K_VALUES}
            anomalies += check_monotonicity(curve, K_VALUES, f"precision:{tag}:S={S}:k-axis")
        for k in K_VALUES:
            curve = {S: (agg_by_k[str(k)][str(S)]["mean"], agg_by_k[str(k)][str(S)]["std"])
                     for S in S_VALUES}
            anomalies += check_monotonicity(curve, S_VALUES, f"precision:{tag}:k={k}:S-axis")

        precision[tag] = width_entry

    precision_seconds = time.time() - tp0
    print(f"Precision grid: {n_calls} evaluate() calls in {precision_seconds:.1f}s")

    # =====================================================================
    # 2. BIAS GRID
    # =====================================================================
    print("\n--- bias grid ---")
    bias_results = {}
    tb0 = time.time()
    n_calls_bias = 0

    for tag in WIDTHS:
        width_entry = {}
        agg_means = {S: {b: [] for b in BIAS_VALUES} for S in BIAS_S_VALUES}

        for seed in TRAIN_SEEDS:
            model = models[(tag, seed)]
            by_b = {}
            for b in BIAS_VALUES:
                by_S = {}
                for S in BIAS_S_VALUES:
                    cfg = NoiseConfig(bitdepth=None, bias=b, rho=0.0)
                    accs = [evaluate(model, (X, y), S=S, cfg=cfg, seed=ns) for ns in BIAS_NOISE_SEEDS]
                    n_calls_bias += len(accs)
                    m, s = mean_std(accs)
                    by_S[str(S)] = {"mean": m, "std": s, "raw": accs,
                                     "noise_seeds_used": list(BIAS_NOISE_SEEDS)}
                    agg_means[S][b].append(m)
                by_b[str(b)] = by_S
            width_entry[f"s{seed}"] = {"det_acc": det_accs[(tag, seed)], "by_b": by_b}

        agg_by_b = {}
        for b in BIAS_VALUES:
            agg_by_S = {}
            for S in BIAS_S_VALUES:
                vals = agg_means[S][b]
                m, s = mean_std(vals)
                agg_by_S[str(S)] = {"mean": m, "std": s, "per_seed_means": vals}
            agg_by_b[str(b)] = agg_by_S
        width_entry["_aggregate"] = {
            "det_acc_mean": float(np.mean([det_accs[(tag, s)] for s in TRAIN_SEEDS])),
            "by_b": agg_by_b,
        }

        for S in BIAS_S_VALUES:
            curve = {b: (agg_by_b[str(b)][str(S)]["mean"], agg_by_b[str(b)][str(S)]["std"])
                     for b in BIAS_VALUES}
            anomalies += check_monotonicity(curve, BIAS_VALUES, f"bias:{tag}:S={S}:b-axis")

        bias_results[tag] = width_entry

    bias_seconds = time.time() - tb0
    print(f"Bias grid: {n_calls_bias} evaluate() calls in {bias_seconds:.1f}s")

    # =====================================================================
    # 3. LFSR CONDITIONS (mid width only)
    # =====================================================================
    print("\n--- LFSR conditions (mid width) ---")
    lfsr_results = {}
    tl0 = time.time()
    n_calls_lfsr = 0

    width_entry = {}
    agg_means = {nbits: {S: [] for S in LFSR_S_VALUES} for nbits in LFSR_NBITS}
    for seed in TRAIN_SEEDS:
        model = models[(LFSR_WIDTH, seed)]
        by_nbits = {}
        for nbits in LFSR_NBITS:
            by_S = {}
            for S in LFSR_S_VALUES:
                accs = []
                for lseed in LFSR_SEEDS:
                    lfsr = GaloisLFSR(nbits=nbits, seed=lseed)  # fresh instance per call
                    cfg = NoiseConfig(uniform_source=lfsr.next_uniform)
                    acc = evaluate(model, (X, y), S=S, cfg=cfg, seed=0)  # seed unused (uniform_source overrides)
                    accs.append(acc)
                n_calls_lfsr += len(accs)
                m, s = mean_std(accs)
                by_S[str(S)] = {"mean": m, "std": s, "raw": accs, "lfsr_seeds_used": list(LFSR_SEEDS)}
                agg_means[nbits][S].append(m)
            by_nbits[f"nbits{nbits}"] = by_S
        width_entry[f"s{seed}"] = {"det_acc": det_accs[(LFSR_WIDTH, seed)], "by_nbits": by_nbits}
        print(f"  mid s{seed} LFSR conditions done ({time.time()-tl0:.1f}s elapsed)")

    agg_by_nbits = {}
    for nbits in LFSR_NBITS:
        agg_by_S = {}
        for S in LFSR_S_VALUES:
            vals = agg_means[nbits][S]
            m, s = mean_std(vals)
            agg_by_S[str(S)] = {"mean": m, "std": s, "per_seed_means": vals}
        agg_by_nbits[f"nbits{nbits}"] = agg_by_S
    width_entry["_aggregate"] = {
        "det_acc_mean": float(np.mean([det_accs[(LFSR_WIDTH, s)] for s in TRAIN_SEEDS])),
        "by_nbits": agg_by_nbits,
    }

    for nbits in LFSR_NBITS:
        curve = {S: (agg_by_nbits[f"nbits{nbits}"][str(S)]["mean"],
                     agg_by_nbits[f"nbits{nbits}"][str(S)]["std"]) for S in LFSR_S_VALUES}
        anomalies += check_monotonicity(curve, LFSR_S_VALUES, f"lfsr:nbits{nbits}:S-axis")

    lfsr_results[LFSR_WIDTH] = width_entry
    lfsr_seconds = time.time() - tl0
    print(f"LFSR conditions: {n_calls_lfsr} evaluate() calls in {lfsr_seconds:.1f}s")

    # =====================================================================
    # Assemble + write output
    # =====================================================================
    total_seconds = time.time() - t0

    baseline_ref = {k: v for k, v in baseline.items() if isinstance(v, dict)}

    results = {
        "_meta": {
            "description": "E2 (H2): p-DNN accuracy vs RNG precision (bitdepth), bias, "
                            "and a realistic LFSR generator. Full 10k MNIST test set, "
                            "frozen weights, logits averaged over S stochastic passes.",
            "widths": WIDTHS,
            "train_seeds": TRAIN_SEEDS,
            "n_test": int(X.shape[0]),
            "precision_grid": {
                "k_values": K_VALUES, "S_values": S_VALUES,
                "noise_seeds": PRECISION_NOISE_SEEDS,
                "reduced_noise_seeds_used": reduced_used,
                "reduced_noise_seeds": REDUCED_NOISE_SEEDS if reduced_used else None,
                "reduced_applies_to": "S<=4 cells only" if reduced_used else None,
                "cfg_notes": "bitdepth=k, rho=0.0 (strictly independent draws per sample, "
                             "fresh every forward pass -- not the H3 correlation condition)",
                "n_calls": n_calls,
                "seconds": precision_seconds,
            },
            "bias_grid": {
                "bias_values": BIAS_VALUES, "S_values": BIAS_S_VALUES,
                "noise_seeds": BIAS_NOISE_SEEDS,
                "cfg_notes": "bitdepth=None (unquantized), bias=b, rho=0.0",
                "n_calls": n_calls_bias,
                "seconds": bias_seconds,
            },
            "lfsr_grid": {
                "nbits": LFSR_NBITS, "S_values": LFSR_S_VALUES,
                "width": LFSR_WIDTH, "train_seeds": TRAIN_SEEDS,
                "lfsr_seeds": LFSR_SEEDS,
                "lfsr_seeds_notes": "distinct initial GaloisLFSR register states; a fresh "
                                     "GaloisLFSR instance is constructed per evaluate() call "
                                     "(no state leakage across cells). These play the role of "
                                     "'noise seeds' for this condition -- there are no separate "
                                     "independent noise seeds layered on top.",
                "axis_convention": "make_noise() draws a flat uniform_source(n) buffer and "
                                    "reshapes to (S,B,H) in numpy's default C order (see "
                                    "models/pdnn.py _draw_uniform). C order fills the LAST "
                                    "axis fastest: the LFSR's serial output walks across H "
                                    "(hidden units) first, then B (batch), then S (samples) "
                                    "slowest. So the LFSR's temporal/period structure lands "
                                    "predominantly on the hidden-unit axis, only a little on "
                                    "the sample axis -- a within-forward-pass defect more than "
                                    "an across-sample one.",
                "n_calls": n_calls_lfsr,
                "seconds": lfsr_seconds,
            },
            "total_seconds": total_seconds,
        },
        "baseline_acc_reference": baseline_ref,
        "sanity_check_ideal_reference_note": "ideal-noise (good RNG, k=None/bias=0/rho=0) "
                                              "accuracy-vs-S curves are NOT recomputed here; "
                                              "see results/sanity_check.json for that reference.",
        "precision": precision,
        "bias": bias_results,
        "lfsr": lfsr_results,
        "anomalies": anomalies,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # =====================================================================
    # Compact summary
    # =====================================================================
    print("\n" + "=" * 78)
    print("MID WIDTH -- k vs accuracy (aggregate over 3 training seeds)")
    print(f"{'k':>4}{'S=1':>10}{'S=32':>10}")
    mid_agg = precision["mid"]["_aggregate"]["by_k"]
    for k in K_VALUES:
        s1 = mid_agg[str(k)]["1"]["mean"]
        s32 = mid_agg[str(k)]["32"]["mean"]
        print(f"{k:>4}{s1*100:>9.2f}%{s32*100:>9.2f}%")

    print("\nMID WIDTH -- bias vs accuracy at S=32")
    print(f"{'b':>6}{'acc':>10}")
    mid_bias_agg = bias_results["mid"]["_aggregate"]["by_b"]
    for b in BIAS_VALUES:
        m = mid_bias_agg[str(b)]["32"]["mean"]
        print(f"{b:>6}{m*100:>9.2f}%")

    print("\nMID WIDTH -- LFSR vs ideal at S=32")
    ideal_mid_s32 = sanity["mid"]["_aggregate"]["pdnn_acc_by_S_mean"]["32"]
    print(f"ideal (sanity_check) S=32: {ideal_mid_s32*100:.2f}%")
    for nbits in LFSR_NBITS:
        m = agg_by_nbits[f"nbits{nbits}"]["32"]["mean"]
        print(f"lfsr nbits={nbits:<3} S=32: {m*100:.2f}%  delta={100*(m-ideal_mid_s32):+.2f}pp")

    print("\nAnomalies flagged (drop > max(0.3pp, 2*pooled std)):", len(anomalies))
    for a in anomalies[:20]:
        print(f"  {a['label']}: {a['x0']}->{a['x1']}  {a['mean0']*100:.2f}%->{a['mean1']*100:.2f}% "
              f"(drop={a['drop']*100:.2f}pp, thresh={a['threshold']*100:.2f}pp)")

    print("=" * 78)
    print(f"Total runtime: {total_seconds:.1f}s")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
