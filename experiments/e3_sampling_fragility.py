"""E3 sampling fragility (H3, the headline): does the multi-sample p-DNN
accuracy gain require INDEPENDENT draws across the S inference samples?

Uses models/pdnn.py's NoiseConfig.rho (AR(1)/Gaussian-copula correlation
along the sample axis, marginal stays Uniform[-1,1]) and
shared_across_neurons (iid across samples, shared across neurons within a
sample -- a different correlation axis). Frozen weights (results/models/*.pt)
throughout; only the injected randomness changes.

Sections (see HANDOFF.md section 4, E3 row):
  1. MAIN GRID: mid width (largest sampling gain, 13.7pts in sanity_check.json)
     x 3 training seeds x rho in {0,.25,.5,.75,.9,.95,.99,1} x S in
     {1,2,4,8,16,32,64}, full precision.
  2. WIDTH CHECK: large & small width x 3 training seeds x reduced grid
     rho in {0,.9,1} x S in {1,4,16,64}.
  3. PRECISION CONTRAST: mid width, k=2 (bitdepth) x rho in {0,.9} x
     S in {1,2,4,8,16,32,64} -- the coarse-but-independent (k2,rho0) curve
     vs the fine-but-correlated (rho0.9, k=None, pulled from the main grid)
     curve; also k2+rho0.9 (coarse AND correlated) for the 2x2 story.
  4. CROSS-NEURON AXIS: mid width, shared_across_neurons=True (iid across
     samples, shared across neurons) x S in {1,2,4,8,16,32,64}.
  5. ECE: mid width, training seed 0 only, S in {1,4,16,64}, conditions
     ideal / k2-rho0 / rho0.9 / rho1.0, 15-bin standard ECE on the
     sample-averaged softmax (evaluate(..., return_probs=True)).

All full 10k MNIST test set, 5 noise seeds per cell, frozen weights,
averaged logits. KEY CHECKS: rho=0 must reproduce results/sanity_check.json
mid curves within noise-seed std; rho=1 at any S must ~= S=1 accuracy
(same noise-seed draws collapse to a single shared sample).

Run: .venv/bin/python experiments/e3_sampling_fragility.py
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

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
SANITY_JSON = ROOT / "results" / "sanity_check.json"
OUT_JSON = ROOT / "results" / "e3_sampling.json"
OUT_ECE_JSON = ROOT / "results" / "e3_ece.json"

TRAIN_SEEDS = (0, 1, 2)
NOISE_SEEDS = (2000, 2001, 2002, 2003, 2004)  # 5 noise seeds/cell, distinct
                                               # from sanity_check.py's 1000s

MAIN_WIDTH = "mid"
MAIN_RHOS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
MAIN_S = (1, 2, 4, 8, 16, 32, 64)

WIDTH_CHECK_WIDTHS = ("large", "small")
WIDTH_CHECK_RHOS = (0.0, 0.9, 1.0)
WIDTH_CHECK_S = (1, 4, 16, 64)

PRECISION_K = 2
PRECISION_S = (1, 2, 4, 8, 16, 32, 64)
PRECISION_RHOS = (0.0, 0.9)  # k2-rho0 (coarse, independent), k2-rho0.9 (coarse, correlated)

CROSS_NEURON_S = (1, 2, 4, 8, 16, 32, 64)

ECE_TRAIN_SEED = 0
ECE_S = (1, 4, 16, 64)
ECE_BINS = 15
# (label, NoiseConfig kwargs)
ECE_CONDITIONS = (
    ("ideal", dict(rho=0.0)),
    ("k2_rho0", dict(bitdepth=2, rho=0.0)),
    ("rho0.9", dict(rho=0.9)),
    ("rho1.0", dict(rho=1.0)),
)

SANITY_S_OVERLAP = (1, 2, 4, 8, 16, 32)  # S values present in both grids


# ---------------------------------------------------------------------------
# data / models
# ---------------------------------------------------------------------------

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
    return model


# ---------------------------------------------------------------------------
# cell runners
# ---------------------------------------------------------------------------

def run_cell(model, X, y, S, cfg, noise_seeds=NOISE_SEEDS):
    accs = [evaluate(model, (X, y), S=S, cfg=cfg, seed=ns) for ns in noise_seeds]
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "raw": accs,
        "noise_seeds_used": list(noise_seeds),
    }


def compute_ece(probs, y, n_bins=ECE_BINS):
    """Standard equal-width-bin ECE on sample-averaged softmax probs."""
    y = torch.as_tensor(y)
    confidences, preds = probs.max(dim=1)
    correct = (preds == y).float()
    N = probs.shape[0]
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_b = int(mask.sum().item())
        if n_b == 0:
            continue
        acc_b = correct[mask].mean().item()
        conf_b = confidences[mask].mean().item()
        ece += (n_b / N) * abs(acc_b - conf_b)
    return float(ece)


def run_ece_cell(model, X, y, S, cfg, noise_seeds=NOISE_SEEDS):
    eces = []
    for ns in noise_seeds:
        acc, probs = evaluate(model, (X, y), S=S, cfg=cfg, seed=ns, return_probs=True)
        eces.append(compute_ece(probs, y))
    return {
        "mean": float(np.mean(eces)),
        "std": float(np.std(eces)),
        "raw": eces,
        "noise_seeds_used": list(noise_seeds),
    }


def aggregate_over_seeds(per_seed_means):
    """per_seed_means: list of floats (one per training seed) -> mean/std."""
    return {"mean": float(np.mean(per_seed_means)), "std": float(np.std(per_seed_means))}


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def run_main_grid(X, y):
    width_entry = {}
    agg = {f"{rho}": {f"{S}": [] for S in MAIN_S} for rho in MAIN_RHOS}
    for seed in TRAIN_SEEDS:
        model = load_model(MAIN_WIDTH, seed)
        seed_entry = {}
        for rho in MAIN_RHOS:
            cfg = NoiseConfig(rho=rho)
            rho_entry = {}
            for S in MAIN_S:
                cell = run_cell(model, X, y, S, cfg)
                rho_entry[str(S)] = cell
                agg[str(rho)][str(S)].append(cell["mean"])
            seed_entry[str(rho)] = rho_entry
        width_entry[f"s{seed}"] = seed_entry
        print(f"  main_grid: {MAIN_WIDTH} s{seed} done")

    aggregate = {
        rho_key: {S_key: aggregate_over_seeds(vals) for S_key, vals in S_dict.items()}
        for rho_key, S_dict in agg.items()
    }
    width_entry["_aggregate"] = aggregate
    return width_entry


def run_width_check(X, y):
    out = {}
    for tag in WIDTH_CHECK_WIDTHS:
        width_entry = {}
        agg = {f"{rho}": {f"{S}": [] for S in WIDTH_CHECK_S} for rho in WIDTH_CHECK_RHOS}
        for seed in TRAIN_SEEDS:
            model = load_model(tag, seed)
            seed_entry = {}
            for rho in WIDTH_CHECK_RHOS:
                cfg = NoiseConfig(rho=rho)
                rho_entry = {}
                for S in WIDTH_CHECK_S:
                    cell = run_cell(model, X, y, S, cfg)
                    rho_entry[str(S)] = cell
                    agg[str(rho)][str(S)].append(cell["mean"])
                seed_entry[str(rho)] = rho_entry
            width_entry[f"s{seed}"] = seed_entry
            print(f"  width_check: {tag} s{seed} done")
        width_entry["_aggregate"] = {
            rho_key: {S_key: aggregate_over_seeds(vals) for S_key, vals in S_dict.items()}
            for rho_key, S_dict in agg.items()
        }
        out[tag] = width_entry
    return out


def run_precision_contrast(X, y, main_grid_mid_aggregate):
    out = {}
    for rho in PRECISION_RHOS:
        label = f"k{PRECISION_K}_rho{rho}"
        agg = {f"{S}": [] for S in PRECISION_S}
        seed_entries = {}
        for seed in TRAIN_SEEDS:
            model = load_model(MAIN_WIDTH, seed)
            cfg = NoiseConfig(bitdepth=PRECISION_K, rho=rho)
            S_entry = {}
            for S in PRECISION_S:
                cell = run_cell(model, X, y, S, cfg)
                S_entry[str(S)] = cell
                agg[str(S)].append(cell["mean"])
            seed_entries[f"s{seed}"] = S_entry
            print(f"  precision_contrast: {label} s{seed} done")
        seed_entries["_aggregate"] = {
            S_key: aggregate_over_seeds(vals) for S_key, vals in agg.items()
        }
        out[label] = seed_entries

    # references pulled from the already-computed main grid (rho=0 = fine
    # independent baseline; rho=0.9 k=None = fine-but-correlated headline
    # contrast partner) -- not recomputed.
    out["reference_kNone_rho0"] = {
        S_key: main_grid_mid_aggregate["0.0"][S_key] for S_key in main_grid_mid_aggregate["0.0"]
    }
    out["reference_kNone_rho0.9"] = {
        S_key: main_grid_mid_aggregate["0.9"][S_key] for S_key in main_grid_mid_aggregate["0.9"]
    }
    return out


def run_cross_neuron(X, y):
    agg = {f"{S}": [] for S in CROSS_NEURON_S}
    seed_entries = {}
    for seed in TRAIN_SEEDS:
        model = load_model(MAIN_WIDTH, seed)
        cfg = NoiseConfig(shared_across_neurons=True)
        S_entry = {}
        for S in CROSS_NEURON_S:
            cell = run_cell(model, X, y, S, cfg)
            S_entry[str(S)] = cell
            agg[str(S)].append(cell["mean"])
        seed_entries[f"s{seed}"] = S_entry
        print(f"  cross_neuron: s{seed} done")
    seed_entries["_aggregate"] = {
        S_key: aggregate_over_seeds(vals) for S_key, vals in agg.items()
    }
    return {MAIN_WIDTH: seed_entries}


def run_ece(X, y):
    model = load_model(MAIN_WIDTH, ECE_TRAIN_SEED)
    out = {}
    for label, cfg_kwargs in ECE_CONDITIONS:
        cfg = NoiseConfig(**cfg_kwargs)
        S_entry = {}
        for S in ECE_S:
            S_entry[str(S)] = run_ece_cell(model, X, y, S, cfg)
        out[label] = S_entry
        print(f"  ece: {label} done")
    return out


# ---------------------------------------------------------------------------
# key checks
# ---------------------------------------------------------------------------

def key_check_rho0_vs_sanity(main_grid_mid_aggregate):
    with open(SANITY_JSON) as f:
        sanity = json.load(f)
    sanity_mid = sanity["mid"]["_aggregate"]["pdnn_acc_by_S_mean"]
    sanity_mid_std = sanity["mid"]["_aggregate"]["pdnn_acc_by_S_std"]

    ours = main_grid_mid_aggregate["0.0"]
    rows = {}
    all_ok = True
    for S in SANITY_S_OVERLAP:
        S_key = str(S)
        our_mean, our_std = ours[S_key]["mean"], ours[S_key]["std"]
        san_mean, san_std = sanity_mid[S_key], sanity_mid_std[S_key]
        diff = abs(our_mean - san_mean)
        combined_std = our_std + san_std
        tol = max(combined_std, 1e-6) * 3  # generous: within ~3x combined noise-seed std
        ok = diff <= tol
        all_ok = all_ok and ok
        rows[S_key] = {
            "our_mean": our_mean, "our_std": our_std,
            "sanity_mean": san_mean, "sanity_std": san_std,
            "diff": diff, "tol_3x_combined_std": tol, "within_tolerance": ok,
        }
    return {"rows": rows, "all_within_tolerance": all_ok}


def key_check_rho1_flat(main_grid_mid_aggregate):
    s1 = main_grid_mid_aggregate["1.0"]["1"]["mean"]
    rows = {}
    all_ok = True
    for S in MAIN_S:
        cell = main_grid_mid_aggregate["1.0"][str(S)]
        diff = abs(cell["mean"] - s1)
        tol = max(cell["std"], 1e-6) * 3
        ok = diff <= tol
        all_ok = all_ok and ok
        rows[str(S)] = {"mean": cell["mean"], "std": cell["std"], "diff_from_S1": diff,
                         "tol_3x_std": tol, "within_tolerance": ok}
    return {"s1_value": s1, "rows": rows, "all_within_tolerance": all_ok}


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_main_grid_table(aggregate):
    print("\n" + "=" * 100)
    print(f"MAIN GRID -- {MAIN_WIDTH} width, aggregated over {len(TRAIN_SEEDS)} training seeds")
    header = f"{'rho':>6}" + "".join(f"{'S='+str(S):>10}" for S in MAIN_S)
    print(header)
    print("-" * len(header))
    for rho in MAIN_RHOS:
        row = f"{rho:>6}"
        for S in MAIN_S:
            m = aggregate[str(rho)][str(S)]["mean"]
            row += f"{m*100:>9.2f}%"
        print(row)
    print("=" * 100)


def print_precision_contrast(precision):
    print("\nPRECISION CONTRAST at S=64 (mid width):")
    k0 = precision["k2_rho0.0"]["_aggregate"]["64"]
    k9 = precision["k2_rho0.9"]["_aggregate"]["64"]
    ref0 = precision["reference_kNone_rho0"]["64"]
    ref9 = precision["reference_kNone_rho0.9"]["64"]
    print(f"  k=2,   rho=0   (coarse, independent):  {k0['mean']*100:.2f}% (std {k0['std']*100:.2f})")
    print(f"  k=None,rho=0   (fine,   independent):  {ref0['mean']*100:.2f}% (std {ref0['std']*100:.2f})")
    print(f"  k=None,rho=0.9 (fine,   correlated):   {ref9['mean']*100:.2f}% (std {ref9['std']*100:.2f})")
    print(f"  k=2,   rho=0.9 (coarse, correlated):   {k9['mean']*100:.2f}% (std {k9['std']*100:.2f})")


def print_cross_neuron(cross_neuron):
    print("\nCROSS-NEURON AXIS (mid width, shared_across_neurons=True, iid across samples):")
    agg = cross_neuron[MAIN_WIDTH]["_aggregate"]
    row = "".join(f"{'S='+str(S):>10}" for S in CROSS_NEURON_S)
    print(row)
    vals = "".join(f"{agg[str(S)]['mean']*100:>9.2f}%" for S in CROSS_NEURON_S)
    print(vals)


def print_ece_table(ece):
    print("\nECE TABLE (mid width, training seed 0, 15-bin standard ECE):")
    header = f"{'condition':>12}" + "".join(f"{'S='+str(S):>10}" for S in ECE_S)
    print(header)
    print("-" * len(header))
    for label, _ in ECE_CONDITIONS:
        row = f"{label:>12}"
        for S in ECE_S:
            m = ece[label][str(S)]["mean"]
            row += f"{m:>10.4f}"
        print(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    torch.set_num_threads(8)
    t0 = time.time()

    X, y = load_test_set()
    print(f"Loaded MNIST test set: X={tuple(X.shape)} y={tuple(y.shape)}")

    results = {"_meta": {
        "train_seeds": TRAIN_SEEDS,
        "noise_seeds": NOISE_SEEDS,
        "n_test": int(X.shape[0]),
        "main_grid": {"width": MAIN_WIDTH, "rhos": MAIN_RHOS, "S_values": MAIN_S, "bitdepth": None},
        "width_check": {"widths": WIDTH_CHECK_WIDTHS, "rhos": WIDTH_CHECK_RHOS, "S_values": WIDTH_CHECK_S},
        "precision_contrast": {"width": MAIN_WIDTH, "bitdepth": PRECISION_K,
                                "rhos": PRECISION_RHOS, "S_values": PRECISION_S},
        "cross_neuron": {"width": MAIN_WIDTH, "shared_across_neurons": True, "S_values": CROSS_NEURON_S},
        "ece": {"width": MAIN_WIDTH, "train_seed": ECE_TRAIN_SEED, "S_values": ECE_S,
                "n_bins": ECE_BINS, "conditions": [c[0] for c in ECE_CONDITIONS]},
    }}

    t = time.time()
    print("\n[1/5] main grid ...")
    main_grid = run_main_grid(X, y)
    results["main_grid"] = {MAIN_WIDTH: main_grid}
    print(f"  main grid done in {time.time()-t:.1f}s")

    t = time.time()
    print("\n[2/5] width check ...")
    width_check = run_width_check(X, y)
    results["width_check"] = width_check
    print(f"  width check done in {time.time()-t:.1f}s")

    t = time.time()
    print("\n[3/5] precision contrast ...")
    precision_contrast = run_precision_contrast(X, y, main_grid["_aggregate"])
    results["precision_contrast"] = {MAIN_WIDTH: precision_contrast}
    print(f"  precision contrast done in {time.time()-t:.1f}s")

    t = time.time()
    print("\n[4/5] cross-neuron axis ...")
    cross_neuron = run_cross_neuron(X, y)
    results["cross_neuron"] = cross_neuron
    print(f"  cross-neuron done in {time.time()-t:.1f}s")

    t = time.time()
    print("\n[5/5] ECE ...")
    ece = run_ece(X, y)
    ece_results = {"_meta": results["_meta"]["ece"], MAIN_WIDTH: {f"s{ECE_TRAIN_SEED}": ece}}
    print(f"  ECE done in {time.time()-t:.1f}s")

    # ---- key checks ----
    key_checks = {
        "rho0_vs_sanity_check": key_check_rho0_vs_sanity(main_grid["_aggregate"]),
        "rho1_flat_vs_S1": key_check_rho1_flat(main_grid["_aggregate"]),
    }
    results["key_checks"] = key_checks

    total_dt = time.time() - t0
    results["_meta"]["total_seconds"] = total_dt

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_ECE_JSON, "w") as f:
        json.dump(ece_results, f, indent=2)

    # ---- summary ----
    print_main_grid_table(main_grid["_aggregate"])
    print_precision_contrast(precision_contrast)
    print_cross_neuron(cross_neuron)
    print_ece_table(ece)

    print("\n" + "=" * 78)
    print("KEY CHECK 1: rho=0 main-grid mid curve vs results/sanity_check.json mid curve")
    for S_key, row in key_checks["rho0_vs_sanity_check"]["rows"].items():
        flag = "OK" if row["within_tolerance"] else "MISMATCH"
        print(f"  S={S_key:>3}: ours={row['our_mean']*100:.3f}% (std {row['our_std']*100:.3f}) "
              f"sanity={row['sanity_mean']*100:.3f}% (std {row['sanity_std']*100:.3f}) "
              f"diff={row['diff']*100:.3f}pt [{flag}]")
    print(f"  ALL WITHIN TOLERANCE: {key_checks['rho0_vs_sanity_check']['all_within_tolerance']}")

    print("\nKEY CHECK 2: rho=1.0 accuracy flat across S (should ~= S=1 value)")
    print(f"  S=1 value: {key_checks['rho1_flat_vs_S1']['s1_value']*100:.3f}%")
    for S_key, row in key_checks["rho1_flat_vs_S1"]["rows"].items():
        flag = "OK" if row["within_tolerance"] else "MISMATCH"
        print(f"  S={S_key:>3}: {row['mean']*100:.3f}% (std {row['std']*100:.3f}) "
              f"diff_from_S1={row['diff_from_S1']*100:.3f}pt [{flag}]")
    print(f"  ALL WITHIN TOLERANCE: {key_checks['rho1_flat_vs_S1']['all_within_tolerance']}")
    print("=" * 78)

    print(f"\nTotal runtime: {total_dt:.1f}s")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_ECE_JSON}")


if __name__ == "__main__":
    main()
