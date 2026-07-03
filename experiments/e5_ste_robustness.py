"""E5 (H5): does route-2 STE (sample-aware) training buy ROBUSTNESS to
DEFECTIVE inference noise, relative to the frozen route-1 baseline?

Both net types are plain models.net.MLP weight sets scored by the IDENTICAL
models.pdnn.evaluate protocol -- the only difference is how the weights were
obtained (route 1: deterministic tanh backprop, frozen; route 2: STE p-bit
training with ideal noise). So any robustness difference is attributable to
the training route, not the eval harness.

Nets evaluated (MID (64,32) only):
  frozen : mid_s0, mid_s1                         (route-1 baselines)
  ste_S1 : ste_S1_s0, ste_S1_s1                   (route-2, 1 stoch pass/step)
  ste_S4 : ste_S4_s0, ste_S4_s1                   (route-2, 4 stoch passes/step)

Inference conditions (full 10k test, 5 noise seeds each):
  ideal, k=1, k=2, bias=0.2, rho=0.9, rho=1.0
Each at S in {1, 4, 16, 64}.

Honesty framing (H5 is genuinely open): [2] trains with IDEAL noise, so there
is no strong prior that STE helps under DEFECTIVE noise. 'No benefit' or 'STE
worse' are valid outcomes. We report:
  - absolute acc per net-type/condition/S with combined std (pooled over the
    2 training seeds x 5 noise seeds = 10 raw accs),
  - delta = STE_variant - frozen, with a 2x-combined-std 'established' gate,
  - baseline-relative degradation = acc(ideal,S) - acc(cond,S) per net-type,
    so robustness is compared FROM each net's own ideal-noise accuracy (the
    STE and frozen nets need not share a clean-accuracy baseline).

Run: .venv/bin/python experiments/e5_ste_robustness.py
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
OUT_JSON = ROOT / "results" / "e5_ste.json"

# net-type -> list of (checkpoint tag) for its 2 training seeds
NET_TYPES = {
    "frozen": ["mid_s0", "mid_s1"],
    "ste_S1": ["ste_S1_s0", "ste_S1_s1"],
    "ste_S4": ["ste_S4_s0", "ste_S4_s1"],
}

# condition name -> NoiseConfig kwargs
CONDITIONS = {
    "ideal":    dict(),
    "k1":       dict(bitdepth=1),
    "k2":       dict(bitdepth=2),
    "bias0.2":  dict(bias=0.2),
    "rho0.9":   dict(rho=0.9),
    "rho1.0":   dict(rho=1.0),
}
COND_ORDER = ["ideal", "k1", "k2", "bias0.2", "rho0.9", "rho1.0"]

S_VALUES = (1, 4, 16, 64)
NOISE_SEEDS = (2000, 2001, 2002, 2003, 2004)


def load_test_set():
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    X = test_ds.data.float().div(255.0).view(len(test_ds), -1)
    y = test_ds.targets.clone()
    return X, y


def load_model(tag):
    ckpt = torch.load(MODELS_DIR / f"{tag}.pt", map_location="cpu")
    model = MLP(hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def det_tanh_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def main():
    torch.set_num_threads(8)
    t0 = time.time()

    X, y = load_test_set()
    print(f"MNIST test: X={tuple(X.shape)}")

    # ---- load all nets, record deterministic-tanh accuracy ----
    models = {}      # tag -> model
    det_acc = {}     # tag -> det tanh acc
    ckpt_meta = {}   # tag -> selected metadata
    for net_type, tags in NET_TYPES.items():
        for tag in tags:
            model, ckpt = load_model(tag)
            models[tag] = model
            det_acc[tag] = det_tanh_accuracy(model, X, y)
            ckpt_meta[tag] = {
                "hidden": list(ckpt["hidden"]), "seed": ckpt.get("seed"),
                "route": ckpt.get("route", "frozen"),
                "S_train": ckpt.get("S_train"),
                "det_tanh_acc": det_acc[tag],
                "stored_test_acc": ckpt.get("test_acc"),
            }
            print(f"  {tag}: route={ckpt.get('route','frozen')} "
                  f"S_train={ckpt.get('S_train')} det_tanh_acc={det_acc[tag]:.4f}")

    # =====================================================================
    # main grid: net-type x condition x S, pooled over 2 train seeds x 5 noise seeds
    # =====================================================================
    # raw[net_type][cond][S] = list of 10 accs (2 tags x 5 noise seeds)
    raw = {nt: {c: {S: [] for S in S_VALUES} for c in COND_ORDER} for nt in NET_TYPES}
    # also keep per-tag breakdown for transparency
    per_tag = {}

    n_calls = 0
    for net_type, tags in NET_TYPES.items():
        for tag in tags:
            model = models[tag]
            per_tag.setdefault(net_type, {})[tag] = {}
            for cond in COND_ORDER:
                cfg = NoiseConfig(**CONDITIONS[cond])
                per_tag[net_type][tag][cond] = {}
                for S in S_VALUES:
                    accs = [evaluate(model, (X, y), S=S, cfg=cfg, seed=ns)
                            for ns in NOISE_SEEDS]
                    n_calls += len(accs)
                    raw[net_type][cond][S].extend(accs)
                    m, s = mean_std(accs)
                    per_tag[net_type][tag][cond][str(S)] = {
                        "mean": m, "std": s, "raw": accs,
                    }
            print(f"  scored {tag} ({time.time()-t0:.1f}s elapsed)")

    # aggregate per net-type: pooled mean/std over the 10 raw accs
    agg = {nt: {c: {} for c in COND_ORDER} for nt in NET_TYPES}
    for nt in NET_TYPES:
        for c in COND_ORDER:
            for S in S_VALUES:
                m, s = mean_std(raw[nt][c][S])
                agg[nt][c][str(S)] = {"mean": m, "std": s, "n": len(raw[nt][c][S])}

    # baseline-relative degradation: acc(ideal,S) - acc(cond,S) per net-type
    rel_degr = {nt: {c: {} for c in COND_ORDER} for nt in NET_TYPES}
    for nt in NET_TYPES:
        for c in COND_ORDER:
            for S in S_VALUES:
                ideal_m = agg[nt]["ideal"][str(S)]["mean"]
                cond_m = agg[nt][c][str(S)]["mean"]
                rel_degr[nt][c][str(S)] = ideal_m - cond_m

    # deltas: (ste_S1 - frozen) and (ste_S4 - frozen), absolute and relative-degradation
    deltas = {}
    for ste_type in ("ste_S1", "ste_S4"):
        d = {c: {} for c in COND_ORDER}
        for c in COND_ORDER:
            for S in S_VALUES:
                Ss = str(S)
                fm = agg["frozen"][c][Ss]["mean"]
                fs = agg["frozen"][c][Ss]["std"]
                sm = agg[ste_type][c][Ss]["mean"]
                ss = agg[ste_type][c][Ss]["std"]
                abs_delta = sm - fm
                combined = (fs ** 2 + ss ** 2) ** 0.5
                established = abs(abs_delta) > 2.0 * combined
                # relative-degradation delta (STE degrades less if this is negative)
                rel_delta = rel_degr[ste_type][c][Ss] - rel_degr["frozen"][c][Ss]
                d[c][Ss] = {
                    "abs_delta": abs_delta,
                    "combined_std": combined,
                    "threshold_2sigma": 2.0 * combined,
                    "established": bool(established),
                    "frozen_mean": fm, "ste_mean": sm,
                    "rel_degr_frozen": rel_degr["frozen"][c][Ss],
                    "rel_degr_ste": rel_degr[ste_type][c][Ss],
                    "rel_degr_delta": rel_delta,
                }
        deltas[ste_type] = d

    total_seconds = time.time() - t0

    out = {
        "_meta": {
            "description": "E5 (H5): route-2 STE training vs route-1 frozen baseline, "
                           "robustness to defective inference noise. Identical "
                           "models.pdnn.evaluate protocol for both net types. MID (64,32).",
            "net_types": {k: v for k, v in NET_TYPES.items()},
            "conditions": {c: CONDITIONS[c] for c in COND_ORDER},
            "condition_order": COND_ORDER,
            "S_values": list(S_VALUES),
            "noise_seeds": list(NOISE_SEEDS),
            "n_test": int(X.shape[0]),
            "aggregation": "per net-type: pooled over 2 training seeds x 5 noise seeds "
                           "= 10 raw accs; mean and std are of those 10.",
            "delta_convention": "abs_delta = ste_mean - frozen_mean (positive => STE higher "
                                "accuracy). 'established' = |abs_delta| > 2*sqrt(std_frozen^2 "
                                "+ std_ste^2). rel_degr = acc(ideal,S)-acc(cond,S) within a "
                                "net-type; rel_degr_delta = rel_degr_ste - rel_degr_frozen "
                                "(negative => STE degrades LESS from its own baseline).",
            "det_tanh_acc_note": "deterministic tanh-mode accuracy per net (no p-bit noise), "
                                 "for reference; STE and frozen nets need not share it.",
            "n_calls": n_calls,
            "total_seconds": total_seconds,
        },
        "det_tanh_acc": det_acc,
        "ckpt_meta": ckpt_meta,
        "aggregate": agg,
        "rel_degradation": rel_degr,
        "deltas": deltas,
        "per_tag": per_tag,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # =====================================================================
    # compact table: condition x S -> frozen | ste_S1 | ste_S4 (means)
    # =====================================================================
    print("\n" + "=" * 78)
    print("E5 accuracy (%) by condition x S:  frozen | ste_S1 | ste_S4  (net-type means)")
    print(f"{'cond':>9}{'S':>4}{'frozen':>10}{'ste_S1':>10}{'ste_S4':>10}"
          f"{'d(S1)':>9}{'d(S4)':>9}")
    for c in COND_ORDER:
        for S in S_VALUES:
            Ss = str(S)
            fm = agg["frozen"][c][Ss]["mean"] * 100
            s1 = agg["ste_S1"][c][Ss]["mean"] * 100
            s4 = agg["ste_S4"][c][Ss]["mean"] * 100
            d1 = deltas["ste_S1"][c][Ss]
            d4 = deltas["ste_S4"][c][Ss]
            m1 = "*" if d1["established"] else " "
            m4 = "*" if d4["established"] else " "
            print(f"{c:>9}{S:>4}{fm:>10.2f}{s1:>10.2f}{s4:>10.2f}"
                  f"{d1['abs_delta']*100:>+8.2f}{m1}{d4['abs_delta']*100:>+8.2f}{m4}")
    print("  ('*' = |delta| exceeds 2x combined std; else not established)")

    print("\nDeterministic tanh-mode accuracy per net:")
    for nt, tags in NET_TYPES.items():
        for tag in tags:
            print(f"  {tag:>12}: {det_acc[tag]*100:.2f}%")

    print("=" * 78)
    print(f"{n_calls} evaluate() calls in {total_seconds:.1f}s")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
