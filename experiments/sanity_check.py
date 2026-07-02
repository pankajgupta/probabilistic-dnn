"""Gate-A sanity check: reproduce the multi-sample p-DNN accuracy benefit with
IDEAL noise (no defects, independent-across-samples uniforms) on all 9 trained
baseline checkpoints, on the FULL 10k MNIST test set.

For each width (large, mid, small) x training seed (0,1,2):
  - loads results/models/{tag}_s{seed}.pt, rebuilds the MLP, recomputes the
    deterministic test accuracy on the full test set (does not trust the
    checkpoint's stored test_acc or results/baseline_acc.json blindly --
    both are compared against the freshly recomputed number).
  - evaluates the p-DNN (models.pdnn.evaluate, NoiseConfig() = ideal,
    independent Uniform[-1,1] noise, no defects) at S in {1,2,4,8,16,32},
    3 noise seeds per (model, S) cell (2 for S>=16 if runtime demands it --
    see NOISE_SEEDS_LARGE_S below), mean/std recorded.
  - derived quantities: gain = acc(S=32) - acc(S=1), gap_to_det = det - acc(S=32).

Writes results/sanity_check.json (per-model detail + per-width aggregates over
training seeds) and prints a compact summary table.

Run: .venv/bin/python experiments/sanity_check.py
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
sys.path.insert(0, str(ROOT / "models"))

from models.net import MLP
from models.pdnn import NoiseConfig, evaluate

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
BASELINE_JSON = ROOT / "results" / "baseline_acc.json"
OUT_JSON = ROOT / "results" / "sanity_check.json"

WIDTHS = ("large", "mid", "small")
TRAIN_SEEDS = (0, 1, 2)
S_VALUES = (1, 2, 4, 8, 16, 32)
NOISE_SEEDS = (1000, 1001, 1002)  # 3 noise seeds per (model, S) cell


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


def main():
    torch.set_num_threads(8)
    t0 = time.time()

    X, y = load_test_set()
    print(f"Loaded MNIST test set: X={tuple(X.shape)} y={tuple(y.shape)}")

    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    cfg = NoiseConfig()  # ideal: no defects, independent uniform r

    results = {"_meta": {
        "widths": WIDTHS,
        "train_seeds": TRAIN_SEEDS,
        "S_values": S_VALUES,
        "noise_seeds": NOISE_SEEDS,
        "n_test": int(X.shape[0]),
    }}

    summary_rows = []  # (tag, seed_label, det, acc_s1, acc_s16, acc_s32, gain, gap)

    for tag in WIDTHS:
        width_entry = {}
        det_accs = []
        gains = []
        gaps = []
        acc_by_S_per_seed = {S: [] for S in S_VALUES}

        for seed in TRAIN_SEEDS:
            model, ckpt = load_model(tag, seed)
            det_acc = deterministic_accuracy(model, X, y)
            ckpt_stored_acc = ckpt.get("test_acc")
            baseline_acc = baseline.get(f"{tag}_s{seed}", {}).get("test_acc")

            det_mismatch_ckpt = abs(det_acc - ckpt_stored_acc) if ckpt_stored_acc is not None else None
            det_mismatch_baseline = abs(det_acc - baseline_acc) if baseline_acc is not None else None

            per_S = {}
            for S in S_VALUES:
                n_seeds = len(NOISE_SEEDS)
                accs = []
                for ns in NOISE_SEEDS[:n_seeds]:
                    acc = evaluate(model, (X, y), S=S, cfg=cfg, seed=ns)
                    accs.append(acc)
                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs))
                per_S[str(S)] = {
                    "mean": mean_acc,
                    "std": std_acc,
                    "raw": accs,
                    "noise_seeds_used": list(NOISE_SEEDS[:n_seeds]),
                }
                acc_by_S_per_seed[S].append(mean_acc)
                print(f"  {tag} s{seed} S={S:>2d}: acc mean={mean_acc:.4f} std={std_acc:.4f} "
                      f"(raw={[f'{a:.4f}' for a in accs]})")

            gain = per_S[str(S_VALUES[-1])]["mean"] - per_S[str(S_VALUES[0])]["mean"]
            gap = det_acc - per_S[str(S_VALUES[-1])]["mean"]

            width_entry[f"s{seed}"] = {
                "tag": tag,
                "seed": seed,
                "det_acc": det_acc,
                "ckpt_stored_test_acc": ckpt_stored_acc,
                "baseline_json_test_acc": baseline_acc,
                "det_mismatch_vs_ckpt": det_mismatch_ckpt,
                "det_mismatch_vs_baseline_json": det_mismatch_baseline,
                "pdnn_acc_by_S": per_S,
                "gain_S1_to_S32": gain,
                "gap_to_det_at_S32": gap,
            }

            det_accs.append(det_acc)
            gains.append(gain)
            gaps.append(gap)

            summary_rows.append((
                tag, f"s{seed}", det_acc,
                per_S["1"]["mean"], per_S["16"]["mean"], per_S["32"]["mean"],
                gain, gap,
            ))

        width_entry["_aggregate"] = {
            "det_acc_mean": float(np.mean(det_accs)),
            "det_acc_std": float(np.std(det_accs)),
            "gain_S1_to_S32_mean": float(np.mean(gains)),
            "gain_S1_to_S32_std": float(np.std(gains)),
            "gap_to_det_at_S32_mean": float(np.mean(gaps)),
            "gap_to_det_at_S32_std": float(np.std(gaps)),
            "pdnn_acc_by_S_mean": {
                str(S): float(np.mean(acc_by_S_per_seed[S])) for S in S_VALUES
            },
            "pdnn_acc_by_S_std": {
                str(S): float(np.std(acc_by_S_per_seed[S])) for S in S_VALUES
            },
        }
        summary_rows.append((
            tag, "AGG", width_entry["_aggregate"]["det_acc_mean"],
            width_entry["_aggregate"]["pdnn_acc_by_S_mean"]["1"],
            width_entry["_aggregate"]["pdnn_acc_by_S_mean"]["16"],
            width_entry["_aggregate"]["pdnn_acc_by_S_mean"]["32"],
            width_entry["_aggregate"]["gain_S1_to_S32_mean"],
            width_entry["_aggregate"]["gap_to_det_at_S32_mean"],
        ))

        results[tag] = width_entry

    total_dt = time.time() - t0
    results["_meta"]["total_seconds"] = total_dt

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"{'width':<8}{'seed':<6}{'det':>8}{'S=1':>8}{'S=16':>8}{'S=32':>8}"
          f"{'gain':>8}{'gap':>8}")
    print("-" * 78)
    for tag, label, det, s1, s16, s32, gain, gap in summary_rows:
        print(f"{tag:<8}{label:<6}{det*100:>7.2f}%{s1*100:>7.2f}%{s16*100:>7.2f}%"
              f"{s32*100:>7.2f}%{gain*100:>7.2f}%{gap*100:>7.2f}%")
    print("=" * 78)
    print(f"Total runtime: {total_dt:.1f}s")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
