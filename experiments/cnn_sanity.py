"""CNN-extension sanity check: does the p-DNN multi-sample accuracy benefit
(reproduced for the MLP in results/sanity_check.json) survive the move to a
LeNet-style CNN? IDEAL noise only (no defects, independent-across-samples
uniforms) -- this is the CNN analogue of experiments/sanity_check.py, not
the H3 correlation sweep (that would be a CNN analogue of
experiments/e3_sampling_fragility.py, out of this script's scope).

For each training seed (0, 1):
  - loads results/models/{tag}_s{seed}.pt, rebuilds the CNN, recomputes the
    deterministic test accuracy on the full 10k test set (compared against
    the checkpoint's stored test_acc and results/cnn_baseline.json as a
    cross-check, not trusted blindly).
  - evaluates the conv p-DNN (models.pdnn.evaluate_conv, NoiseConfig() =
    ideal, independent Uniform[-1,1] noise at every tanh -- conv feature
    maps AND fc hidden) at S in {1,2,4,8,16,32,64}, 3 noise seeds per
    (model, S) cell, mean/std recorded.
  - derived quantities: gain = acc(S=64) - acc(S=1), gap = det - acc(S=64).

Writes results/cnn_sanity.json (default architecture) or
results/cnn_narrow_sanity.json (--narrow, the gate-(c) fallback architecture
from models/train_cnn.py --narrow) and prints a compact summary table.

Run: .venv/bin/python experiments/cnn_sanity.py [--narrow]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn import CNN
from models.pdnn import NoiseConfig, evaluate_conv

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"

TRAIN_SEEDS = (0, 1)
S_VALUES = (1, 2, 4, 8, 16, 32, 64)
NOISE_SEEDS = (1000, 1001, 1002)  # 3 noise seeds per (model, S) cell


def load_test_set():
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    X = test_ds.data.float().div(255.0).view(len(test_ds), 1, 28, 28)
    y = test_ds.targets.clone()
    return X, y


def load_model(tag, seed):
    ckpt = torch.load(MODELS_DIR / f"{tag}_s{seed}.pt", map_location="cpu")
    model = CNN(conv_channels=tuple(ckpt["conv_channels"]), fc_hidden=ckpt["fc_hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def deterministic_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrow", action="store_true",
                         help="use the halved-width checkpoints (gate-(c) fallback)")
    args = parser.parse_args()

    tag = "cnn_narrow" if args.narrow else "cnn"
    baseline_json = ROOT / "results" / (f"{tag}_baseline.json" if args.narrow
                                         else "cnn_baseline.json")
    out_json = ROOT / "results" / (f"{tag}_sanity.json" if args.narrow
                                    else "cnn_sanity.json")

    torch.set_num_threads(8)
    t0 = time.time()

    X, y = load_test_set()
    print(f"Loaded MNIST test set: X={tuple(X.shape)} y={tuple(y.shape)}")

    with open(baseline_json) as f:
        baseline = json.load(f)

    cfg = NoiseConfig()  # ideal: no defects, independent uniform r

    results = {"_meta": {
        "tag": tag,
        "train_seeds": TRAIN_SEEDS,
        "S_values": S_VALUES,
        "noise_seeds": NOISE_SEEDS,
        "n_test": int(X.shape[0]),
    }}

    summary_rows = []  # (seed_label, det, s1, s16, s64, gain, gap)
    det_accs, gains, gaps = [], [], []
    acc_by_S_per_seed = {S: [] for S in S_VALUES}
    per_seed_results = {}

    for seed in TRAIN_SEEDS:
        model, ckpt = load_model(tag, seed)
        det_acc = deterministic_accuracy(model, X, y)
        ckpt_stored_acc = ckpt.get("test_acc")
        baseline_acc = baseline.get(f"{tag}_s{seed}", {}).get("test_acc")

        det_mismatch_ckpt = abs(det_acc - ckpt_stored_acc) if ckpt_stored_acc is not None else None
        det_mismatch_baseline = abs(det_acc - baseline_acc) if baseline_acc is not None else None

        per_S = {}
        for S in S_VALUES:
            accs = []
            t_s = time.time()
            for ns in NOISE_SEEDS:
                acc = evaluate_conv(model, (X, y), S=S, cfg=cfg, seed=ns)
                accs.append(acc)
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            per_S[str(S)] = {
                "mean": mean_acc,
                "std": std_acc,
                "raw": accs,
                "noise_seeds_used": list(NOISE_SEEDS),
            }
            acc_by_S_per_seed[S].append(mean_acc)
            print(f"  {tag} s{seed} S={S:>2d}: acc mean={mean_acc:.4f} std={std_acc:.4f} "
                  f"(raw={[f'{a:.4f}' for a in accs]})  [{time.time()-t_s:.1f}s]")

        gain = per_S[str(S_VALUES[-1])]["mean"] - per_S[str(S_VALUES[0])]["mean"]
        gap = det_acc - per_S[str(S_VALUES[-1])]["mean"]

        per_seed_results[f"s{seed}"] = {
            "tag": tag,
            "seed": seed,
            "det_acc": det_acc,
            "ckpt_stored_test_acc": ckpt_stored_acc,
            "baseline_json_test_acc": baseline_acc,
            "det_mismatch_vs_ckpt": det_mismatch_ckpt,
            "det_mismatch_vs_baseline_json": det_mismatch_baseline,
            "pdnn_acc_by_S": per_S,
            "gain_S1_to_S64": gain,
            "gap_to_det_at_S64": gap,
        }

        det_accs.append(det_acc)
        gains.append(gain)
        gaps.append(gap)

        summary_rows.append((
            f"s{seed}", det_acc,
            per_S["1"]["mean"], per_S["16"]["mean"], per_S["64"]["mean"],
            gain, gap,
        ))

    aggregate = {
        "det_acc_mean": float(np.mean(det_accs)),
        "det_acc_std": float(np.std(det_accs)),
        "gain_S1_to_S64_mean": float(np.mean(gains)),
        "gain_S1_to_S64_std": float(np.std(gains)),
        "gap_to_det_at_S64_mean": float(np.mean(gaps)),
        "gap_to_det_at_S64_std": float(np.std(gaps)),
        "pdnn_acc_by_S_mean": {str(S): float(np.mean(acc_by_S_per_seed[S])) for S in S_VALUES},
        "pdnn_acc_by_S_std": {str(S): float(np.std(acc_by_S_per_seed[S])) for S in S_VALUES},
    }
    summary_rows.append((
        "AGG", aggregate["det_acc_mean"],
        aggregate["pdnn_acc_by_S_mean"]["1"],
        aggregate["pdnn_acc_by_S_mean"]["16"],
        aggregate["pdnn_acc_by_S_mean"]["64"],
        aggregate["gain_S1_to_S64_mean"],
        aggregate["gap_to_det_at_S64_mean"],
    ))

    results[tag] = per_seed_results
    results[tag]["_aggregate"] = aggregate

    total_dt = time.time() - t0
    results["_meta"]["total_seconds"] = total_dt

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"{'seed':<6}{'det':>8}{'S=1':>8}{'S=16':>8}{'S=64':>8}{'gain':>8}{'gap':>8}")
    print("-" * 78)
    for label, det, s1, s16, s64, gain, gap in summary_rows:
        print(f"{label:<6}{det*100:>7.2f}%{s1*100:>7.2f}%{s16*100:>7.2f}%"
              f"{s64*100:>7.2f}%{gain*100:>7.2f}%{gap*100:>7.2f}%")
    print("=" * 78)
    print(f"Total runtime: {total_dt:.1f}s")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
