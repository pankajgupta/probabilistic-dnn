"""E6: pool-compatible p-bit order for the CNN (models/pdnn.py's
NoiseConfig.binarize_after_pool).

CNN-A established: the naive conv-path order (comparator BEFORE maxpool,
i.e. pool applied to the p-bit codes m) is biased -- E[maxpool(m)] !=
maxpool(E[m]) = maxpool(tanh(z)) -- and gives near-chance accuracy on the
trained MNIST CNN regardless of S (see results/cnn_sanity.json). This
script uses the fix (pool applied to the continuous tanh, THEN the
comparator: m = sign(pool(tanh(z)) - r), see models/pdnn.py's
pdnn_forward_conv lookahead when cfg.binarize_after_pool=True) and asks:
does the fix actually restore the multi-sample p-DNN accuracy benefit?

Section 1 (Task 2): SANITY GATE. Both training seeds, ideal (independent
Uniform[-1,1]) noise, full 10k MNIST test set, S in {1,2,4,8,16,32,64}, 3
noise seeds/cell. GATE: gap = det_acc - acc(S=64) <= ~1.5pp AND
gain = acc(S=64) - acc(S=1) >= 3pp (aggregate over the 2 seeds). If the
gate fails, section 2 is skipped -- Task 3 (the E3-core sweep) is only
meaningful once the fixed order actually recovers a working S-curve.

Section 2 (Task 3, gate permitting): E3-core sweep, full 10k test, both
train seeds, 5 noise seeds/cell, all cells use binarize_after_pool=True.
  2a. rho in {0, 0.9, 1.0} x S in {1,4,16,64} (AR(1) copula along the
      sample axis, as in the MLP's experiments/e3_sampling_fragility.py).
  2b. k=2 (bitdepth), rho=0 x same S (coarse-but-independent contrast).
  2c. conv_broadcast in {share_per_channel, share_per_position}, rho=0
      (iid across samples) x same S -- the CNN-specific hardware-sharing
      question: does per-feature-map / per-position noise reuse behave
      like the MLP's shared_across_neurons mode (S=1 penalty, gain intact)
      or worse?

Writes results/e6_cnn.json (self-describing meta, per-cell mean/std/raw,
det refs). Frozen weights throughout (results/models/cnn_s{0,1}.pt, no
retraining) -- only the injected p-bit randomness / stage order changes.

Run: .venv/bin/python experiments/e6_cnn.py
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

from models.cnn import CNN
from models.pdnn import NoiseConfig, evaluate_conv

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
BASELINE_JSON = ROOT / "results" / "cnn_baseline.json"
OUT_JSON = ROOT / "results" / "e6_cnn.json"

TRAIN_SEEDS = (0, 1)

# --- Task 2: sanity gate ---
SANITY_S = (1, 2, 4, 8, 16, 32, 64)
SANITY_NOISE_SEEDS = (3000, 3001, 3002)  # distinct from cnn_sanity.py's 1000s
GATE_GAP_MAX = 0.015   # 1.5pp
GATE_GAIN_MIN = 0.03   # 3pp

# --- Task 3: E3-core sweep (only if gate passes) ---
SWEEP_S = (1, 4, 16, 64)
SWEEP_NOISE_SEEDS = (4000, 4001, 4002, 4003, 4004)  # 5 noise seeds/cell
SWEEP_RHOS = (0.0, 0.9, 1.0)
SWEEP_K = 2
SWEEP_BROADCAST_MODES = ("share_per_channel", "share_per_position")

MLP_MID_GAIN_REF = 0.1375  # docs/COLLABORATOR_HANDOFF.md V6 table, rho=0, mid MLP, 13.75pp


# ---------------------------------------------------------------------------
# data / models
# ---------------------------------------------------------------------------

def load_test_set():
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    X = test_ds.data.float().div(255.0).view(len(test_ds), 1, 28, 28)
    y = test_ds.targets.clone()
    return X, y


def load_model(seed):
    ckpt = torch.load(MODELS_DIR / f"cnn_s{seed}.pt", map_location="cpu")
    model = CNN(conv_channels=tuple(ckpt["conv_channels"]), fc_hidden=ckpt["fc_hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def deterministic_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# cell runner
# ---------------------------------------------------------------------------

def run_cell(model, X, y, S, cfg, noise_seeds):
    accs = [evaluate_conv(model, (X, y), S=S, cfg=cfg, seed=ns) for ns in noise_seeds]
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "raw": accs,
        "noise_seeds_used": list(noise_seeds),
    }


def aggregate_over_seeds(per_seed_means):
    return {"mean": float(np.mean(per_seed_means)), "std": float(np.std(per_seed_means))}


# ---------------------------------------------------------------------------
# Task 2: sanity gate
# ---------------------------------------------------------------------------

def run_sanity_gate(X, y, baseline):
    cfg = NoiseConfig(binarize_after_pool=True)
    per_seed = {}
    det_accs, gains, gaps = [], [], []
    acc_by_S_agg = {S: [] for S in SANITY_S}

    for seed in TRAIN_SEEDS:
        model, ckpt = load_model(seed)
        det_acc = deterministic_accuracy(model, X, y)
        ckpt_acc = ckpt.get("test_acc")
        baseline_acc = baseline.get(f"cnn_s{seed}", {}).get("test_acc")

        per_S = {}
        t0 = time.time()
        for S in SANITY_S:
            cell = run_cell(model, X, y, S, cfg, SANITY_NOISE_SEEDS)
            per_S[str(S)] = cell
            acc_by_S_agg[S].append(cell["mean"])
            print(f"  sanity s{seed} S={S:>2d}: acc mean={cell['mean']:.4f} "
                  f"std={cell['std']:.4f} raw={[f'{a:.4f}' for a in cell['raw']]}")
        dt = time.time() - t0

        gain = per_S[str(SANITY_S[-1])]["mean"] - per_S[str(SANITY_S[0])]["mean"]
        gap = det_acc - per_S[str(SANITY_S[-1])]["mean"]
        det_accs.append(det_acc)
        gains.append(gain)
        gaps.append(gap)

        per_seed[f"s{seed}"] = {
            "seed": seed,
            "det_acc": det_acc,
            "ckpt_stored_test_acc": ckpt_acc,
            "baseline_json_test_acc": baseline_acc,
            "det_mismatch_vs_ckpt": abs(det_acc - ckpt_acc) if ckpt_acc is not None else None,
            "pdnn_acc_by_S": per_S,
            "gain_S1_to_S64": gain,
            "gap_to_det_at_S64": gap,
            "seconds": dt,
        }
        print(f"  sanity s{seed} done in {dt:.1f}s: det={det_acc:.4f} "
              f"gain={gain*100:.2f}pp gap={gap*100:.2f}pp")

    agg_by_S = {str(S): aggregate_over_seeds(acc_by_S_agg[S]) for S in SANITY_S}
    agg_det = float(np.mean(det_accs))
    agg_gain = float(np.mean(gains))
    agg_gap = float(np.mean(gaps))
    gate_pass = (agg_gap <= GATE_GAP_MAX) and (agg_gain >= GATE_GAIN_MIN)

    return {
        "per_seed": per_seed,
        "aggregate": {
            "acc_by_S": agg_by_S,
            "det_acc_mean": agg_det,
            "gain_S1_to_S64_mean": agg_gain,
            "gap_to_det_at_S64_mean": agg_gap,
        },
        "gate": {
            "gap_max_allowed": GATE_GAP_MAX,
            "gain_min_required": GATE_GAIN_MIN,
            "gap_actual": agg_gap,
            "gain_actual": agg_gain,
            "pass": gate_pass,
        },
    }


def run_diagnostic(X, y):
    """One diagnostic round if the gate fails: isolate whether the residual
    gap is (i) still concentrated at the conv/pool stages themselves, by
    comparing accuracy with the fc1 stage's comparator ALSO disabled (kept
    deterministic/tanh, only conv stages stochastic) vs (ii) a generic
    multi-stage p-bit stacking bias (present even for a single stochastic
    stage), by comparing S=64 accuracy when ONLY ONE of the three stochastic
    stages (conv1+pool1, conv2+pool2, fc1) is noised at a time, holding the
    other two at their deterministic tanh value (S=1 with r fixed at 0 is
    NOT what we want -- instead we directly monkeypatch by building a
    reduced-stage model view). Implemented directly (not via pdnn.py, to
    avoid growing production surface for a one-off diagnostic): manual
    stage replay per model, S stochastic passes only at the targeted stage.
    """
    model, _ = load_model(0)
    Xs, ys = X[:2000], y[:2000]
    with torch.no_grad():
        det_acc = (model(Xs).argmax(dim=1) == ys).float().mean().item()

    S = 64
    gen_seed = 5000

    def replay(stochastic_stage):
        """stochastic_stage in {'conv1','conv2','fc1', 'all', 'none'}."""
        from models.pdnn import _apply_over_sb, make_noise, make_noise_conv
        gen = torch.Generator().manual_seed(gen_seed)
        stages = model.stages
        n = len(stages)
        x = Xs.unsqueeze(0)  # (1,B,1,28,28), S folded in via broadcast per-stage
        i = 0
        stage_names = {0: "conv1", 2: "conv2", 5: "fc1"}
        while i < n:
            layer, kind = stages[i]
            if kind == "conv":
                a = torch.tanh(_apply_over_sb(x, layer))
                next_kind = stages[i + 1][1] if i + 1 < n else None
                if next_kind == "pool":
                    pool_layer = stages[i + 1][0]
                    pooled_a = _apply_over_sb(a, pool_layer)
                    name = stage_names.get(i)
                    noisy = stochastic_stage in ("all", name)
                    if noisy:
                        target_shape = (S,) + pooled_a.shape[1:]
                        pooled_a_exp = pooled_a.expand(target_shape) if pooled_a.shape[0] == 1 \
                            else pooled_a
                        r = make_noise_conv(target_shape, gen, NoiseConfig())
                        x = torch.where(pooled_a_exp - r >= 0, torch.ones_like(pooled_a_exp),
                                         -torch.ones_like(pooled_a_exp))
                    else:
                        x = pooled_a
                    i += 2
                    continue
                x = a
            elif kind == "pool":
                x = _apply_over_sb(x, layer)
            elif kind == "flatten":
                x = x.reshape(x.shape[0], x.shape[1], -1)
            elif kind == "fc":
                a = torch.tanh(_apply_over_sb(x, layer))
                name = stage_names.get(i)
                noisy = stochastic_stage in ("all", name)
                if noisy:
                    target_shape = (S,) + a.shape[1:]
                    a_exp = a.expand(target_shape) if a.shape[0] == 1 else a
                    r = make_noise(target_shape, gen, NoiseConfig())
                    x = torch.where(a_exp - r >= 0, torch.ones_like(a_exp), -torch.ones_like(a_exp))
                else:
                    x = a
            elif kind == "fc_out":
                x = _apply_over_sb(x, layer)
            else:
                raise ValueError(kind)
            i += 1
        avg_logits = x.mean(dim=0)
        preds = avg_logits.argmax(dim=1)
        return (preds == ys).float().mean().item()

    rows = {}
    for stage in ("none", "conv1", "conv2", "fc1", "all"):
        acc = replay(stage)
        rows[stage] = acc
        print(f"  diagnostic stochastic_stage={stage:>6s}: acc={acc:.4f} "
              f"(det={det_acc:.4f}, gap={det_acc-acc:.4f})")

    return {"n_subset": int(Xs.shape[0]), "S": S, "det_acc": det_acc, "acc_by_stage": rows}


# ---------------------------------------------------------------------------
# Task 3: E3-core sweep
# ---------------------------------------------------------------------------

def run_sweep_rho(X, y):
    out = {}
    for rho in SWEEP_RHOS:
        cfg = NoiseConfig(binarize_after_pool=True, rho=rho)
        agg = {str(S): [] for S in SWEEP_S}
        seed_entries = {}
        for seed in TRAIN_SEEDS:
            model, _ = load_model(seed)
            S_entry = {}
            t0 = time.time()
            for S in SWEEP_S:
                cell = run_cell(model, X, y, S, cfg, SWEEP_NOISE_SEEDS)
                S_entry[str(S)] = cell
                agg[str(S)].append(cell["mean"])
            seed_entries[f"s{seed}"] = S_entry
            print(f"  sweep_rho rho={rho} s{seed} done in {time.time()-t0:.1f}s")
        seed_entries["_aggregate"] = {S_key: aggregate_over_seeds(v) for S_key, v in agg.items()}
        out[str(rho)] = seed_entries
    return out


def run_sweep_k2(X, y):
    cfg = NoiseConfig(binarize_after_pool=True, bitdepth=SWEEP_K, rho=0.0)
    agg = {str(S): [] for S in SWEEP_S}
    seed_entries = {}
    for seed in TRAIN_SEEDS:
        model, _ = load_model(seed)
        S_entry = {}
        t0 = time.time()
        for S in SWEEP_S:
            cell = run_cell(model, X, y, S, cfg, SWEEP_NOISE_SEEDS)
            S_entry[str(S)] = cell
            agg[str(S)].append(cell["mean"])
        seed_entries[f"s{seed}"] = S_entry
        print(f"  sweep_k2 s{seed} done in {time.time()-t0:.1f}s")
    seed_entries["_aggregate"] = {S_key: aggregate_over_seeds(v) for S_key, v in agg.items()}
    return seed_entries


def run_sweep_broadcast(X, y):
    out = {}
    for mode in SWEEP_BROADCAST_MODES:
        cfg = NoiseConfig(binarize_after_pool=True, rho=0.0, conv_broadcast=mode)
        agg = {str(S): [] for S in SWEEP_S}
        seed_entries = {}
        for seed in TRAIN_SEEDS:
            model, _ = load_model(seed)
            S_entry = {}
            t0 = time.time()
            for S in SWEEP_S:
                cell = run_cell(model, X, y, S, cfg, SWEEP_NOISE_SEEDS)
                S_entry[str(S)] = cell
                agg[str(S)].append(cell["mean"])
            seed_entries[f"s{seed}"] = S_entry
            print(f"  sweep_broadcast {mode} s{seed} done in {time.time()-t0:.1f}s")
        seed_entries["_aggregate"] = {S_key: aggregate_over_seeds(v) for S_key, v in agg.items()}
        out[mode] = seed_entries
    return out


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_sanity_table(sanity):
    print("\n" + "=" * 90)
    print("TASK 2 -- SANITY GATE (binarize_after_pool=True, ideal noise, full 10k, "
          "aggregate over 2 train seeds)")
    header = f"{'S':>6}" + "".join("" for _ in [None])
    row_S = f"{'S=':>8}" + "".join(f"{S:>9d}" for S in SANITY_S)
    row_acc = f"{'acc%':>8}" + "".join(
        f"{sanity['aggregate']['acc_by_S'][str(S)]['mean']*100:>9.3f}" for S in SANITY_S)
    print(row_S)
    print(row_acc)
    g = sanity["gate"]
    print(f"det_acc(mean)={sanity['aggregate']['det_acc_mean']*100:.3f}%  "
          f"gap@S64={g['gap_actual']*100:.3f}pp (<= {g['gap_max_allowed']*100:.2f}pp)  "
          f"gain S1->S64={g['gain_actual']*100:.3f}pp (>= {g['gain_min_required']*100:.2f}pp)  "
          f"GATE: {'PASS' if g['pass'] else 'FAIL'}")
    print("=" * 90)


def print_sweep_rho_table(sweep_rho):
    print("\nTASK 3a -- rho sweep (aggregate over 2 train seeds, 5 noise seeds/cell):")
    header = f"{'rho':>6}" + "".join(f"{'S='+str(S):>10}" for S in SWEEP_S)
    print(header)
    for rho in SWEEP_RHOS:
        agg = sweep_rho[str(rho)]["_aggregate"]
        row = f"{rho:>6}" + "".join(f"{agg[str(S)]['mean']*100:>9.3f}%" for S in SWEEP_S)
        print(row)


def print_sweep_k2_table(sweep_k2):
    print(f"\nTASK 3b -- k={SWEEP_K}, rho=0 (coarse-but-independent):")
    header = "".join(f"{'S='+str(S):>10}" for S in SWEEP_S)
    print(header)
    agg = sweep_k2["_aggregate"]
    row = "".join(f"{agg[str(S)]['mean']*100:>9.3f}%" for S in SWEEP_S)
    print(row)


def print_sweep_broadcast_table(sweep_broadcast):
    print("\nTASK 3c -- conv_broadcast sweep (rho=0):")
    header = f"{'mode':>20}" + "".join(f"{'S='+str(S):>10}" for S in SWEEP_S)
    print(header)
    for mode in SWEEP_BROADCAST_MODES:
        agg = sweep_broadcast[mode]["_aggregate"]
        row = f"{mode:>20}" + "".join(f"{agg[str(S)]['mean']*100:>9.3f}%" for S in SWEEP_S)
        print(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    torch.set_num_threads(8)
    t0 = time.time()

    X, y = load_test_set()
    print(f"Loaded MNIST test set: X={tuple(X.shape)} y={tuple(y.shape)}")

    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    results = {"_meta": {
        "description": "E6: CNN pool-compatible p-bit order "
                        "(NoiseConfig.binarize_after_pool=True) -- sanity gate + "
                        "E3-core sweep. Frozen weights (results/models/cnn_s{0,1}.pt).",
        "train_seeds": TRAIN_SEEDS,
        "n_test": int(X.shape[0]),
        "sanity": {"S_values": SANITY_S, "noise_seeds": SANITY_NOISE_SEEDS,
                   "gate_gap_max": GATE_GAP_MAX, "gate_gain_min": GATE_GAIN_MIN},
        "sweep": {"S_values": SWEEP_S, "noise_seeds": SWEEP_NOISE_SEEDS,
                  "rhos": SWEEP_RHOS, "k": SWEEP_K,
                  "broadcast_modes": SWEEP_BROADCAST_MODES},
    }}

    print("\n[1/2] TASK 2: sanity gate ...")
    t = time.time()
    sanity = run_sanity_gate(X, y, baseline)
    sanity_dt = time.time() - t
    results["sanity"] = sanity
    results["_meta"]["sanity_seconds"] = sanity_dt
    print_sanity_table(sanity)
    print(f"sanity gate done in {sanity_dt:.1f}s")

    gate_pass = sanity["gate"]["pass"]

    if not gate_pass:
        print("\nGATE FAILED -- running one diagnostic round, then stopping "
              "(Task 3 skipped per instructions).")
        t = time.time()
        diagnostic = run_diagnostic(X, y)
        diag_dt = time.time() - t
        results["diagnostic"] = diagnostic
        results["_meta"]["diagnostic_seconds"] = diag_dt
        print(f"diagnostic done in {diag_dt:.1f}s")
        results["_meta"]["task3_ran"] = False
    else:
        print("\n[2/2] TASK 3: E3-core sweep ...")
        t = time.time()
        sweep_rho = run_sweep_rho(X, y)
        print_sweep_rho_table(sweep_rho)
        sweep_k2 = run_sweep_k2(X, y)
        print_sweep_k2_table(sweep_k2)
        sweep_broadcast = run_sweep_broadcast(X, y)
        print_sweep_broadcast_table(sweep_broadcast)
        sweep_dt = time.time() - t
        results["sweep"] = {
            "rho": sweep_rho,
            "k2_rho0": sweep_k2,
            "broadcast": sweep_broadcast,
        }
        results["_meta"]["sweep_seconds"] = sweep_dt
        results["_meta"]["task3_ran"] = True
        print(f"sweep done in {sweep_dt:.1f}s")

    total_dt = time.time() - t0
    results["_meta"]["total_seconds"] = total_dt

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal runtime: {total_dt:.1f}s")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
