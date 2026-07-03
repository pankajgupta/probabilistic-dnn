"""Route-2 training: sample-aware STE training of the MID (64,32) p-DNN.

Phase C / H5. All Phase B results use route 1 (train deterministic tanh-MLP,
inject p-bit noise only at inference). This script implements route 2 from [2]:
train WITH the stochastic p-bit hidden activation in the loop, using a
straight-through estimator (STE).

Per hidden layer, given pre-activation z:
    a       = tanh(z)                          # smooth surrogate
    r       ~ Uniform[-1,1]  (fresh, ideal)    # IDEAL noise during training
    m_stoch = sign(a - r)  in {-1,+1}          # stochastic binary activation
    m       = m_stoch.detach() + a - a.detach()# STE: forward=m_stoch, grad=da

The output layer is a plain linear layer on the {-1,+1} hidden codes -- exactly
the convention models/pdnn.py uses at inference, so a checkpoint trained here
loads straight into models.net.MLP and is scored by the identical evaluate()
protocol as the frozen baselines.

Two S_train variants:
  S_train=1: one stochastic forward pass per step; loss on its logits.
  S_train=4: four stochastic forward passes per step; AVERAGE the 4 logits,
             then one CE loss on the averaged logits; backprop flows through
             all four passes.

Training seeds {0,1}. Adam 1e-3, batch 128, 6 epochs (see EPOCHS).

Checkpoints (models.net.MLP-compatible):
  results/models/ste_S{Strain}_s{seed}.pt
Curve/metadata summary:
  results/ste_train.json

Run: .venv/bin/python models/ste_train.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.net import MLP
from models.pdnn import NoiseConfig, evaluate

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
OUT_JSON = ROOT / "results" / "ste_train.json"

HIDDEN = (64, 32)          # MID architecture
TAG = "mid"
S_TRAIN_VALUES = (1, 4)
SEEDS = (0, 1)
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 6                 # may be nudged; recorded in output
NUM_THREADS = 8

# Eval-mode headroom for the timebox gate: score the trained net under its own
# stochastic p-bit inference mode with S=16 (ideal noise), full 10k test set.
GATE_S = 16
GATE_THRESHOLD = 0.90


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def test_tensors(test_ds):
    X = test_ds.data.float().div(255.0).view(len(test_ds), -1)
    y = test_ds.targets.clone()
    return X, y


def ste_forward(fcs, x, S, gen):
    """S stochastic STE passes; return AVERAGED logits (differentiable).

    fcs: nn.ModuleList (hidden layers tanh->p-bit, last layer linear).
    x:   (B, 784). gen: torch.Generator for the training-time ideal noise.
    """
    hidden = fcs[:-1]
    out = fcs[-1]
    logits_sum = None
    for _ in range(S):
        h = x
        for fc in hidden:
            z = fc(h)
            a = torch.tanh(z)
            r = torch.rand(z.shape, generator=gen) * 2.0 - 1.0  # Uniform[-1,1], ideal
            m_stoch = torch.where(a - r >= 0, torch.ones_like(a), -torch.ones_like(a))
            h = m_stoch.detach() + a - a.detach()               # STE
        logits = out(h)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / S


def det_tanh_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one(S_train, seed, train_ds, X_test, y_test):
    set_seed(seed)
    g_data = torch.Generator().manual_seed(seed)             # dataloader shuffle
    g_noise = torch.Generator().manual_seed(10_000 + seed)   # training p-bit noise
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_data)

    model = MLP(hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses = []
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        running = 0.0
        nb = 0
        for xb, yb in train_loader:
            xb = xb.view(xb.shape[0], -1)
            opt.zero_grad()
            logits = ste_forward(model.fcs, xb, S_train, g_noise)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        epoch_losses.append(running / nb)
        print(f"    S_train={S_train} seed={seed} epoch {epoch}: train_loss={running/nb:.4f}")

    train_seconds = time.time() - t0

    # own-mode stochastic eval (ideal noise) at the gate S, plus deterministic tanh
    model.eval()
    ideal = NoiseConfig()
    gate_acc = evaluate(model, (X_test, y_test), S=GATE_S, cfg=ideal, seed=2000)
    det_acc = det_tanh_accuracy(model, X_test, y_test)

    ckpt_path = MODELS_DIR / f"ste_S{S_train}_s{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": HIDDEN,
            "seed": seed,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "tag": TAG,
            "route": "ste",
            "S_train": S_train,
            "train_noise": "ideal Uniform[-1,1]",
            "epoch_losses": epoch_losses,
            "train_seconds": train_seconds,
            "gate_S": GATE_S,
            "gate_stoch_acc": gate_acc,
            "det_tanh_acc": det_acc,
        },
        ckpt_path,
    )
    print(f"    -> saved {ckpt_path.name}: gate(S={GATE_S}) stoch_acc={gate_acc:.4f}  "
          f"det_tanh_acc={det_acc:.4f}  ({train_seconds:.1f}s)")
    return {
        "S_train": S_train, "seed": seed, "epoch_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1], "gate_stoch_acc": gate_acc,
        "det_tanh_acc": det_acc, "train_seconds": train_seconds,
        "ckpt": str(ckpt_path.name),
    }


def main():
    torch.set_num_threads(NUM_THREADS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = load_data()
    X_test, y_test = test_tensors(test_ds)
    print(f"MNIST train={len(train_ds)} test={tuple(X_test.shape)}")

    runs = []
    t0 = time.time()
    for S_train in S_TRAIN_VALUES:
        for seed in SEEDS:
            print(f"  training STE S_train={S_train} seed={seed} ...")
            runs.append(train_one(S_train, seed, train_ds, X_test, y_test))
    total_seconds = time.time() - t0

    # timebox gate: S_train=1 variant must reach >= GATE_THRESHOLD under own
    # stochastic (S=GATE_S) eval, else record failure.
    s1_gate = [r["gate_stoch_acc"] for r in runs if r["S_train"] == 1]
    s1_pass = all(a >= GATE_THRESHOLD for a in s1_gate)

    out = {
        "_meta": {
            "description": "Route-2 STE (sample-aware) training of MID (64,32) p-DNN. "
                           "STE forward=sign(tanh(z)-r), backward=d/dz tanh(z); ideal "
                           "Uniform[-1,1] noise during training. Checkpoints are "
                           "models.net.MLP-compatible and scored by the same "
                           "models.pdnn.evaluate protocol as the frozen baselines.",
            "hidden": list(HIDDEN),
            "tag": TAG,
            "S_train_values": list(S_TRAIN_VALUES),
            "seeds": list(SEEDS),
            "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "gate": {
                "S": GATE_S, "threshold": GATE_THRESHOLD,
                "mode": "own stochastic p-bit inference, ideal noise, full 10k test",
                "applies_to": "S_train=1 variant",
                "s1_gate_accs": s1_gate,
                "passed": bool(s1_pass),
            },
            "total_seconds": total_seconds,
        },
        "runs": runs,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'variant':>16}{'final_loss':>12}{'gate_acc':>10}{'det_acc':>10}")
    for r in runs:
        print(f"{'S%d_s%d' % (r['S_train'], r['seed']):>16}"
              f"{r['final_train_loss']:>12.4f}{r['gate_stoch_acc']*100:>9.2f}%"
              f"{r['det_tanh_acc']*100:>9.2f}%")
    print(f"\nTIMEBOX GATE (S_train=1, S={GATE_S} stoch >= {GATE_THRESHOLD*100:.0f}%): "
          f"{'PASS' if s1_pass else 'FAIL'}  accs={['%.4f' % a for a in s1_gate]}")
    print(f"Total training time: {total_seconds:.1f}s")
    print(f"Wrote {OUT_JSON}")
    return s1_pass


if __name__ == "__main__":
    main()
