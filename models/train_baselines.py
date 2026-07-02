"""Train baseline (deterministic, tanh-hidden) MLPs on full MNIST.

Trains 3 architectures x 3 seeds = 9 models with plain backprop. These are
the frozen-weight baselines that later get wrapped with p-neuron activations
at inference (the "add noise to a trained model" route, HANDOFF.md section 3).

Outputs:
  results/models/{tag}_s{seed}.pt   - checkpoint: state_dict + config dict
  results/baseline_acc.json         - full 10k test accuracy per model + stats

Run: .venv/bin/python models/train_baselines.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import MLP

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
ACC_JSON = ROOT / "results" / "baseline_acc.json"

WIDTHS = {"large": (256, 128), "mid": (64, 32), "small": (32, 16)}
SEEDS = (0, 1, 2)
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 4
NUM_THREADS = 8


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def train_one(tag, hidden, seed, train_ds, test_ds):
    set_seed(seed)
    # generator for the DataLoader shuffle, seeded for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g
    )
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = MLP(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    acc = correct / total

    ckpt_path = MODELS_DIR / f"{tag}_s{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": hidden,
            "seed": seed,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "tag": tag,
            "test_acc": acc,
        },
        ckpt_path,
    )
    return acc


def main():
    torch.set_num_threads(NUM_THREADS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = load_data()

    results = {}
    t0 = time.time()
    for tag, hidden in WIDTHS.items():
        accs = []
        for seed in SEEDS:
            t_start = time.time()
            acc = train_one(tag, hidden, seed, train_ds, test_ds)
            dt = time.time() - t_start
            print(f"{tag} (hidden={hidden}) seed={seed}: test_acc={acc:.4f}  ({dt:.1f}s)")
            results[f"{tag}_s{seed}"] = {
                "tag": tag,
                "hidden": list(hidden),
                "seed": seed,
                "test_acc": acc,
                "train_seconds": dt,
            }
            accs.append(acc)
        results[f"{tag}_mean"] = float(np.mean(accs))
        results[f"{tag}_std"] = float(np.std(accs))
    total_dt = time.time() - t0

    results["_meta"] = {
        "widths": {k: list(v) for k, v in WIDTHS.items()},
        "seeds": list(SEEDS),
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_threads": NUM_THREADS,
        "total_train_seconds": total_dt,
    }

    with open(ACC_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal training time: {total_dt:.1f}s")
    print(f"Wrote {ACC_JSON}")


if __name__ == "__main__":
    main()
