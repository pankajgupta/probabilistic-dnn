"""Train baseline (deterministic, tanh-hidden) LeNet-style CNNs on full MNIST.

Trains 2 seeds with plain backprop -- the frozen-weight baselines later
wrapped with p-neuron activations at inference (models/pdnn.py's
pdnn_forward_conv, the "add noise to a trained model" route,
docs/HANDOFF.md section 3). Mirrors models/train_baselines.py's conventions.

Outputs:
  results/models/cnn_s{seed}.pt   - checkpoint: state_dict + config dict
  results/cnn_baseline.json       - full 10k test accuracy per model + stats

Run: .venv/bin/python models/train_cnn.py [--narrow]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
ACC_JSON = ROOT / "results" / "cnn_baseline.json"

SEEDS = (0, 1)
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 4
NUM_THREADS = 8

# default architecture per the brief; --narrow halves conv channels + fc
# width (used for the "retrain narrower" gate-(c) fallback, see
# experiments/cnn_sanity.py / docs/HANDOFF.md sampling-gain discussion --
# narrower nets show a bigger multi-sample gain in the MLP study).
DEFAULT_CONV_CHANNELS = (8, 16)
DEFAULT_FC_HIDDEN = 64
NARROW_CONV_CHANNELS = (4, 8)
NARROW_FC_HIDDEN = 32


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def train_one(tag, conv_channels, fc_hidden, seed, train_ds, test_ds):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g
    )
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = CNN(conv_channels=conv_channels, fc_hidden=fc_hidden)
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
            "conv_channels": list(conv_channels),
            "fc_hidden": fc_hidden,
            "seed": seed,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "tag": tag,
            "test_acc": acc,
            "num_params": model.num_params(),
        },
        ckpt_path,
    )
    return acc, model.num_params()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrow", action="store_true",
                         help="halve conv channels + fc width (gate-(c) fallback)")
    args = parser.parse_args()

    torch.set_num_threads(NUM_THREADS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.narrow:
        tag = "cnn_narrow"
        conv_channels, fc_hidden = NARROW_CONV_CHANNELS, NARROW_FC_HIDDEN
        out_json = ROOT / "results" / "cnn_narrow_baseline.json"
    else:
        tag = "cnn"
        conv_channels, fc_hidden = DEFAULT_CONV_CHANNELS, DEFAULT_FC_HIDDEN
        out_json = ACC_JSON

    train_ds, test_ds = load_data()

    results = {}
    t0 = time.time()
    accs = []
    for seed in SEEDS:
        t_start = time.time()
        acc, n_params = train_one(tag, conv_channels, fc_hidden, seed, train_ds, test_ds)
        dt = time.time() - t_start
        print(f"{tag} (conv={conv_channels}, fc={fc_hidden}) seed={seed}: "
              f"test_acc={acc:.4f}  ({dt:.1f}s)  params={n_params}")
        results[f"{tag}_s{seed}"] = {
            "tag": tag,
            "conv_channels": list(conv_channels),
            "fc_hidden": fc_hidden,
            "seed": seed,
            "test_acc": acc,
            "num_params": n_params,
            "train_seconds": dt,
        }
        accs.append(acc)
    results[f"{tag}_mean"] = float(np.mean(accs))
    results[f"{tag}_std"] = float(np.std(accs))
    total_dt = time.time() - t0

    results["_meta"] = {
        "conv_channels": list(conv_channels),
        "fc_hidden": fc_hidden,
        "seeds": list(SEEDS),
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_threads": NUM_THREADS,
        "total_train_seconds": total_dt,
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal training time: {total_dt:.1f}s")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
