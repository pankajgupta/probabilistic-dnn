"""Tests for models/pdnn.py: noise generation defects and p-DNN forward/evaluate.

Run: .venv/bin/python -m pytest tests/test_pdnn.py -q
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chisquare

if __name__ == "__main__":
    # allow `python tests/test_pdnn.py` to find the repo root package
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pdnn import NoiseConfig, evaluate, make_noise, pdnn_forward, quantize


# ---------------------------------------------------------------------------
# helpers (deliberately not importing models/net.py -- see task scope)
# ---------------------------------------------------------------------------

def _tiny_weights(sizes, seed):
    """List of (W, b) float32 tensors for a small MLP with the given sizes."""
    g = torch.Generator().manual_seed(seed)
    layers = []
    for i in range(len(sizes) - 1):
        W = torch.randn(sizes[i + 1], sizes[i], generator=g) * 0.5
        b = torch.randn(sizes[i + 1], generator=g) * 0.1
        layers.append((W, b))
    return layers


class _TinyFcsNet(nn.Module):
    """Minimal stand-in matching the fcs-ModuleList / tanh convention used
    by models/net.py, built locally so this test file does not import it."""

    def __init__(self, sizes, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.fcs = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = torch.tanh(fc(x))
        return self.fcs[-1](x)


def _deterministic_forward(weights, X):
    """Reference deterministic forward: tanh hidden, linear output."""
    x = X
    for W, b in weights[:-1]:
        x = torch.tanh(x @ W.T + b)
    W, b = weights[-1]
    return x @ W.T + b


# ---------------------------------------------------------------------------
# a. E[m] -> tanh
# ---------------------------------------------------------------------------

def test_expected_pbit_matches_tanh():
    z = torch.linspace(-3.0, 3.0, 25)
    a = torch.tanh(z)
    S = 20_000
    gen = torch.Generator().manual_seed(0)
    r = make_noise((S, 1, len(z)), gen, NoiseConfig())
    m = torch.where(a - r >= 0, torch.ones_like(r), -torch.ones_like(r))
    mean_m = m.mean(dim=0).squeeze(0)
    max_err = (mean_m - a).abs().max().item()
    assert max_err < 0.02, f"max |E[m]-tanh(z)| = {max_err}"


# ---------------------------------------------------------------------------
# b. marginal uniformity under correlation
# ---------------------------------------------------------------------------

def test_marginal_uniform_under_correlation():
    gen = torch.Generator().manual_seed(1)
    cfg = NoiseConfig(rho=0.9)
    r = make_noise((400, 50, 1), gen, cfg).flatten().numpy()
    n_bins = 20
    counts, _ = np.histogram(r, bins=n_bins, range=(-1.0, 1.0))
    expected = np.full(n_bins, counts.sum() / n_bins)
    stat, p = chisquare(counts, expected)
    assert p > 1e-3, f"chi-square p={p} (stat={stat}) -- marginal not ~uniform"


# ---------------------------------------------------------------------------
# c. sample-axis autocorrelation (checked precisely on the Gaussian scale)
# ---------------------------------------------------------------------------

def test_sample_axis_autocorrelation():
    rho = 0.9
    gen = torch.Generator().manual_seed(2)
    cfg = NoiseConfig(rho=rho)
    S = 8000
    r = make_noise((S, 1, 1), gen, cfg).squeeze(-1).squeeze(-1)
    u = ((r + 1.0) / 2.0).clamp(1e-9, 1 - 1e-9)
    g = torch.special.ndtri(u.to(torch.float64)).numpy()
    lag1 = np.corrcoef(g[:-1], g[1:])[0, 1]
    assert abs(lag1 - rho) < 0.05, f"lag-1 corr on Gaussian scale = {lag1}, expected ~{rho}"

    # sanity: uncorrelated case should show ~0 lag-1 correlation.
    gen0 = torch.Generator().manual_seed(3)
    r0 = make_noise((S, 1, 1), gen0, NoiseConfig(rho=0.0)).squeeze(-1).squeeze(-1)
    u0 = ((r0 + 1.0) / 2.0).clamp(1e-9, 1 - 1e-9)
    g0 = torch.special.ndtri(u0.to(torch.float64)).numpy()
    lag1_0 = np.corrcoef(g0[:-1], g0[1:])[0, 1]
    assert abs(lag1_0) < 0.05, f"rho=0 lag-1 corr = {lag1_0}, expected ~0"


# ---------------------------------------------------------------------------
# d. rho=1 shares one draw across S; shared_across_neurons shares across H
# ---------------------------------------------------------------------------

def test_rho_one_shares_across_samples():
    gen = torch.Generator().manual_seed(4)
    r = make_noise((10, 4, 4), gen, NoiseConfig(rho=1.0))
    for s in range(1, 10):
        assert torch.allclose(r[s], r[0], atol=1e-5)


def test_shared_across_neurons():
    gen = torch.Generator().manual_seed(5)
    r = make_noise((10, 4, 6), gen, NoiseConfig(shared_across_neurons=True))
    # identical across H for a given (s, b)
    for h in range(1, 6):
        assert torch.allclose(r[:, :, h], r[:, :, 0], atol=1e-6)
    # varying across S (not degenerate)
    assert r[:, :, 0].std().item() > 0.1


def test_rho_and_shared_mutually_exclusive():
    gen = torch.Generator().manual_seed(6)
    cfg = NoiseConfig(rho=0.5, shared_across_neurons=True)
    try:
        make_noise((4, 2, 2), gen, cfg)
        assert False, "expected ValueError for rho + shared_across_neurons"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# e. bitdepth quantization
# ---------------------------------------------------------------------------

def test_bitdepth_quantization():
    k = 3
    levels = 2 ** k
    gen = torch.Generator().manual_seed(7)
    r = make_noise((2000, 1, 1), gen, NoiseConfig(bitdepth=k)).flatten()
    expected_centers = (torch.arange(levels) + 0.5) / levels * 2.0 - 1.0
    unique_vals = torch.unique(r)
    assert len(unique_vals) == levels, f"got {len(unique_vals)} unique levels, want {levels}"
    for v in unique_vals:
        assert torch.min((expected_centers - v).abs()) < 1e-5

    # direct check of the quantize() function against pneuron/noise.py's formula
    r_raw = torch.linspace(-1.0, 1.0 - 1e-6, 100)
    q = quantize(r_raw, k)
    idx = torch.clamp(torch.floor((r_raw + 1.0) / 2.0 * levels), 0, levels - 1)
    expected = (idx + 0.5) / levels * 2.0 - 1.0
    assert torch.allclose(q, expected)


# ---------------------------------------------------------------------------
# f. determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed_same_logits():
    weights = _tiny_weights([4, 6, 3], seed=100)
    X = torch.randn(5, 4)
    cfg = NoiseConfig()
    out1 = pdnn_forward(weights, X, S=16, cfg=cfg, seed=42)
    out2 = pdnn_forward(weights, X, S=16, cfg=cfg, seed=42)
    assert torch.equal(out1, out2)


def test_determinism_different_seed_different_logits():
    weights = _tiny_weights([4, 6, 3], seed=100)
    X = torch.randn(5, 4)
    cfg = NoiseConfig()
    out1 = pdnn_forward(weights, X, S=16, cfg=cfg, seed=42)
    out2 = pdnn_forward(weights, X, S=16, cfg=cfg, seed=43)
    assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# g. convergence to the deterministic net as S grows
# ---------------------------------------------------------------------------

def test_convergence_to_deterministic_net():
    weights = _tiny_weights([3, 5, 2], seed=200)
    X = torch.randn(20, 3) * 0.8
    ref = _deterministic_forward(weights, X)
    avg_logits = pdnn_forward(weights, X, S=512, cfg=NoiseConfig(), seed=999)
    max_err = (avg_logits - ref).abs().max().item()
    assert max_err < 0.25, f"max |avg_logits - deterministic| = {max_err}"


def test_fcs_model_path_matches_weights_path():
    """The .fcs duck-typed path and the explicit weight-list path must agree."""
    net = _TinyFcsNet([4, 5, 3], seed=300)
    weights = [(fc.weight.detach(), fc.bias.detach()) for fc in net.fcs]
    X = torch.randn(6, 4)
    out_model = pdnn_forward(net, X, S=8, cfg=NoiseConfig(), seed=1)
    out_weights = pdnn_forward(weights, X, S=8, cfg=NoiseConfig(), seed=1)
    assert torch.equal(out_model, out_weights)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_accuracy_and_probs():
    weights = _tiny_weights([4, 6, 3], seed=400)
    X = torch.randn(30, 4)
    ref = _deterministic_forward(weights, X)
    y = ref.argmax(dim=1)
    acc, probs = evaluate(weights, (X, y), S=256, cfg=NoiseConfig(), seed=5, return_probs=True)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.8, f"accuracy vs the net's own argmax should be high with ideal noise, got {acc}"
    assert probs.shape == (30, 3)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(30), atol=1e-4)


if __name__ == "__main__":
    import sys

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")
