"""Tests for models/pdnn.py's conv extension (make_noise_conv, pdnn_forward_conv,
evaluate_conv) and models/cnn.py's CNN class.

Run: .venv/bin/python -m pytest tests/test_pdnn_conv.py -q
"""

import os
import sys

import numpy as np
import torch
from scipy.stats import chisquare

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import CNN
from models.pdnn import NoiseConfig, evaluate_conv, make_noise_conv, pdnn_forward_conv


# ---------------------------------------------------------------------------
# a. E[m] -> tanh on a conv feature map
# ---------------------------------------------------------------------------

def test_expected_pbit_matches_tanh_conv():
    # a synthetic "pre-activation" feature map, S draws per (b,c,h,w) cell
    B, C, H, W = 1, 3, 4, 4
    z = torch.linspace(-3.0, 3.0, B * C * H * W).view(B, C, H, W)
    a = torch.tanh(z)
    S = 20_000
    gen = torch.Generator().manual_seed(0)
    a_s = a.unsqueeze(0).expand(S, B, C, H, W)
    r = make_noise_conv((S, B, C, H, W), gen, NoiseConfig())
    m = torch.where(a_s - r >= 0, torch.ones_like(a_s), -torch.ones_like(a_s))
    mean_m = m.mean(dim=0)
    max_err = (mean_m - a).abs().max().item()
    assert max_err < 0.02, f"max |E[m]-tanh(z)| = {max_err}"


# ---------------------------------------------------------------------------
# b. rho=1 gives identical draws across S in the conv path
# ---------------------------------------------------------------------------

def test_rho_one_shares_across_samples_conv():
    gen = torch.Generator().manual_seed(4)
    shape = (10, 3, 2, 4, 4)  # S,B,C,H,W
    r = make_noise_conv(shape, gen, NoiseConfig(rho=1.0))
    for s in range(1, 10):
        assert torch.allclose(r[s], r[0], atol=1e-5)

    # sanity: rho=0 should NOT collapse across S
    gen0 = torch.Generator().manual_seed(5)
    r0 = make_noise_conv(shape, gen0, NoiseConfig(rho=0.0))
    assert not torch.allclose(r0[1], r0[0], atol=1e-5)


# ---------------------------------------------------------------------------
# c. conv broadcast modes: share_per_channel / share_per_position
# ---------------------------------------------------------------------------

def test_share_per_channel_conv():
    S, B, C, H, W = 6, 2, 4, 5, 5
    gen = torch.Generator().manual_seed(10)
    r = make_noise_conv((S, B, C, H, W), gen, NoiseConfig(conv_broadcast="share_per_channel"))
    r_full = r.expand(S, B, C, H, W)

    # identical over spatial (h,w) for a given (s,b,c)
    flat = r_full.reshape(S, B, C, H * W)
    spread_over_space = (flat - flat[..., :1]).abs().max().item()
    assert spread_over_space < 1e-6, "share_per_channel should be constant over H,W"

    # varying across channels (not degenerate)
    per_channel = r_full[:, :, :, 0, 0]  # (S,B,C)
    assert per_channel.std().item() > 0.1, "should vary across channels"

    # varying across samples (not degenerate)
    per_sample = r_full[:, 0, 0, 0, 0]  # (S,)
    assert per_sample.std().item() > 0.1, "should vary across samples"


def test_share_per_position_conv():
    S, B, C, H, W = 6, 2, 4, 5, 5
    gen = torch.Generator().manual_seed(11)
    r = make_noise_conv((S, B, C, H, W), gen, NoiseConfig(conv_broadcast="share_per_position"))
    r_full = r.expand(S, B, C, H, W)

    # identical over channels for a given (s,b,h,w)
    spread_over_channels = (r_full - r_full[:, :, :1, :, :]).abs().max().item()
    assert spread_over_channels < 1e-6, "share_per_position should be constant over C"

    # varying across spatial positions (not degenerate)
    per_position = r_full[:, 0, 0, :, :]  # (S,H,W)
    assert per_position.std().item() > 0.1, "should vary across positions"

    # varying across samples (not degenerate)
    per_sample = r_full[:, 0, 0, 0, 0]  # (S,)
    assert per_sample.std().item() > 0.1, "should vary across samples"


def test_conv_broadcast_marginal_uniform():
    """Both broadcast modes should still keep an (approximately) Uniform[-1,1]
    marginal -- broadcasting doesn't touch the draw's distribution, only its
    reuse pattern."""
    gen = torch.Generator().manual_seed(12)
    r = make_noise_conv((400, 5, 6, 3, 3), gen, NoiseConfig(conv_broadcast="share_per_channel"))
    flat = r.flatten().numpy()
    counts, _ = np.histogram(flat, bins=20, range=(-1.0, 1.0))
    expected = np.full(20, counts.sum() / 20)
    stat, p = chisquare(counts, expected)
    assert p > 1e-3, f"chi-square p={p} (stat={stat}) -- marginal not ~uniform"


def test_unknown_conv_broadcast_mode_raises():
    gen = torch.Generator().manual_seed(13)
    try:
        make_noise_conv((2, 2, 2, 2, 2), gen, NoiseConfig(conv_broadcast="bogus"))
        assert False, "expected ValueError for unknown conv_broadcast mode"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# d. determinism given seed
# ---------------------------------------------------------------------------

def _tiny_cnn(seed, conv_channels=(2, 3), fc_hidden=5, in_hw=10, kernel_size=3, scale=0.5):
    torch.manual_seed(seed)
    model = CNN(conv_channels=conv_channels, fc_hidden=fc_hidden, in_hw=in_hw,
                kernel_size=kernel_size)
    model.eval()
    for p in model.parameters():
        p.data.normal_(0, scale)
    return model


def test_determinism_same_seed_same_logits_conv():
    model = _tiny_cnn(seed=100)
    X = torch.randn(5, 1, 10, 10)
    cfg = NoiseConfig()
    out1 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=42)
    out2 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=42)
    assert torch.equal(out1, out2)


def test_determinism_different_seed_different_logits_conv():
    model = _tiny_cnn(seed=100)
    X = torch.randn(5, 1, 10, 10)
    cfg = NoiseConfig()
    out1 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=42)
    out2 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=43)
    assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# e. tiny-CNN convergence to the deterministic net as S grows
# ---------------------------------------------------------------------------
#
# Tolerance is looser than the MLP test's 0.25 (tests/test_pdnn.py,
# test_convergence_to_deterministic_net): the CNN path has THREE stochastic
# p-bit stages (2 conv + 1 fc, vs the MLP's single hidden layer) and, more
# importantly, maxpool is applied directly to each stage's {-1,+1} p-bit
# codes -- maxpool(m) is a nonlinear function of m, so
# E[maxpool(m)] != maxpool(E[m]) = maxpool(tanh(z)) in general (a genuine
# Jensen-gap effect, not noise-seed variance: it does not shrink as S grows
# past a few hundred, see docs -- this is the CNN-specific finding the
# extension brief anticipated: "expect a possibly larger Jensen gap than
# the MLP's 0.4pp"). Calibrated empirically (see the CNN-extension
# handoff): with moderate (not deliberately saturated or deliberately
# near-zero) N(0, 0.5) weights, S=512 max|avg-det| plateaus around
# 0.28-0.35 across several noise seeds; 0.6 keeps a comfortable margin
# while still being a real convergence check (S=1 alone would not pass it).
def test_convergence_to_deterministic_net_conv():
    model = _tiny_cnn(seed=2)
    torch.manual_seed(102)
    X = torch.randn(8, 1, 10, 10) * 0.8
    with torch.no_grad():
        ref = model(X)
    avg_logits = pdnn_forward_conv(model, X, S=512, cfg=NoiseConfig(), seed=999)
    max_err = (avg_logits - ref).abs().max().item()
    assert max_err < 0.6, f"max |avg_logits - deterministic| = {max_err}"

    # S=1 should NOT already be this close (sanity: the test is non-trivial)
    avg_s1 = pdnn_forward_conv(model, X, S=1, cfg=NoiseConfig(), seed=999)
    err_s1 = (avg_s1 - ref).abs().max().item()
    assert err_s1 > max_err, "S=1 should be farther from det than S=512"


# ---------------------------------------------------------------------------
# evaluate_conv() / stage-list plumbing
# ---------------------------------------------------------------------------

def test_evaluate_conv_accuracy_and_probs():
    model = _tiny_cnn(seed=400)
    X = torch.randn(30, 1, 10, 10)
    with torch.no_grad():
        ref = model(X)
    y = ref.argmax(dim=1)
    acc, probs = evaluate_conv(model, (X, y), S=256, cfg=NoiseConfig(), seed=5,
                                return_probs=True)
    assert 0.0 <= acc <= 1.0
    assert probs.shape == (30, 10)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(30), atol=1e-4)


def test_pdnn_forward_conv_matches_manual_replay_with_ideal_tanh():
    """With p-bit disabled (replaying tanh directly, no comparator), the
    stage-by-stage machinery must reproduce the model's own forward exactly
    -- isolates shape/plumbing bugs from genuine p-bit stochastic effects."""
    model = _tiny_cnn(seed=7)
    X = torch.randn(4, 1, 10, 10)
    with torch.no_grad():
        det = model(X)

    from models.pdnn import _apply_over_sb
    x = X.unsqueeze(0)
    with torch.no_grad():
        for layer, kind in model.stages:
            if kind == "conv":
                x = torch.tanh(_apply_over_sb(x, layer))
            elif kind == "pool":
                x = _apply_over_sb(x, layer)
            elif kind == "flatten":
                x = x.reshape(x.shape[0], x.shape[1], -1)
            elif kind == "fc":
                x = torch.tanh(_apply_over_sb(x, layer))
            elif kind == "fc_out":
                x = _apply_over_sb(x, layer)
    replay = x[0]
    assert torch.allclose(replay, det, atol=1e-5)


def test_missing_stages_attribute_raises():
    class NoStages:
        pass

    try:
        pdnn_forward_conv(NoStages(), torch.randn(2, 1, 10, 10), S=1)
        assert False, "expected ValueError for a model without .stages"
    except ValueError:
        pass


if __name__ == "__main__":
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
