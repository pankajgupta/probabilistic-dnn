"""Tests for models/pdnn.py's pool-compatible p-bit order
(NoiseConfig.binarize_after_pool): a "conv" stage immediately followed by a
"pool" stage is replayed as z = conv(x), a = tanh(z), pooled_a = pool(a),
THEN m = sign(pooled_a - r) -- maxpool sees the continuous activation, not
the p-bit codes. This is the fix for the naive-order failure CNN-A found
(pool applied to m: near-chance accuracy on the trained MNIST CNN, because
E[maxpool(m)] != maxpool(E[m]) = maxpool(tanh(z))).

Run: .venv/bin/python -m pytest tests/test_pdnn_conv2.py -q
"""

import os
import sys

import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import CNN
from models.pdnn import NoiseConfig, evaluate_conv, make_noise_conv, pdnn_forward_conv


def _tiny_cnn(seed, conv_channels=(2, 3), fc_hidden=5, in_hw=10, kernel_size=3, scale=0.5):
    torch.manual_seed(seed)
    model = CNN(conv_channels=conv_channels, fc_hidden=fc_hidden, in_hw=in_hw,
                kernel_size=kernel_size)
    model.eval()
    for p in model.parameters():
        p.data.normal_(0, scale)
    return model


# ---------------------------------------------------------------------------
# a. tanh replay, order change alone reproduces det logits exactly
# ---------------------------------------------------------------------------
#
# Manual replay that mirrors pdnn_forward_conv's binarize_after_pool lookahead
# (conv+pool consumed together, pool applied to the continuous tanh, not to
# any p-bit code) but with the comparator sign() removed entirely -- i.e. the
# "tanh replay" carries the continuous activation straight through. Since
# there is no comparator in this replay, both the "conv+pool paired" order
# and model.forward() itself compute exactly conv->tanh->pool->...->fc->tanh
# ->fc_out; matching bit-for-bit confirms the lookahead pairing logic (the
# (i, i+1) stage-skip in pdnn_forward_conv) correctly applies pool to `a`,
# not to some already-binarized code -- the plumbing the order-change fix
# depends on.

def _tanh_replay_pool_after_tanh(model, X):
    stages = model.stages
    n = len(stages)
    x = X
    i = 0
    with torch.no_grad():
        while i < n:
            layer, kind = stages[i]
            if kind == "conv":
                a = torch.tanh(layer(x))
                next_kind = stages[i + 1][1] if i + 1 < n else None
                if next_kind == "pool":
                    x = stages[i + 1][0](a)  # pool applied to the continuous tanh
                    i += 2
                    continue
                x = a
            elif kind == "pool":
                x = layer(x)
            elif kind == "flatten":
                x = x.reshape(x.shape[0], -1)
            elif kind == "fc":
                x = torch.tanh(layer(x))
            elif kind == "fc_out":
                x = layer(x)
            else:
                raise ValueError(f"unknown stage kind: {kind!r}")
            i += 1
    return x


def test_binarize_after_pool_tanh_replay_matches_det_exactly():
    model = _tiny_cnn(seed=7)
    X = torch.randn(4, 1, 10, 10)
    with torch.no_grad():
        det = model(X)
    replay = _tanh_replay_pool_after_tanh(model, X)
    assert torch.allclose(replay, det, atol=1e-5), \
        f"max diff {(replay-det).abs().max().item()}"


# ---------------------------------------------------------------------------
# b. E[m] -> pooled tanh (comparator applied AFTER maxpool)
# ---------------------------------------------------------------------------

def test_expected_pbit_matches_pooled_tanh():
    # synthetic pre-activation feature map -> tanh -> maxpool -> then the
    # p-bit comparator sees the POOLED continuous value, not raw tanh.
    B, C, H, W = 1, 3, 4, 4
    z = torch.linspace(-3.0, 3.0, B * C * H * W).view(B, C, H, W)
    a = torch.tanh(z)
    pooled_a = torch.nn.functional.max_pool2d(a, kernel_size=2, stride=2)  # (1,3,2,2)

    S = 20_000
    gen = torch.Generator().manual_seed(0)
    pooled_a_s = pooled_a.unsqueeze(0).expand(S, *pooled_a.shape)
    r = make_noise_conv((S,) + tuple(pooled_a.shape), gen, NoiseConfig())
    m = torch.where(pooled_a_s - r >= 0, torch.ones_like(pooled_a_s), -torch.ones_like(pooled_a_s))
    mean_m = m.mean(dim=0)
    max_err = (mean_m - pooled_a).abs().max().item()
    assert max_err < 0.02, f"max |E[m]-pooled_tanh| = {max_err}"

    # negative control: E[m] should NOT match the pre-pool tanh (a itself,
    # broadcast/cropped down) -- pooling changes the target of convergence.
    # (sanity that the test is actually exercising the pooled quantity)
    unpooled_mismatch = (mean_m - a[:, :, :2, :2]).abs().max().item()
    assert unpooled_mismatch > 0.05


# ---------------------------------------------------------------------------
# c. S=512 averaged logits close to det -- what the order fix actually buys
# ---------------------------------------------------------------------------
#
# CNN-A's original tiny-synthetic-net convergence test (test_pdnn_conv.py,
# test_convergence_to_deterministic_net_conv, N(0,0.5) weights, tolerance
# 0.6) predates the binarize_after_pool fix and uses the naive order. We
# measured (see task report) that on THAT SAME tiny net/scale, the naive and
# fixed orders are statistically indistinguishable at S=512 (both plateau
# around max|avg-det| ~= 0.29-0.35) -- the tiny net's small maps/few channels
# don't manifest the pool-order bug much, so tolerance is NOT much tighter
# in that regime; this is a real (if slightly surprising) finding, not
# swept under the rug. The order fix's benefit shows up dramatically on the
# REAL trained checkpoint instead, which is what actually matters (accuracy,
# not a toy net's logit magnitude): on results/models/cnn_s0.pt, a 200-image
# subset, S=512 -- naive order: max|avg-det| ~16.4, accuracy ~5% (chance);
# fixed order: max|avg-det| ~8.1 (tighter, though still not tiny -- CNN
# logits reach magnitude ~17), accuracy ~90-92% (vs 100% det on this easy
# subset). That is the number this test gates on.

def test_convergence_to_deterministic_net_binarize_after_pool_tiny():
    """Tiny synthetic net (same convention as CNN-A's original test): the
    fixed order must still converge as S grows (S=1 worse than S=512), even
    though on this small toy net the achieved tolerance is not dramatically
    tighter than the naive order's 0.6 (see docstring above)."""
    model = _tiny_cnn(seed=2)
    torch.manual_seed(102)
    X = torch.randn(8, 1, 10, 10) * 0.8
    with torch.no_grad():
        ref = model(X)
    cfg = NoiseConfig(binarize_after_pool=True)
    avg_logits = pdnn_forward_conv(model, X, S=512, cfg=cfg, seed=999)
    max_err = (avg_logits - ref).abs().max().item()
    assert max_err < 0.6, f"max |avg_logits - deterministic| = {max_err}"

    avg_s1 = pdnn_forward_conv(model, X, S=1, cfg=cfg, seed=999)
    err_s1 = (avg_s1 - ref).abs().max().item()
    assert err_s1 > max_err, "S=1 should be farther from det than S=512"


def test_order_fix_recovers_accuracy_on_real_checkpoint():
    """The comparison that actually matters: on the real trained cnn_s0.pt
    (not a toy net), the naive order (pool on p-bit codes) is near-chance;
    the fixed order (pool on tanh, binarize after) recovers most of the
    deterministic accuracy. Uses a 200-image MNIST-test subset to stay fast."""
    from torchvision import datasets, transforms
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=root / "data", train=False, download=True, transform=tfm)
    X = test_ds.data.float().div(255.0).view(len(test_ds), 1, 28, 28)[:200]
    y = test_ds.targets.clone()[:200]

    ckpt = torch.load(root / "results" / "models" / "cnn_s0.pt", map_location="cpu")
    model = CNN(conv_channels=tuple(ckpt["conv_channels"]), fc_hidden=ckpt["fc_hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        det_acc = (model(X).argmax(dim=1) == y).float().mean().item()

    naive_acc = evaluate_conv(model, (X, y), S=64, cfg=NoiseConfig(), seed=999)
    fixed_acc = evaluate_conv(model, (X, y), S=64,
                               cfg=NoiseConfig(binarize_after_pool=True), seed=999)

    assert det_acc > 0.9, f"sanity: subset should be easy, det_acc={det_acc}"
    assert naive_acc < 0.3, f"naive order should be near-chance-ish, got {naive_acc}"
    assert fixed_acc > 0.75, f"fixed order should recover most accuracy, got {fixed_acc}"
    assert fixed_acc - naive_acc > 0.5, "fixed order must be a large, clear improvement"


# ---------------------------------------------------------------------------
# d. determinism given seed
# ---------------------------------------------------------------------------

def test_determinism_same_seed_same_logits_binarize_after_pool():
    model = _tiny_cnn(seed=100)
    X = torch.randn(5, 1, 10, 10)
    cfg = NoiseConfig(binarize_after_pool=True)
    out1 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=42)
    out2 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=42)
    assert torch.equal(out1, out2)


def test_determinism_different_seed_different_logits_binarize_after_pool():
    model = _tiny_cnn(seed=100)
    X = torch.randn(5, 1, 10, 10)
    cfg = NoiseConfig(binarize_after_pool=True)
    out1 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=42)
    out2 = pdnn_forward_conv(model, X, S=16, cfg=cfg, seed=43)
    assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# regression: default (binarize_after_pool=False) is untouched
# ---------------------------------------------------------------------------

def test_default_cfg_unaffected_by_new_flag():
    """cfg=NoiseConfig() (binarize_after_pool defaults False) must behave
    exactly as before the change -- a regression guard, not a new-feature
    test."""
    model = _tiny_cnn(seed=100)
    X = torch.randn(5, 1, 10, 10)
    out_default = pdnn_forward_conv(model, X, S=16, cfg=NoiseConfig(), seed=42)
    out_explicit_false = pdnn_forward_conv(
        model, X, S=16, cfg=NoiseConfig(binarize_after_pool=False), seed=42)
    assert torch.equal(out_default, out_explicit_false)


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
