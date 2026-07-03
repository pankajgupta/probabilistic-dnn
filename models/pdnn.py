"""p-DNN: wrap a trained tanh-MLP for stochastic p-bit inference.

Each hidden tanh(z) is replaced by a p-bit comparator m = sign(tanh(z) - r),
r ~ Uniform[-1,1] drawn fresh every stochastic pass (see pneuron/neuron.py for
the numpy single-neuron version of the same primitive). S stochastic passes
are run per input and the *logits* are averaged over S; since E[m] = tanh(z)
for ideal r, averaged logits converge to the deterministic net's logits as S
grows (exactly, for a single p-bit hidden layer; see docs/HANDOFF.md section 2).

This module is weight-only: it does not import models/net.py. It accepts
either an object exposing `.fcs` (a ModuleList of nn.Linear, tanh between
them, linear output -- the convention documented in models/net.py) or an
explicit list of (weight, bias) pairs / nn.Linear-like objects, one per
layer, in order, with the last entry being the (deterministic, linear)
output layer.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Composable noise defects for r, the p-bit comparator's random draw.

    bitdepth: quantize r to 2**bitdepth level centers on [-1,1] (None = off).
    bias: shift r by this amount, then clip back to [-1,1] (0.0 = off).
    rho: AR(1) correlation of r along the *sample* axis (dim 0 of the
        (S,B,H) noise tensor), via a Gaussian copula so the marginal stays
        exactly Uniform[-1,1]. rho=0 is iid; rho=1 makes every sample share
        one draw. Mutually exclusive with shared_across_neurons.
    shared_across_neurons: if True, the same r is used by all H neurons in
        a layer for a given (sample, batch item), but iid across samples and
        batch items. An alternative correlation axis to rho. MLP-only (the
        conv analogue is conv_broadcast, below).
    uniform_source: optional callable(n) -> array-like of n uniforms in
        [0,1), e.g. a hardware-like LFSR generator. If given, it replaces
        the default torch.Generator draw everywhere r is needed (including
        as the base variate for the AR(1) copula).
    conv_broadcast: broadcast mode for conv-stage noise, shape (S,B,C,H,W)
        (see make_noise_conv). One of:
          None (default): fully independent draw at every (s,b,c,h,w).
          "share_per_channel": one draw per (s,b,c), broadcast over h,w --
              models one shared RNG per feature-map channel in hardware.
          "share_per_position": one draw per (s,b,h,w), broadcast over c --
              models one shared RNG per spatial position, reused by every
              channel's comparator at that position.
        Ignored by the MLP path (make_noise/pdnn_forward). Composable with
        rho (rho decorrelates along S; conv_broadcast reduces independence
        along C and/or H,W -- orthogonal axes).
    binarize_after_pool: conv-path stage order (see pdnn_forward_conv). If
        False (default, the naive order): each "conv" stage binarizes its
        own tanh immediately (m = sign(tanh(z) - r)), and the following
        "pool" stage maxpools the *p-bit codes* m -- this is biased,
        E[maxpool(m)] != maxpool(E[m]) = maxpool(tanh(z)), because maxpool
        is nonlinear and does not commute with expectation (Jensen gap);
        near-chance accuracy results (see docs/HANDOFF.md, CNN-A's finding).
        If True: a "conv" stage immediately followed by a "pool" stage is
        replayed as z = conv(x), a = tanh(z), pooled_a = pool(a) (maxpool
        applied to the CONTINUOUS activation), THEN m = sign(pooled_a - r)
        -- the comparator is deferred until after pooling, so
        E[m] = pooled_a = maxpool(tanh(z)) exactly, matching what the
        deterministic net itself computes; the S->inf limit is unbiased
        again. FC stages (no pooling) are unaffected either way. Ignored by
        the MLP path (pdnn_forward has no pool stages).
    """

    bitdepth: Optional[int] = None
    bias: float = 0.0
    rho: float = 0.0
    shared_across_neurons: bool = False
    uniform_source: Optional[Callable[[int], np.ndarray]] = None
    conv_broadcast: Optional[str] = None
    binarize_after_pool: bool = False


_UNIT_EPS = 1e-12  # clamp for ndtri so 0/1 don't map to +-inf


def quantize(r, k):
    """Quantize r in [-1,1] to 2**k level centers (matches pneuron/noise.py)."""
    levels = 2 ** k
    idx = torch.clamp(torch.floor((r + 1.0) / 2.0 * levels), 0, levels - 1)
    return (idx + 0.5) / levels * 2.0 - 1.0


def _draw_uniform(shape, generator, uniform_source):
    """Base uniforms in [0,1) as a float64 tensor of the given shape."""
    n = 1
    for d in shape:
        n *= d
    if uniform_source is not None:
        u = np.asarray(uniform_source(n), dtype=np.float64).reshape(shape)
        return torch.from_numpy(u)
    if generator is not None:
        return torch.rand(shape, generator=generator, dtype=torch.float64)
    return torch.rand(shape, dtype=torch.float64)


def _ar1_copula_uniform(shape, rho, generator, uniform_source):
    """Uniform[-1,1] r with AR(1) correlation rho along dim 0 (sample axis).

    Gaussian copula: u -> g = Phi^-1(u) (iid N(0,1) base variates), then
    g_0 = eps_0, g_s = rho*g_{s-1} + sqrt(1-rho^2)*eps_s, then r = 2*Phi(g)-1.
    This keeps the marginal exactly Uniform[-1,1] for any rho; rho=0 reduces
    algebraically to g=eps (iid), rho=1 collapses the chain to a single g_0
    shared by every sample.
    """
    S = shape[0]
    u = _draw_uniform(shape, generator, uniform_source)
    u = u.clamp(_UNIT_EPS, 1.0 - _UNIT_EPS)
    eps = torch.special.ndtri(u)
    g = torch.empty_like(eps)
    g[0] = eps[0]
    c = math.sqrt(max(0.0, 1.0 - rho * rho))
    for s in range(1, S):
        g[s] = rho * g[s - 1] + c * eps[s]
    return 2.0 * torch.special.ndtr(g) - 1.0


def make_noise(shape, generator=None, cfg=None):
    """Draw the comparator noise r, shape (S, B, H), composing cfg's defects.

    generator: torch.Generator for the default uniform source (ignored if
        cfg.uniform_source is set). None uses the global torch RNG.
    """
    cfg = cfg or NoiseConfig()
    if cfg.rho and cfg.shared_across_neurons:
        raise ValueError("rho and shared_across_neurons are mutually exclusive")

    S, B, H = shape
    if cfg.shared_across_neurons:
        u = _draw_uniform((S, B), generator, cfg.uniform_source)
        r = (2.0 * u - 1.0).unsqueeze(-1).expand(S, B, H)
    elif cfg.rho:
        r = _ar1_copula_uniform((S, B, H), cfg.rho, generator, cfg.uniform_source)
    else:
        u = _draw_uniform((S, B, H), generator, cfg.uniform_source)
        r = 2.0 * u - 1.0

    if cfg.bitdepth is not None:
        r = quantize(r, cfg.bitdepth)
    if cfg.bias:
        r = torch.clamp(r + cfg.bias, -1.0, 1.0)
    return r.to(torch.float32)


def make_noise_conv(shape, generator=None, cfg=None):
    """Draw the comparator noise r for a conv feature map, shape (S,B,C,H,W).

    Composes the same defects as make_noise (bitdepth, bias, rho along the
    sample axis -- reuses _ar1_copula_uniform / _draw_uniform directly, no
    duplicated math) plus two conv-only broadcast modes selected by
    cfg.conv_broadcast (see NoiseConfig's docstring):
      None                  -> draw at the full (S,B,C,H,W).
      "share_per_channel"   -> draw at (S,B,C,1,1); relies on ordinary
                                 tensor broadcasting to cover H,W when the
                                 caller does `a - r`.
      "share_per_position"  -> draw at (S,B,1,H,W); broadcasts over C.
    Returning the reduced-shape tensor (rather than an expanded copy) both
    avoids allocating S*B*C*H*W elements for the "shared" cases and is the
    most direct way to make the sharing explicit: any two elements that
    map to the same reduced-shape entry are, by construction, the same
    draw. cfg.shared_across_neurons is MLP-only and is ignored here.
    """
    cfg = cfg or NoiseConfig()
    S, B, C, H, W = shape
    mode = cfg.conv_broadcast
    if mode is None:
        draw_shape = (S, B, C, H, W)
    elif mode == "share_per_channel":
        draw_shape = (S, B, C, 1, 1)
    elif mode == "share_per_position":
        draw_shape = (S, B, 1, H, W)
    else:
        raise ValueError(f"unknown conv_broadcast mode: {mode!r}")

    if cfg.rho:
        r = _ar1_copula_uniform(draw_shape, cfg.rho, generator, cfg.uniform_source)
    else:
        u = _draw_uniform(draw_shape, generator, cfg.uniform_source)
        r = 2.0 * u - 1.0

    if cfg.bitdepth is not None:
        r = quantize(r, cfg.bitdepth)
    if cfg.bias:
        r = torch.clamp(r + cfg.bias, -1.0, 1.0)
    return r.to(torch.float32)


# ---------------------------------------------------------------------------
# p-DNN forward / evaluate
# ---------------------------------------------------------------------------

_CHUNK_BUDGET_BYTES = 500 * 1024 * 1024  # target ceiling for a live (S,B,H) tensor
_CHUNK_SAFETY_FACTOR = 3  # a, r, m coexist per layer


def _get_layers(model_or_weights):
    """Return [(weight, bias), ...] tensors, hidden layers first, output last."""
    if hasattr(model_or_weights, "fcs"):
        return [(fc.weight.detach(), fc.bias.detach()) for fc in model_or_weights.fcs]
    layers = []
    for item in model_or_weights:
        if hasattr(item, "weight"):
            layers.append((item.weight.detach(), item.bias.detach()))
        else:
            W, b = item
            layers.append((torch.as_tensor(W, dtype=torch.float32),
                            torch.as_tensor(b, dtype=torch.float32)))
    return layers


def _batch_chunk_size(S, max_hidden, B):
    denom = max(1, S * max_hidden * 4 * _CHUNK_SAFETY_FACTOR)
    chunk = max(1, _CHUNK_BUDGET_BYTES // denom)
    return min(chunk, B)


def pdnn_forward(model_or_weights, X, S, cfg=None, seed=0, return_samples=False):
    """Run S stochastic p-bit passes over X and average the logits.

    Hidden layers: z = x @ W.T + b, a = tanh(z), r = make_noise(...),
    m = sign(a - r) in {-1,+1} feeds the next layer. Output layer is a plain
    linear layer applied to the last hidden layer's {-1,+1} codes (no p-bit).

    Deterministic given seed: a single torch.Generator seeded once is
    consumed sequentially (chunk by chunk, layer by layer), so every sample,
    layer and batch chunk draws independent noise -- there is no reused
    state across the "independent" (rho=0) baseline.

    Returns avg_logits (B, n_classes); with return_samples also returns the
    per-sample logits (S, B, n_classes).
    """
    layers = _get_layers(model_or_weights)
    hidden_layers, (W_out, b_out) = layers[:-1], layers[-1]
    cfg = cfg or NoiseConfig()

    X = torch.as_tensor(X, dtype=torch.float32)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    B, in_dim = X.shape
    n_classes = W_out.shape[0]

    gen = torch.Generator().manual_seed(seed)
    max_hidden = max((W.shape[0] for W, _ in hidden_layers), default=1)
    chunk = _batch_chunk_size(S, max_hidden, B)

    sample_logits = torch.empty(S, B, n_classes)
    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        b = end - start
        x = X[start:end].unsqueeze(0).expand(S, b, in_dim)
        for W, bias in hidden_layers:
            z = x @ W.T + bias
            a = torch.tanh(z)
            r = make_noise((S, b, W.shape[0]), gen, cfg)
            x = torch.where(a - r >= 0, torch.ones_like(a), -torch.ones_like(a))
        logits = x @ W_out.T + b_out
        sample_logits[:, start:end, :] = logits

    avg_logits = sample_logits.mean(dim=0)
    if return_samples:
        return avg_logits, sample_logits
    return avg_logits


def evaluate(model_or_weights, dataset_tensors, S, cfg=None, seed=0, return_probs=False):
    """Accuracy of the p-DNN on (X, y), averaging logits over S passes.

    With return_probs=True, also returns the per-example, per-sample-averaged
    softmax (mean over S of softmax(sample_logits), not softmax of the mean
    logit) -- the quantity ECE calibration should use.
    """
    X, y = dataset_tensors
    avg_logits, sample_logits = pdnn_forward(model_or_weights, X, S, cfg, seed,
                                              return_samples=True)
    y = torch.as_tensor(y)
    preds = avg_logits.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    if return_probs:
        probs = torch.softmax(sample_logits, dim=-1).mean(dim=0)
        return acc, probs
    return acc


# ---------------------------------------------------------------------------
# p-DNN forward / evaluate -- conv path (models/cnn.py's CNN class)
# ---------------------------------------------------------------------------
#
# Same primitive as the MLP path above (m = sign(tanh(z) - r), logits
# averaged over S), replayed stage-by-stage over a model's `.stages` list
# (see models/cnn.py): every "conv" and "fc" stage's tanh is replaced by the
# p-bit comparator; "pool" and the final "fc_out" stage are deterministic
# and operate directly on the {-1,+1} p-bit codes, exactly mirroring how
# pdnn_forward's linear output layer consumes the last hidden layer's codes.

_CONV_CHUNK_SAFETY_FACTOR = 4  # conv activations, r, m coexist per stage (+ headroom)


def _apply_over_sb(x, module):
    """Reshape (S,B,*rest) -> (S*B,*rest), apply an nn.Module, reshape back."""
    S, B = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    flat = x.reshape(S * B, *rest)
    out = module(flat)
    return out.view(S, B, *out.shape[1:])


def _get_conv_stages(model):
    """Return model.stages: an ordered [(module_or_'flatten', kind), ...]
    list, kind in {"conv","pool","flatten","fc","fc_out"} (see models/cnn.py's
    CNN.stages). This module is weight-only for the MLP path but the conv
    path needs actual nn.Conv2d/nn.MaxPool2d/nn.Linear modules (their forward
    logic, e.g. padding/stride, isn't just a matmul), so it takes a live
    model rather than duck-typed weight tuples.
    """
    if not hasattr(model, "stages"):
        raise ValueError(
            "pdnn_forward_conv needs a model exposing `.stages` "
            "(see models/cnn.py's CNN class)"
        )
    return model.stages


def _trace_max_elems_per_sample(stages, x1):
    """Run a shape-only trace (batch size 1, S folded into batch) to find the
    largest per-(s,b) element count across all stochastic stages, and the
    final output width. Used to size the chunk so a live (S,b,...) tensor at
    any stage stays within budget; the trace itself is a single cheap
    B=1 forward pass, not part of the timed/noised computation.
    """
    with torch.no_grad():
        x = x1
        max_elems = x.shape[1:].numel() if x.dim() > 1 else 1
        for layer, kind in stages:
            if kind == "conv":
                x = layer(x)
                max_elems = max(max_elems, x.shape[1:].numel())
            elif kind == "pool":
                x = layer(x)
            elif kind == "flatten":
                x = x.flatten(1)
            elif kind == "fc":
                x = layer(x)
                max_elems = max(max_elems, x.shape[1:].numel())
            elif kind == "fc_out":
                x = layer(x)
            else:
                raise ValueError(f"unknown stage kind: {kind!r}")
    return max_elems, x.shape[-1]


def _batch_chunk_size_conv(S, max_elems_per_sample, B):
    denom = max(1, S * max_elems_per_sample * 4 * _CONV_CHUNK_SAFETY_FACTOR)
    chunk = max(1, _CHUNK_BUDGET_BYTES // denom)
    return min(chunk, B)


def pdnn_forward_conv(model, X, S, cfg=None, seed=0, return_samples=False):
    """Run S stochastic p-bit passes over a conv net and average the logits.

    model must expose `.stages` (see models/cnn.py's CNN class): an ordered
    list of (module, kind) pairs. For "conv"/"fc" stages: z = module(x),
    a = tanh(z), r = make_noise_conv(...) [conv] or make_noise(...) [fc],
    m = sign(a - r) in {-1,+1} feeds the next stage. "pool" stages apply
    their module directly (deterministic, operates on the {-1,+1} codes).
    "flatten" reshapes (S,B,C,H,W) -> (S,B,C*H*W). "fc_out" is the plain
    linear output stage (no p-bit), matching pdnn_forward's convention.

    This is the naive/default order (cfg.binarize_after_pool=False): pool
    sees the p-bit codes m, not the continuous tanh -- see NoiseConfig's
    docstring for why that biases E[maxpool(m)] away from maxpool(tanh(z)).
    When cfg.binarize_after_pool=True, a "conv" stage immediately followed
    by a "pool" stage is instead replayed as z = conv(x), a = tanh(z),
    pooled_a = pool(a), m = sign(pooled_a - r) -- the comparator is deferred
    until after pooling (pool-compatible order), consuming both stages in
    one loop step (see the (i, i+1) lookahead below). A "conv" stage NOT
    followed by "pool" (not the case in models/cnn.py's CNN today) falls
    back to binarizing immediately, same as the naive order.

    Deterministic given seed: one torch.Generator, consumed sequentially
    (chunk by chunk, stage by stage) -- same discipline as pdnn_forward.

    Batches are chunked (see _batch_chunk_size_conv) so a live (S,b,C,H,W)
    tensor at the widest conv stage stays under budget even at S=64.

    Returns avg_logits (B, n_classes); with return_samples also returns the
    per-sample logits (S, B, n_classes).
    """
    stages = _get_conv_stages(model)
    cfg = cfg or NoiseConfig()

    X = torch.as_tensor(X, dtype=torch.float32)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    if X.dim() == 2:
        side = int(round(X.shape[1] ** 0.5))
        X = X.view(X.shape[0], 1, side, side)
    elif X.dim() == 3:
        X = X.unsqueeze(1)
    B = X.shape[0]

    gen = torch.Generator().manual_seed(seed)

    max_elems, n_classes = _trace_max_elems_per_sample(stages, X[:1])
    chunk = _batch_chunk_size_conv(S, max_elems, B)

    sample_logits = torch.empty(S, B, n_classes)
    n_stages = len(stages)
    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        b = end - start
        x = X[start:end].unsqueeze(0).expand(S, b, *X.shape[1:])
        i = 0
        while i < n_stages:
            layer, kind = stages[i]
            if kind == "conv":
                z = _apply_over_sb(x, layer)
                a = torch.tanh(z)
                next_kind = stages[i + 1][1] if i + 1 < n_stages else None
                if cfg.binarize_after_pool and next_kind == "pool":
                    pool_layer = stages[i + 1][0]
                    pooled_a = _apply_over_sb(a, pool_layer)
                    r = make_noise_conv(pooled_a.shape, gen, cfg)
                    x = torch.where(pooled_a - r >= 0, torch.ones_like(pooled_a),
                                     -torch.ones_like(pooled_a))
                    i += 2
                    continue
                r = make_noise_conv(a.shape, gen, cfg)
                x = torch.where(a - r >= 0, torch.ones_like(a), -torch.ones_like(a))
            elif kind == "pool":
                x = _apply_over_sb(x, layer)
            elif kind == "flatten":
                x = x.reshape(x.shape[0], x.shape[1], -1)
            elif kind == "fc":
                z = _apply_over_sb(x, layer)
                a = torch.tanh(z)
                r = make_noise(a.shape, gen, cfg)
                x = torch.where(a - r >= 0, torch.ones_like(a), -torch.ones_like(a))
            elif kind == "fc_out":
                x = _apply_over_sb(x, layer)
            else:
                raise ValueError(f"unknown stage kind: {kind!r}")
            i += 1
        sample_logits[:, start:end, :] = x

    avg_logits = sample_logits.mean(dim=0)
    if return_samples:
        return avg_logits, sample_logits
    return avg_logits


def evaluate_conv(model, dataset_tensors, S, cfg=None, seed=0, return_probs=False):
    """Accuracy of the conv p-DNN on (X, y), averaging logits over S passes.

    Same contract as evaluate(), for the conv path. With return_probs=True,
    also returns the per-example, per-sample-averaged softmax (mean over S
    of softmax(sample_logits)) -- the quantity ECE calibration should use.
    """
    X, y = dataset_tensors
    avg_logits, sample_logits = pdnn_forward_conv(model, X, S, cfg, seed,
                                                    return_samples=True)
    y = torch.as_tensor(y)
    preds = avg_logits.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    if return_probs:
        probs = torch.softmax(sample_logits, dim=-1).mean(dim=0)
        return acc, probs
    return acc
