"""p-DNN: wrap a trained tanh-MLP for stochastic p-bit inference.

Each hidden tanh(z) is replaced by a p-bit comparator m = sign(tanh(z) - r),
r ~ Uniform[-1,1] drawn fresh every stochastic pass (see pneuron/neuron.py for
the numpy single-neuron version of the same primitive). S stochastic passes
are run per input and the *logits* are averaged over S; since E[m] = tanh(z)
for ideal r, averaged logits converge to the deterministic net's logits as S
grows (exactly, for a single p-bit hidden layer; see HANDOFF.md section 2).

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
        batch items. An alternative correlation axis to rho.
    uniform_source: optional callable(n) -> array-like of n uniforms in
        [0,1), e.g. a hardware-like LFSR generator. If given, it replaces
        the default torch.Generator draw everywhere r is needed (including
        as the base variate for the AR(1) copula).
    """

    bitdepth: Optional[int] = None
    bias: float = 0.0
    rho: float = 0.0
    shared_across_neurons: bool = False
    uniform_source: Optional[Callable[[int], np.ndarray]] = None


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
