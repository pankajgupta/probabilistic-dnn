"""Noise sources for p-bit comparators, with controlled defects.

All sources emit r in [-1, 1] intended for the comparator m = sign(f(I) - r).
With ideal r ~ Uniform[-1,1] and f = tanh, P(m=+1) = (1 + tanh(I)) / 2.

Defect families (mini version):
  - bit-depth k: quantize r to 2^k levels on [-1, 1]
  - bias b: shift the mean of r (clipped to support)
  - correlation rho: AR(1) via a Gaussian copula, so the *marginal* stays
    Uniform[-1,1] and only serial independence degrades. This isolates the
    correlation variable (a plain Gaussian AR(1) would also change the
    activation shape, confounding H3).
"""

import numpy as np
from scipy.special import ndtr  # standard normal CDF


def quantize(r, k):
    """Quantize r in [-1,1] to 2^k level centers."""
    levels = 2 ** k
    idx = np.clip(np.floor((r + 1.0) / 2.0 * levels), 0, levels - 1)
    return (idx + 0.5) / levels * 2.0 - 1.0


class UniformNoise:
    """Ideal baseline: r ~ Uniform[-1,1] from PCG64, optionally quantized/biased."""

    def __init__(self, seed, k=None, bias=0.0):
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.bias = bias

    def sample(self, shape):
        r = self.rng.uniform(-1.0, 1.0, size=shape)
        if self.k is not None:
            r = quantize(r, self.k)
        if self.bias:
            r = np.clip(r + self.bias, -1.0, 1.0)
        return r


class AR1UniformNoise:
    """Serially correlated noise with exact Uniform[-1,1] marginal.

    Gaussian AR(1): g_t = rho*g_{t-1} + sqrt(1-rho^2)*eps, then r = 2*Phi(g) - 1.
    Successive calls to sample() continue the chain, so correlation spans draws.
    rho=0 reduces to the ideal independent source.
    """

    def __init__(self, seed, rho, k=None, bias=0.0):
        self.rng = np.random.default_rng(seed)
        self.rho = rho
        self.k = k
        self.bias = bias
        self._g = self.rng.standard_normal()

    def sample(self, shape):
        n = int(np.prod(shape)) if shape else 1
        eps = self.rng.standard_normal(n)
        g = np.empty(n)
        prev = self._g
        c = np.sqrt(1.0 - self.rho ** 2)
        for i in range(n):
            prev = self.rho * prev + c * eps[i]
            g[i] = prev
        self._g = prev
        r = 2.0 * ndtr(g) - 1.0
        if self.k is not None:
            r = quantize(r, self.k)
        if self.bias:
            r = np.clip(r + self.bias, -1.0, 1.0)
        return r.reshape(shape)
