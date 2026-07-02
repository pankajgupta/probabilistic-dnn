"""The p-neuron primitive: m = sign(f(I) - r), r drawn fresh every pass.

The three sub-operations (pre-activation f, RNG draw, compare) are explicit so
a defective generator can be substituted at the comparator.
"""

import numpy as np


def p_sample(I, noise, pre="tanh"):
    """One stochastic forward pass. Returns m in {-1, +1}, same shape as I."""
    I = np.asarray(I, dtype=float)
    a = np.tanh(I) if pre == "tanh" else I
    r = noise.sample(I.shape)
    return np.where(a - r >= 0, 1.0, -1.0)


def firing_prob(I, noise, n_draws=20000, pre="tanh"):
    """Empirical P(m=+1) at each input I, from n_draws fresh passes each."""
    I = np.asarray(I, dtype=float)
    hits = np.zeros_like(I)
    for _ in range(n_draws):
        hits += p_sample(I, noise, pre=pre) > 0
    return hits / n_draws
