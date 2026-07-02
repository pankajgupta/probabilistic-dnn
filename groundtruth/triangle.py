"""Frustrated-triangle ground truth: exact Boltzmann target vs Gibbs sampling.

Three p-bits, J12=J13=J23=-1, h=0, E(m) = m1*m2 + m1*m3 + m2*m3, pi ~ exp(-E).
Exact: each of the 6 frustrated states = 1/(2*(3+e^-4)) ~ 0.16565,
       each of the 2 aligned states  = e^-4/(2*(3+e^-4)) ~ 0.00303.
Validation gate: with a good RNG, TV(empirical, exact) -> 0.
"""

import numpy as np

J = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=float)


def exact_dist():
    """Exact target over the 8 states, keyed by state index sum(bit_i * 2^i)."""
    p = np.empty(8)
    for s in range(8):
        m = np.array([1 if (s >> i) & 1 else -1 for i in range(3)])
        E = m[0] * m[1] + m[0] * m[2] + m[1] * m[2]
        p[s] = np.exp(-E)
    return p / p.sum()


def gibbs(noise, sweeps=100_000, burn=2_000, seed_state=0):
    """Sequential Gibbs with p-bit updates m_i = sign(tanh(I_i) - r).

    `noise` supplies every r, so any defect there propagates to the sampler.
    Returns empirical distribution over the 8 states.
    """
    rng = np.random.default_rng(seed_state)
    m = rng.choice([-1.0, 1.0], size=3)
    counts = np.zeros(8)
    for t in range(sweeps + burn):
        for i in range(3):
            I = J[i] @ m
            r = float(noise.sample(()))
            m[i] = 1.0 if np.tanh(I) - r >= 0 else -1.0
        if t >= burn:
            s = int((m[0] > 0) * 1 + (m[1] > 0) * 2 + (m[2] > 0) * 4)
            counts[s] += 1
    return counts / counts.sum()


def tv(p, q):
    """Total-variation distance."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()
