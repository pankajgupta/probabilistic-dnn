"""RNG utilities: a bit-level Galois LFSR and statistical validation tests.

Every noise source we plug into the p-neuron comparator must be validated
BEFORE use (HANDOFF.md sec 3.3 / sec 7 step 2). This module provides:

  - GaloisLFSR: a hardware-motivated, deliberately-defective generator. Its
    n-bit register cycles through at most 2^n - 1 nonzero states before
    repeating exactly -- that hard periodicity is the physical defect this
    project studies (short-period correlation), distinct from the AR(1)
    copula source in noise.py which stays full-period but adds *serial*
    correlation with a stationary marginal.
  - A statistical test suite (period, lag autocorrelation, histogram chi2)
    used both to validate baseline generators and to *detect* the LFSR's
    periodicity defect.
  - run_stat_report(): runs the suite on the baseline (PCG64), the LFSR at
    two bit widths, and the AR(1) copula source, and writes results/rng_stats.json.

Galois LFSR mechanics: state is the full n-bit register. Each step, the LSB
is examined, the register is shifted right by one, and if the LSB was 1 the
register is XORed with a fixed tap mask. With a primitive feedback
polynomial this visits every nonzero n-bit state exactly once per period
(period = 2^n - 1); the all-zero state is a fixed point and is forbidden as
a seed. Word-level output (the raw n-bit register, not a single output bit)
is used to build next_uniform(), matching how p-bit hardware would drain the
whole register per draw.

Default taps below are standard maximal-length polynomials for the Galois
(internal-XOR) configuration, written as tap positions (1-indexed, including
the top tap n); mask = OR of (1 << (t - 1)) for t in taps. Verified
empirically (see tests/test_rng.py): 8-bit period == 255, 16-bit period ==
65535, 24-bit period == 16777215.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

from pneuron.noise import AR1UniformNoise, UniformNoise


class GaloisLFSR:
    """Galois-configuration LFSR with maximal-length default taps.

    next_uniform(n) builds n floats in [0,1) from successive n-bit LFSR
    words (state / 2**nbits). sample(shape) rescales to [-1,1] to match the
    noise-source API in pneuron/noise.py, so this drops into p_sample().
    """

    # tap positions (1-indexed, includes n), standard primitive polynomials
    # for the Galois configuration.
    DEFAULT_TAPS = {
        8: (8, 6, 5, 4),
        16: (16, 15, 13, 4),
        24: (24, 23, 22, 17),
    }

    def __init__(self, nbits, seed, taps=None):
        if taps is None:
            if nbits not in self.DEFAULT_TAPS:
                raise ValueError(
                    f"no default taps for nbits={nbits}; pass taps explicitly"
                )
            taps = self.DEFAULT_TAPS[nbits]
        self.nbits = nbits
        self.taps = tuple(taps)
        self.mask = 0
        for t in self.taps:
            self.mask |= 1 << (t - 1)
        self.modulus = 1 << nbits  # 2**nbits
        self.max_val = self.modulus - 1

        seed = int(seed) & self.max_val
        if seed == 0:
            seed = 1  # all-zero state is a fixed point; not a valid seed
        self.seed = seed
        self.state = seed

    def step(self):
        """Advance the register by one bit; return the new n-bit word."""
        lsb = self.state & 1
        self.state >>= 1
        if lsb:
            self.state ^= self.mask
        return self.state

    def next_uniform(self, n):
        """n floats in [0,1), one per successive n-bit LFSR word."""
        out = np.empty(n, dtype=float)
        for i in range(n):
            out[i] = self.step() / self.modulus
        return out

    def sample(self, shape):
        """r in [-1,1], same convention as pneuron/noise.py sources."""
        shape_t = shape if isinstance(shape, tuple) else (shape,) if np.isscalar(shape) else tuple(shape)
        n = int(np.prod(shape_t)) if shape_t else 1
        u = self.next_uniform(n)
        r = 2.0 * u - 1.0
        return r.reshape(shape)


# ---------------------------------------------------------------------------
# Statistical test suite
# ---------------------------------------------------------------------------

def period_estimate(source, max_draws):
    """Detect a repeat in source.sample((max_draws,)) up to a draw cap.

    Draws max_draws scalars and looks for the first exact repeated value.
    A maximal LFSR's raw words are a permutation of its nonzero states, so
    the first repeat marks exactly one full period. A good PRNG (continuous
    doubles, period far beyond max_draws) should show no repeat at all.

    pass = True iff no repeat was observed within max_draws (i.e. no short
    cycle detected -- this is the "good generator" outcome).
    """
    x = np.asarray(source.sample((max_draws,)), dtype=float)
    seen = {}
    period = None
    for i, v in enumerate(x):
        key = float(v)
        if key in seen:
            period = i - seen[key]
            break
        seen[key] = i
    passed = period is None
    return {
        "test": "period_estimate",
        "max_draws": max_draws,
        "period": period,
        "pass": bool(passed),
    }


def lag_autocorrelation(samples, lags=(1, 2, 3, 10), alpha=0.01):
    """Sample lag-k autocorrelation of samples, for k in lags.

    Under the iid null, Corr(r_t, r_{t+k}) ~ approximately Normal(0, 1/n)
    for large n, so |r_hat| <= z_{1-alpha/2} / sqrt(n) is the two-sided
    pass band at the given n (Bartlett's approximation, standard for white
    -noise checks). Reports per-lag statistics and passes overall iff every
    tested lag is within its band.
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    z = norm.ppf(1.0 - alpha / 2.0)
    threshold = z / np.sqrt(n)

    per_lag = {}
    overall_pass = True
    for lag in lags:
        if lag <= 0 or lag >= n:
            continue
        num = np.sum(xc[:-lag] * xc[lag:])
        r = float(num / denom) if denom > 0 else 0.0
        ok = abs(r) <= threshold
        overall_pass = overall_pass and ok
        per_lag[str(lag)] = {"r": r, "threshold": float(threshold), "pass": bool(ok)}

    return {
        "test": "lag_autocorrelation",
        "n": int(n),
        "alpha": alpha,
        "lags": per_lag,
        "pass": bool(overall_pass),
    }


def histogram_chi2(samples, bins=64, alpha=0.01):
    """Chi-square goodness-of-fit test for uniformity of samples on [-1,1]."""
    x = np.asarray(samples, dtype=float)
    n = x.size
    counts, edges = np.histogram(x, bins=bins, range=(-1.0, 1.0))
    expected = n / bins
    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    dof = bins - 1
    critical = float(chi2_dist.ppf(1.0 - alpha, dof))
    p_value = float(chi2_dist.sf(chi2_stat, dof))
    passed = chi2_stat <= critical

    return {
        "test": "histogram_chi2",
        "n": int(n),
        "bins": bins,
        "chi2": chi2_stat,
        "dof": dof,
        "critical": critical,
        "p_value": p_value,
        "alpha": alpha,
        "pass": bool(passed),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _decorrelation_stride(x, max_stride=50):
    """Smallest stride m (capped) such that x[::m] is ~serially independent.

    histogram_chi2 assumes independent draws (its variance formula is the
    plain multinomial one). A strongly autocorrelated sequence violates that
    and inflates the chi2 statistic even when the true marginal is exactly
    uniform (bin counts cluster because neighboring draws are similar) --
    that is a real property of chi2-on-correlated-data, not a bug. Thinning
    by the lag-1 correlation length restores near-independence for the
    marginal test, while lag_autocorrelation is always run on the raw
    (unthinned) sequence, where the correlation defect actually lives.
    """
    r1 = lag_autocorrelation(x, lags=(1,))["lags"].get("1", {}).get("r", 0.0)
    if abs(r1) < 0.05:
        return 1
    m = int(np.ceil(np.log(0.05) / np.log(abs(r1))))
    return max(1, min(m, max_stride))


def _run_suite(name, source, seed, n_samples, extra_lags=(), period_max_draws=None):
    """Run period/autocorrelation/chi2 on one source and package the result."""
    samples = np.asarray(source.sample((n_samples,)), dtype=float)
    lags = (1, 2, 3, 10) + tuple(extra_lags)
    lags = tuple(sorted(set(l for l in lags if 0 < l < n_samples)))

    pmax = period_max_draws if period_max_draws is not None else n_samples
    period = period_estimate(source, pmax)
    autocorr = lag_autocorrelation(samples, lags=lags)

    stride = _decorrelation_stride(samples)
    if stride > 1:
        # draw a fresh, longer batch so the thinned array still has enough
        # points for chi2 power, rather than shrinking n_samples by `stride`.
        chi2_raw = np.asarray(source.sample((n_samples * stride,)), dtype=float)
        chi2_samples = chi2_raw[::stride]
    else:
        chi2_samples = samples
    chi2r = histogram_chi2(chi2_samples, bins=64)
    chi2r["decorrelation_stride"] = stride

    overall_pass = period["pass"] and autocorr["pass"] and chi2r["pass"]
    return {
        "source": name,
        "seed": seed,
        "n_samples": n_samples,
        "period_estimate": period,
        "lag_autocorrelation": autocorr,
        "histogram_chi2": chi2r,
        "overall_pass": bool(overall_pass),
    }


def run_stat_report(n_samples=20000, seed=12345, out_path="results/rng_stats.json"):
    """Run the stat suite on PCG64, 8/16-bit GaloisLFSR, and AR(1) rho=0.9.

    Writes the full report to out_path and returns it. The AR(1) source is
    included as a suite self-check: it should PASS histogram_chi2 (its
    marginal is exact Uniform[-1,1] by construction, see noise.py) but FAIL
    lag_autocorrelation at lag 1 (rho=0.9 serial correlation) -- that
    contrast is the evidence the test suite actually detects what it claims to.
    """
    report = {}

    pcg64 = UniformNoise(seed=seed)
    report["pcg64"] = _run_suite("pcg64", pcg64, seed, n_samples)

    lfsr8 = GaloisLFSR(nbits=8, seed=seed)
    report["lfsr8"] = _run_suite(
        "lfsr8", lfsr8, seed, n_samples,
        extra_lags=(255,), period_max_draws=max(n_samples, 1000),
    )

    lfsr16 = GaloisLFSR(nbits=16, seed=seed)
    report["lfsr16"] = _run_suite(
        "lfsr16", lfsr16, seed, n_samples,
        period_max_draws=max(n_samples, 70000),
    )

    ar1 = AR1UniformNoise(seed=seed, rho=0.9)
    report["ar1_rho0.9"] = _run_suite("ar1_rho0.9", ar1, seed, n_samples)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_stat_report()
    for name, r in rep.items():
        print(name, "overall_pass=", r["overall_pass"])
