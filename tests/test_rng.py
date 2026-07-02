"""Tests for pneuron/rng.py: GaloisLFSR and the RNG statistical test suite.

No pytest in .venv; plain assert-based tests, runnable directly:
  .venv/bin/python tests/test_rng.py
(also collectible by `pytest -q` if pytest is ever installed).
"""

import os
import sys

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pneuron.noise import AR1UniformNoise, UniformNoise
from pneuron.rng import (
    GaloisLFSR,
    histogram_chi2,
    lag_autocorrelation,
    period_estimate,
    run_stat_report,
)

SEED = 12345


# ---------------------------------------------------------------------------
# GaloisLFSR mechanics
# ---------------------------------------------------------------------------

def test_lfsr8_period_is_255():
    lfsr = GaloisLFSR(nbits=8, seed=1)
    start = lfsr.state
    steps = 0
    for _ in range(300):
        lfsr.step()
        steps += 1
        if lfsr.state == start:
            break
    assert steps == 255, f"8-bit LFSR period = {steps}, want 255"


def test_lfsr16_period_is_65535():
    lfsr = GaloisLFSR(nbits=16, seed=1)
    start = lfsr.state
    steps = 0
    for _ in range(70000):
        lfsr.step()
        steps += 1
        if lfsr.state == start:
            break
    assert steps == 65535, f"16-bit LFSR period = {steps}, want 65535"


def test_lfsr_seed_zero_is_remapped():
    # all-zero state is a fixed point of a Galois LFSR; must not be a valid seed
    lfsr = GaloisLFSR(nbits=8, seed=0)
    assert lfsr.state != 0
    for _ in range(10):
        lfsr.step()
        assert lfsr.state != 0


def test_lfsr_sample_range_and_shape():
    lfsr = GaloisLFSR(nbits=8, seed=SEED)
    r = lfsr.sample((100, 3))
    assert r.shape == (100, 3)
    assert np.all(r >= -1.0) and np.all(r < 1.0)


def test_lfsr_next_uniform_range():
    lfsr = GaloisLFSR(nbits=8, seed=SEED)
    u = lfsr.next_uniform(1000)
    assert np.all(u >= 0.0) and np.all(u < 1.0)


def test_lfsr_reproducible_with_same_seed():
    a = GaloisLFSR(nbits=16, seed=777).sample((500,))
    b = GaloisLFSR(nbits=16, seed=777).sample((500,))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Statistic-suite functions in isolation
# ---------------------------------------------------------------------------

def test_period_estimate_finds_lfsr8_period():
    lfsr = GaloisLFSR(nbits=8, seed=SEED)
    result = period_estimate(lfsr, max_draws=2000)
    assert result["period"] == 255, result
    assert result["pass"] is False  # short cycle detected = defect flagged


def test_period_estimate_pcg64_no_cycle_within_cap():
    src = UniformNoise(seed=SEED)
    result = period_estimate(src, max_draws=20000)
    assert result["period"] is None
    assert result["pass"] is True


def test_lag1_autocorrelation_near_one_at_lfsr8_period():
    # the defining defect: sampling far beyond the period (n >> 255) makes
    # the sequence-at-lag=period look almost perfectly self-correlated.
    lfsr = GaloisLFSR(nbits=8, seed=SEED)
    samples = lfsr.sample((20000,))
    result = lag_autocorrelation(samples, lags=(255,))
    r255 = result["lags"]["255"]["r"]
    assert r255 > 0.95, f"lag-255 autocorrelation = {r255}, expected ~1"
    assert result["pass"] is False


def test_ar1_fails_lag1_autocorrelation():
    src = AR1UniformNoise(seed=SEED, rho=0.9)
    samples = src.sample((20000,))
    result = lag_autocorrelation(samples, lags=(1, 2, 3, 10))
    assert result["lags"]["1"]["pass"] is False
    assert result["pass"] is False


def test_ar1_passes_chi2_uniformity():
    # marginal is exact Uniform[-1,1] by construction (Gaussian-copula AR(1));
    # decorrelate by the lag-1 correlation length before testing, since chi2
    # goodness-of-fit assumes independent draws (see pneuron/rng._decorrelation_stride).
    src = AR1UniformNoise(seed=SEED, rho=0.9)
    raw = src.sample((20000 * 30,))
    thinned = raw[::30]
    result = histogram_chi2(thinned, bins=64)
    assert result["pass"] is True, result


def test_histogram_chi2_detects_bias():
    src = UniformNoise(seed=SEED, bias=0.2)
    samples = src.sample((20000,))
    result = histogram_chi2(samples, bins=64)
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def test_report_pcg64_passes_all():
    report = run_stat_report(seed=SEED, out_path="/tmp/_rng_stats_test.json")
    pcg = report["pcg64"]
    assert pcg["period_estimate"]["pass"] is True
    assert pcg["lag_autocorrelation"]["pass"] is True
    assert pcg["histogram_chi2"]["pass"] is True
    assert pcg["overall_pass"] is True


def test_report_lfsr8_fails_period_relative_tests():
    report = run_stat_report(seed=SEED, out_path="/tmp/_rng_stats_test.json")
    lfsr8 = report["lfsr8"]
    assert lfsr8["period_estimate"]["period"] == 255
    assert lfsr8["period_estimate"]["pass"] is False
    assert lfsr8["lag_autocorrelation"]["lags"]["255"]["r"] > 0.95
    assert lfsr8["overall_pass"] is False


def test_report_lfsr16_period_correct():
    report = run_stat_report(seed=SEED, out_path="/tmp/_rng_stats_test.json")
    lfsr16 = report["lfsr16"]
    assert lfsr16["period_estimate"]["period"] == 65535
    assert lfsr16["period_estimate"]["pass"] is False


def test_report_ar1_contrast():
    # the suite-validating contrast: correlated-but-uniform-marginal source
    # fails autocorrelation but passes chi2.
    report = run_stat_report(seed=SEED, out_path="/tmp/_rng_stats_test.json")
    ar1 = report["ar1_rho0.9"]
    assert ar1["lag_autocorrelation"]["lags"]["1"]["pass"] is False
    assert ar1["histogram_chi2"]["pass"] is True
    assert ar1["overall_pass"] is False


def test_report_writes_json():
    import json

    out_path = "/tmp/_rng_stats_test2.json"
    report = run_stat_report(seed=SEED, out_path=out_path)
    with open(out_path) as f:
        loaded = json.load(f)
    assert set(loaded.keys()) == set(report.keys())
    os.remove(out_path)


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
