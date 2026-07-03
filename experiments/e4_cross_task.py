"""E4 cross-task TV-vs-defect-strength on the frustrated triangle (H4).

Same defect axes as the MNIST experiments (E2 bit-depth, E3 correlation, plus
bias and LFSR), applied to the exact-target frustrated-triangle Gibbs sampler
(groundtruth.triangle, UNMODIFIED) instead of MNIST accuracy. Output: TV
distance from the exact 8-state distribution vs defect strength, so Phase D
can overlay this against the MNIST accuracy-sensitivity curves.

Open question this experiment answers (not assumes): is the triangle sampler
MORE sensitive than MNIST accuracy to the same nominal defect strength,
including for correlation? For rho in particular the mechanism differs from
the feedforward case -- rho correlates successive Gibbs UPDATE draws, which
directly corrupts the Markov chain of a 3-spin sampler (a structural channel
that a feedforward, single-pass p-DNN does not have). Results are reported
plainly; no equivalence between the two mechanisms is claimed.

Performance strategy: groundtruth.triangle.gibbs() is an unmodified Python
loop that calls noise.sample(()) 3x per sweep. Rather than let a stateful
noise object (esp. AR1UniformNoise, which is a per-element Python loop)
sit in that hot loop, we PRE-GENERATE the full r-sequence for a run
vectorized, then hand gibbs() a thin NoiseCursor that pops pre-generated
values sequentially and raises if a run would need to wrap. This keeps
gibbs() itself completely unmodified while making every source equally fast.

Run: .venv/bin/python experiments/e4_cross_task.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, lfiltic
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.triangle import gibbs, exact_dist, tv
from pneuron.noise import quantize
from pneuron.rng import GaloisLFSR, histogram_chi2, lag_autocorrelation

OUT_JSON = ROOT / "results" / "e4_cross_task.json"

SWEEPS = 1_000_000
BURN = 10_000
SEEDS = (0, 1, 2, 3, 4)  # 5 RNG seeds per condition

K_VALUES = (1, 2, 3, 4, 6, 8, 16)
BIAS_VALUES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
RHO_VALUES = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
LFSR_BITS = (8, 16)

EXACT = exact_dist()

# Total planned runs, for the runtime-budget calibration.
N_RUNS = len(SEEDS) * (1 + len(K_VALUES) + len(BIAS_VALUES) + len(RHO_VALUES) + len(LFSR_BITS))


# ---------------------------------------------------------------------------
# Noise cursor: pre-generated sequence, sequential pop, never wraps/reuses.
# ---------------------------------------------------------------------------

class NoiseCursor:
    """Feeds a pre-generated r-array to gibbs() via the same .sample(shape)
    API as the live noise sources. Raises if a run would need to reuse
    (wrap) values -- that would silently reintroduce short-period-style
    correlation the experiment isn't asking for.
    """

    def __init__(self, r, name=""):
        self.r = np.asarray(r, dtype=float)
        self.pos = 0
        self.name = name

    def sample(self, shape):
        shape_t = shape if isinstance(shape, tuple) else (shape,)
        n = int(np.prod(shape_t)) if shape_t else 1
        if self.pos + n > self.r.size:
            raise RuntimeError(
                f"NoiseCursor[{self.name}] exhausted: requested {n} at "
                f"pos {self.pos}, have {self.r.size} pre-generated"
            )
        out = self.r[self.pos:self.pos + n]
        self.pos += n
        return out.reshape(shape_t)


def n_draws(sweeps, burn):
    """gibbs() draws one r per p-bit per sweep, 3 p-bits per sweep."""
    return 3 * (sweeps + burn)


def make_seed(family_id, param_idx, seed_idx):
    """Deterministic, collision-free integer seed per (family, param, seed)."""
    return family_id * 10_000_000 + param_idx * 100_000 + seed_idx * 1000 + 7


# ---------------------------------------------------------------------------
# Vectorized pre-generators, one per noise source (must match the semantics
# of pneuron/noise.py and pneuron/rng.py, just batched instead of per-call).
# ---------------------------------------------------------------------------

def gen_uniform(seed, n, k=None, bias=0.0):
    """Matches UniformNoise(seed, k, bias).sample(...), batched."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(-1.0, 1.0, size=n)
    if k is not None:
        r = quantize(r, k)
    if bias:
        r = np.clip(r + bias, -1.0, 1.0)
    return r


def gen_ar1(seed, n, rho):
    """Vectorized reproduction of AR1UniformNoise's recursion.

    g_0 ~ N(0,1) (the class's initial chain state), then for t=1..n:
    g_t = rho*g_{t-1} + sqrt(1-rho^2)*eps_t (a first-order IIR filter of the
    innovations, done here with scipy.signal.lfilter/lfiltic instead of
    AR1UniformNoise's per-element Python loop). r = 2*Phi(g)-1 is the
    Gaussian-copula transform back to an exact Uniform[-1,1] marginal.
    """
    rng = np.random.default_rng(seed)
    g0 = rng.standard_normal()
    eps = rng.standard_normal(n)
    c = np.sqrt(1.0 - rho ** 2)
    b_coef = [c]
    a_coef = [1.0, -rho]
    zi = lfiltic(b_coef, a_coef, y=[g0])
    g, _ = lfilter(b_coef, a_coef, eps, zi=zi)
    r = 2.0 * ndtr(g) - 1.0
    return r


def gen_lfsr(seed, n, nbits):
    """Full-period GaloisLFSR draws, batched via its own next_uniform(n)."""
    lfsr = GaloisLFSR(nbits=nbits, seed=seed)
    u = lfsr.next_uniform(n)
    return 2.0 * u - 1.0


# ---------------------------------------------------------------------------
# Correctness checks (run before the full sweep)
# ---------------------------------------------------------------------------

def check_ideal_tv():
    """(a) Reproduce ideal TV < 0.01 at 100k sweeps with a good RNG."""
    n = n_draws(100_000, 2_000)
    r = gen_uniform(seed=42, n=n)
    cursor = NoiseCursor(r, name="check_ideal")
    dist = gibbs(cursor, sweeps=100_000, burn=2_000, seed_state=42)
    t = float(tv(dist, EXACT))
    return {"tv": t, "pass": bool(t < 0.01)}


def check_ar1_marginal_and_corr(rho=0.9, n=200_000):
    """(b) Pre-generated AR(1) sequence: uniform marginal (chi2), and lag-1
    correlation in the ballpark of rho. Note the Gaussian-copula transform
    compresses linear (Pearson) correlation of the *uniform* output relative
    to the underlying Gaussian rho by the standard factor
    Corr(U,V) = (6/pi) * arcsin(rho/2) for a Gaussian copula -- so we check
    against that compressed target, not against rho directly.
    """
    r = gen_ar1(seed=999, n=n, rho=rho)
    chi2r = histogram_chi2(r, bins=64)
    autocorr = lag_autocorrelation(r, lags=(1,))
    measured_r1 = autocorr["lags"]["1"]["r"]
    expected_pearson = (6.0 / np.pi) * np.arcsin(rho / 2.0)
    close = abs(measured_r1 - expected_pearson) < 0.02
    return {
        "rho": rho,
        "n": n,
        "chi2_pass_marginal_uniform": chi2r["pass"],
        "chi2_p_value": chi2r["p_value"],
        "measured_lag1_pearson_r": measured_r1,
        "expected_lag1_pearson_r_gaussian_copula": float(expected_pearson),
        "pass": bool(chi2r["pass"] and close),
    }


def check_cursor_no_wraparound():
    """(c) Cursor must raise, not wrap/reuse, when exhausted."""
    r = np.linspace(-1, 1, 10)
    cursor = NoiseCursor(r, name="check_wrap")
    cursor.sample((10,))  # exhausts exactly
    try:
        cursor.sample(())
        raised = False
    except RuntimeError:
        raised = True
    return {"pass": bool(raised)}


def run_correctness_checks():
    print("Running correctness checks...")
    a = check_ideal_tv()
    print(f"  (a) ideal TV@100k sweeps = {a['tv']:.5f}  pass={a['pass']}")
    b = check_ar1_marginal_and_corr()
    print(f"  (b) AR1 rho=0.9: chi2_pass={b['chi2_pass_marginal_uniform']} "
          f"measured_r1={b['measured_lag1_pearson_r']:.4f} "
          f"expected(copula-compressed)={b['expected_lag1_pearson_r_gaussian_copula']:.4f} "
          f"pass={b['pass']}")
    c = check_cursor_no_wraparound()
    print(f"  (c) cursor no-wraparound pass={c['pass']}")
    all_pass = a["pass"] and b["pass"] and c["pass"]
    print(f"  correctness checks overall pass={all_pass}")
    return {"ideal_tv_check": a, "ar1_check": b, "cursor_check": c, "all_pass": bool(all_pass)}


# ---------------------------------------------------------------------------
# Runtime calibration
# ---------------------------------------------------------------------------

def calibrate(sweeps_probe=20_000, burn_probe=1_000):
    n = n_draws(sweeps_probe, burn_probe)
    r = gen_uniform(seed=7, n=n)
    cursor = NoiseCursor(r, name="calibrate")
    t0 = time.time()
    gibbs(cursor, sweeps=sweeps_probe, burn=burn_probe, seed_state=7)
    dt = time.time() - t0
    per_sweep = dt / (sweeps_probe + burn_probe)
    projected_per_run = per_sweep * (SWEEPS + BURN)
    projected_total = projected_per_run * N_RUNS
    print(f"Calibration: {dt:.3f}s for {sweeps_probe + burn_probe} sweeps "
          f"-> {per_sweep * 1e6:.3f} us/sweep, projected per-run "
          f"{projected_per_run:.2f}s, projected total for {N_RUNS} runs: "
          f"{projected_total:.1f}s ({projected_total / 60:.1f} min)")
    return projected_total


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def run_condition(family_id, param_idx, source_fn, sweeps, burn, seeds=SEEDS):
    n = n_draws(sweeps, burn)
    tvs = []
    rep_dist = None
    for i, s in enumerate(seeds):
        seed = make_seed(family_id, param_idx, s)
        r = source_fn(seed, n)
        cursor = NoiseCursor(r, name=f"fam{family_id}:param{param_idx}:seed{s}")
        dist = gibbs(cursor, sweeps=sweeps, burn=burn, seed_state=seed)
        t = float(tv(dist, EXACT))
        tvs.append(t)
        if i == 0:
            rep_dist = dist.tolist()
    return {
        "tv_per_seed": tvs,
        "tv_mean": float(np.mean(tvs)),
        "tv_std": float(np.std(tvs)),
        "seeds_used": list(seeds),
        "representative_empirical_dist": rep_dist,
        "representative_seed": seeds[0],
    }


def main():
    t_start = time.time()

    checks = run_correctness_checks()

    projected_total = calibrate()
    sweeps, burn = SWEEPS, BURN
    reduced = False
    if projected_total > 25 * 60:
        print("Projected runtime exceeds 25 min budget; reducing sweeps to 500k uniformly.")
        sweeps, burn = 500_000, BURN
        reduced = True

    results = {}

    # 1. ideal uniform (reference)
    print("\n[1/5] ideal uniform (reference)...")
    results["ideal"] = run_condition(
        family_id=0, param_idx=0,
        source_fn=lambda seed, n: gen_uniform(seed, n),
        sweeps=sweeps, burn=burn,
    )
    print(f"  tv_mean={results['ideal']['tv_mean']:.5f} tv_std={results['ideal']['tv_std']:.5f}")

    # 2. bit-depth k
    print("\n[2/5] bit-depth k sweep...")
    bitdepth = {"axis_name": "k", "axis_values": list(K_VALUES), "results": {}}
    for pidx, k in enumerate(K_VALUES):
        r = run_condition(
            family_id=1, param_idx=pidx,
            source_fn=lambda seed, n, k=k: gen_uniform(seed, n, k=k),
            sweeps=sweeps, burn=burn,
        )
        bitdepth["results"][str(k)] = r
        print(f"  k={k:<3d} tv_mean={r['tv_mean']:.5f} tv_std={r['tv_std']:.5f}")
    results["bitdepth"] = bitdepth

    # 3. bias b
    print("\n[3/5] bias b sweep...")
    bias = {"axis_name": "b", "axis_values": list(BIAS_VALUES), "results": {}}
    for pidx, bval in enumerate(BIAS_VALUES):
        r = run_condition(
            family_id=2, param_idx=pidx,
            source_fn=lambda seed, n, bval=bval: gen_uniform(seed, n, bias=bval),
            sweeps=sweeps, burn=burn,
        )
        bias["results"][str(bval)] = r
        print(f"  b={bval:<5g} tv_mean={r['tv_mean']:.5f} tv_std={r['tv_std']:.5f}")
    results["bias"] = bias

    # 4. copula-AR(1) rho
    print("\n[4/5] copula-AR(1) rho sweep (correlation across successive Gibbs update draws)...")
    correlation = {"axis_name": "rho", "axis_values": list(RHO_VALUES), "results": {}}
    for pidx, rho in enumerate(RHO_VALUES):
        r = run_condition(
            family_id=3, param_idx=pidx,
            source_fn=lambda seed, n, rho=rho: gen_ar1(seed, n, rho=rho),
            sweeps=sweeps, burn=burn,
        )
        correlation["results"][str(rho)] = r
        print(f"  rho={rho:<5g} tv_mean={r['tv_mean']:.5f} tv_std={r['tv_std']:.5f}")
    results["correlation"] = correlation

    # 5. GaloisLFSR 8-bit / 16-bit
    print("\n[5/5] GaloisLFSR sweep...")
    lfsr = {"axis_name": "nbits", "axis_values": list(LFSR_BITS), "results": {}}
    for pidx, nbits in enumerate(LFSR_BITS):
        r = run_condition(
            family_id=4 + pidx, param_idx=0,
            source_fn=lambda seed, n, nbits=nbits: gen_lfsr(seed, n, nbits=nbits),
            sweeps=sweeps, burn=burn,
        )
        lfsr["results"][str(nbits)] = r
        print(f"  nbits={nbits:<3d} tv_mean={r['tv_mean']:.5f} tv_std={r['tv_std']:.5f}")
    results["lfsr"] = lfsr

    total_dt = time.time() - t_start

    out = {
        "_meta": {
            "sweeps": sweeps,
            "burn": burn,
            "sweeps_reduced_from_1M": reduced,
            "seeds": list(SEEDS),
            "n_seeds_per_condition": len(SEEDS),
            "exact_dist": EXACT.tolist(),
            "exact_dist_state_key": "state index = m0>0*1 + m1>0*2 + m2>0*4 "
                                     "(bit i set iff m_i=+1); states {0,7} aligned, "
                                     "{1..6} frustrated (see groundtruth/triangle.py)",
            "defect_axes": {
                "bitdepth_k": list(K_VALUES),
                "bias_b": list(BIAS_VALUES),
                "correlation_rho": list(RHO_VALUES),
                "lfsr_nbits": list(LFSR_BITS),
            },
            "correctness_checks": checks,
            "calibration_projected_total_seconds": projected_total,
            "total_runtime_seconds": total_dt,
            "n_runs": N_RUNS,
            "note_rho_mechanism": (
                "rho correlates successive Gibbs UPDATE draws (the sampling "
                "analog of across-samples correlation), which directly "
                "corrupts the Markov chain of this 3-spin sampler. This is a "
                "structurally different corruption channel from the "
                "feedforward multi-sample case (E3/H3); no equivalence "
                "between the two is claimed here, only that both are "
                "'correlated randomness' defects."
            ),
        },
    }
    out.update(results)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # ---- summary table ----
    print("\n" + "=" * 60)
    print(f"{'condition':<22}{'TV mean±std':>25}")
    print("-" * 60)
    print(f"{'ideal (reference)':<22}"
          f"{results['ideal']['tv_mean']:>14.5f} ± {results['ideal']['tv_std']:<7.5f}")
    for k in K_VALUES:
        rr = bitdepth["results"][str(k)]
        print(f"{'bitdepth k=' + str(k):<22}{rr['tv_mean']:>14.5f} ± {rr['tv_std']:<7.5f}")
    for bval in BIAS_VALUES:
        rr = bias["results"][str(bval)]
        print(f"{'bias b=' + str(bval):<22}{rr['tv_mean']:>14.5f} ± {rr['tv_std']:<7.5f}")
    for rho in RHO_VALUES:
        rr = correlation["results"][str(rho)]
        print(f"{'corr rho=' + str(rho):<22}{rr['tv_mean']:>14.5f} ± {rr['tv_std']:<7.5f}")
    for nbits in LFSR_BITS:
        rr = lfsr["results"][str(nbits)]
        print(f"{'LFSR nbits=' + str(nbits):<22}{rr['tv_mean']:>14.5f} ± {rr['tv_std']:<7.5f}")
    print("=" * 60)
    print(f"Total runtime: {total_dt:.1f}s ({total_dt / 60:.1f} min)")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
