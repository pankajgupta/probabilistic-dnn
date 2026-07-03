"""E1 (H1, mechanism): single p-neuron activation deformation vs IST prediction.

For each randomness defect, the measured activation P(m=+1 | I) of a single
p-neuron (m = sign(tanh(I) - r), r drawn from the defective source) must match
the closed-form/numerically-computed CDF_r(tanh(I)) implied by the defective
source's own distribution -- this is the inverse-sampling-theorem (IST) claim
from HANDOFF.md sec 2 / sec 4 (E1 row).

Uses pneuron/neuron.py::firing_prob (measurement) unmodified, and the noise
sources in pneuron/noise.py (UniformNoise, AR1UniformNoise) and pneuron/rng.py
(GaloisLFSR) unmodified. Two additional noise sources (triangular, truncated
Gaussian) are NOT in pneuron/noise.py, so small local classes implementing the
same .sample(shape) interface are defined here, per the task instructions.

Conditions:
  1. ideal            -- UniformNoise, baseline
  2. bitdepth k=1,2,3  -- UniformNoise(k=k); prediction = exact staircase CDF
                          of the quantized (level-center) discrete uniform
  3. bias b=0.1,0.3    -- UniformNoise(bias=b); prediction = shifted uniform
                          CDF with clipping mass at r=+1 (edge atom)
  4. copula-AR(1) rho=0.9 -- AR1UniformNoise(rho=0.9); prediction = UNCHANGED
                          ideal CDF (marginal-preserving copula) -- negative
                          control: correlation is invisible to a marginal
                          activation measurement
  5. distribution mismatch:
       (a) triangular[-1,1] mode 0 -- prediction = triangular CDF
       (b) truncated Gaussian sigma=0.5 on [-1,1] -- prediction = truncnorm CDF
  6. GaloisLFSR 8-bit  -- prediction = approximately ideal uniform CDF (255
                          levels is fine-grained; the LFSR's defect is
                          temporal/periodic, invisible in a marginal
                          activation measurement) -- second negative control

Run: .venv/bin/python experiments/e1_activation.py
Writes: results/e1_activation.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pneuron.neuron import firing_prob
from pneuron.noise import AR1UniformNoise, UniformNoise
from pneuron.rng import GaloisLFSR

OUT_JSON = ROOT / "results" / "e1_activation.json"

I_GRID = np.linspace(-3.0, 3.0, 61)
N_DRAWS = 20000
SEEDS = (1, 2, 3)  # 3 RNG seeds, reused (as an index) across all conditions


# ---------------------------------------------------------------------------
# Local noise sources not present in pneuron/noise.py (same .sample(shape)
# interface as the pneuron sources; pneuron/noise.py itself is not touched).
# ---------------------------------------------------------------------------

class TriangularNoise:
    """r ~ Triangular(-1, mode=0, 1). Injected where Uniform[-1,1] was intended
    (distribution-mismatch defect, HANDOFF sec 3 family 4 / E1 condition 5a)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self, shape):
        return self.rng.triangular(-1.0, 0.0, 1.0, size=shape)


class TruncatedGaussianNoise:
    """r ~ Normal(0, sigma) truncated (not clipped) to [lo, hi].

    Uses scipy.stats.truncnorm for sampling so the empirical source matches
    the closed-form truncated-Gaussian CDF used as the prediction exactly
    (a true truncation, not a clip-to-edge with a point mass)."""

    def __init__(self, seed, sigma=0.5, lo=-1.0, hi=1.0):
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma
        self.a = lo / sigma
        self.b = hi / sigma

    def sample(self, shape):
        return truncnorm.rvs(
            self.a, self.b, loc=0.0, scale=self.sigma, size=shape, random_state=self.rng
        )


# ---------------------------------------------------------------------------
# Predicted CDFs (IST: predicted firing prob = CDF_r(tanh(I)))
# ---------------------------------------------------------------------------

def predict_ideal(x):
    """CDF of Uniform[-1,1]."""
    x = np.asarray(x, dtype=float)
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def predict_quantized_uniform(x, k):
    """Exact CDF of r quantized to 2^k level CENTERS (pneuron.noise.quantize
    semantics): a discrete uniform over `levels` equally-likely centers, so
    CDF(x) = (# centers <= x) / levels -- a staircase with `levels` risers."""
    x = np.asarray(x, dtype=float)
    levels = 2 ** k
    idx = np.arange(levels)
    centers = (idx + 0.5) / levels * 2.0 - 1.0
    counts = (centers[None, :] <= x[..., None]).sum(axis=-1)
    return counts / levels


def predict_biased_uniform(x, b):
    """CDF of r = clip(Uniform[-1,1] + b, -1, 1), for b >= 0 (the only sign
    tested here): a uniform density 1/2 on [b-1, 1) plus a point mass b/2 at
    r=+1 (the clipped tail). For the I grid used (tanh(I) < 1 strictly), the
    edge atom at x=1 is never queried on the interior, but the piecewise
    formula is written to be exact there too (CDF(x)=1 for x>=1)."""
    x = np.asarray(x, dtype=float)
    lo = b - 1.0
    mid = (x - lo) / 2.0
    return np.where(x < lo, 0.0, np.where(x >= 1.0, 1.0, mid))


def predict_triangular(x):
    """CDF of Triangular(-1, 0, 1) (symmetric, a=-1,b=1,c=0)."""
    x = np.asarray(x, dtype=float)
    left = (x + 1.0) ** 2 / 2.0
    right = 1.0 - (1.0 - x) ** 2 / 2.0
    return np.where(x <= -1.0, 0.0, np.where(x <= 0.0, left, np.where(x < 1.0, right, 1.0)))


def predict_truncnorm(x, sigma=0.5, lo=-1.0, hi=1.0):
    """CDF of Normal(0, sigma) truncated to [lo, hi]."""
    a, b = lo / sigma, hi / sigma
    return truncnorm.cdf(x, a, b, loc=0.0, scale=sigma)


# ---------------------------------------------------------------------------
# Condition table: (name, noise_factory(seed) -> noise, predict_fn(tanh_I))
# ---------------------------------------------------------------------------

def build_conditions():
    conds = []

    conds.append(("ideal", lambda seed: UniformNoise(seed=seed), predict_ideal))

    for k in (1, 2, 3):
        conds.append((
            f"bitdepth_k{k}",
            (lambda seed, k=k: UniformNoise(seed=seed, k=k)),
            (lambda x, k=k: predict_quantized_uniform(x, k)),
        ))

    for b in (0.1, 0.3):
        conds.append((
            f"bias_b{b}",
            (lambda seed, b=b: UniformNoise(seed=seed, bias=b)),
            (lambda x, b=b: predict_biased_uniform(x, b)),
        ))

    conds.append((
        "ar1_copula_rho0.9",
        (lambda seed: AR1UniformNoise(seed=seed, rho=0.9)),
        predict_ideal,  # negative control: marginal-preserving copula
    ))

    conds.append((
        "mismatch_triangular",
        (lambda seed: TriangularNoise(seed=seed)),
        predict_triangular,
    ))

    conds.append((
        "mismatch_truncgauss_sigma0.5",
        (lambda seed: TruncatedGaussianNoise(seed=seed, sigma=0.5)),
        predict_truncnorm,
    ))

    conds.append((
        "lfsr8bit",
        (lambda seed: GaloisLFSR(nbits=8, seed=seed)),
        predict_ideal,  # negative control: temporal defect, invisible in marginal
    ))

    return conds


def main():
    t0 = time.time()
    tanh_I = np.tanh(I_GRID)
    conditions = build_conditions()

    results = {}
    summary_rows = []

    for name, noise_factory, predict_fn in conditions:
        predicted = predict_fn(tanh_I)
        measured_per_seed = []
        max_abs_err_per_seed = []

        for seed in SEEDS:
            noise = noise_factory(seed)
            measured = firing_prob(I_GRID, noise, n_draws=N_DRAWS, pre="tanh")
            measured_per_seed.append(measured)
            err = np.max(np.abs(measured - predicted))
            max_abs_err_per_seed.append(float(err))

        measured_stack = np.stack(measured_per_seed, axis=0)
        measured_mean = measured_stack.mean(axis=0)

        mae_mean = float(np.mean(max_abs_err_per_seed))
        mae_std = float(np.std(max_abs_err_per_seed))

        results[name] = {
            "I_grid": I_GRID.tolist(),
            "measured_firing_prob_mean_over_seeds": measured_mean.tolist(),
            "measured_firing_prob_per_seed": [m.tolist() for m in measured_per_seed],
            "predicted_firing_prob": np.asarray(predicted, dtype=float).tolist(),
            "max_abs_error_per_seed": max_abs_err_per_seed,
            "max_abs_error_mean": mae_mean,
            "max_abs_error_std": mae_std,
            "n_draws": N_DRAWS,
            "seeds": list(SEEDS),
        }
        summary_rows.append((name, mae_mean, mae_std))
        print(f"{name:<32s} max_abs_err mean={mae_mean:.5f} std={mae_std:.5f}")

    # ---- sanity flags ----
    noise_floor_scale = float(np.sqrt(0.25 / N_DRAWS))  # ~binomial se at p=0.5
    smooth_conditions = {
        "ideal", "bias_b0.1", "bias_b0.3", "ar1_copula_rho0.9",
        "mismatch_triangular", "mismatch_truncgauss_sigma0.5", "lfsr8bit",
    }
    flags = []
    for name, mae_mean, mae_std in summary_rows:
        if name in smooth_conditions and mae_mean > 0.02:
            flags.append(f"{name}: max_abs_err mean={mae_mean:.5f} exceeds 0.02 flag threshold")
    for k in (1, 2, 3):
        name = f"bitdepth_k{k}"
        mae_mean = dict((n, m) for n, m, s in summary_rows)[name]
        if mae_mean > 0.02:
            flags.append(f"{name}: max_abs_err mean={mae_mean:.5f} exceeds 0.02 vs STAIRCASE prediction")

    total_dt = time.time() - t0

    results["meta"] = {
        "description": (
            "E1 / H1 mechanism check: for each randomness defect on the p-neuron "
            "comparator (m = sign(tanh(I) - r)), does the empirically measured "
            "firing rate P(m=+1|I) match the IST-predicted CDF_r(tanh(I)) computed "
            "from the defective r's actual (possibly re-derived) distribution?"
        ),
        "fields": {
            "I_grid": "np.linspace(-3,3,61) pre-activation input grid, list[float] len 61",
            "measured_firing_prob_mean_over_seeds": "empirical P(m=+1|I), mean over the 3 seeds, per I_grid point",
            "measured_firing_prob_per_seed": "empirical P(m=+1|I) for each of the 3 seeds individually, list of 3 lists",
            "predicted_firing_prob": "closed-form/numeric CDF_r(tanh(I)) per I_grid point, IST prediction (same for all seeds)",
            "max_abs_error_per_seed": "max_I |measured_seed - predicted| for each seed",
            "max_abs_error_mean": "mean of max_abs_error_per_seed over the 3 seeds",
            "max_abs_error_std": "std of max_abs_error_per_seed over the 3 seeds",
            "n_draws": "Monte Carlo draws per I_grid point per seed",
            "seeds": "the 3 RNG seeds used",
        },
        "n_draws": N_DRAWS,
        "seeds": list(SEEDS),
        "i_grid_spec": "linspace(-3, 3, 61)",
        "pre_activation": "tanh",
        "noise_floor_scale_sqrt_0.25_over_n_draws": noise_floor_scale,
        "sanity_flags": flags,
        "total_seconds": total_dt,
        "notes": [
            "ar1_copula_rho0.9 and lfsr8bit are mechanism-level NEGATIVE CONTROLS: "
            "both defects are temporal/serial (correlation, periodicity) and are "
            "invisible to a per-input marginal activation measurement -- both are "
            "predicted (and expected) to match the IDEAL/ideal-like CDF, not a "
            "deformed one. A large error there would indicate a bug, not H1 support.",
            "bias predictions include a point mass at r=+1 from clipping; the "
            "I grid used here never reaches tanh(I)=1 exactly (max |I|=3 -> "
            "tanh(3)=0.99505), so that atom is not exercised by this measurement.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary table ----
    print("\n" + "=" * 60)
    print(f"{'condition':<32s}{'max_abs_err (mean+-std)':>28s}")
    print("-" * 60)
    for name, mae_mean, mae_std in summary_rows:
        print(f"{name:<32s}{mae_mean:>14.5f} +- {mae_std:<10.5f}")
    print("=" * 60)
    print(f"noise floor scale ~sqrt(0.25/n_draws) = {noise_floor_scale:.5f}")
    if flags:
        print("FLAGS:")
        for f_ in flags:
            print(f"  - {f_}")
    else:
        print("No flags: all conditions within threshold of their prediction.")
    print(f"Total runtime: {total_dt:.1f}s")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
