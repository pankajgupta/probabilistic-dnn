# p-DNN Noise-Reliability Study — Expanded Overnight Plan (v2)

## Context

HANDOFF.md describes the study: a feedforward network of p-bits (`m = sign(f(I) − r)`),
controlled randomness defects injected at the comparator, measuring where accuracy — and
especially the multi-sample benefit — breaks down. Headline hypothesis H3: the sampling
advantage collapses under **correlated** noise but tolerates **low-precision** noise.

v1 was a ~1-hour mini version. This v2 expands to **≥4 hours of agent work, run overnight**,
covering all four defect families, full sweep grids, the sample-aware training extension,
and a complete report. Execution model per user instruction:

- **Fable orchestrates ONLY** — spawns subagents, checks gates between phases, never writes
  project code itself.
- **Sonnet subagents** do the coding/experiments by default; **Opus subagents** for the
  hardest pieces (the p-DNN noise-tensor wrapper review, straight-through training, final
  report synthesis) or after a failed Sonnet attempt — per the user's global routing policy.

**Already done (validated):** `.venv` with deps; `pneuron/noise.py` (uniform + quantize +
bias + copula-AR(1)), `pneuron/neuron.py`, `groundtruth/triangle.py`; triangle gate PASSED
(TV = 0.0026 vs exact). These files are the starting point — subagents extend, don't rewrite.

**Still out of scope:** CIFAR-10, RBMs, FPGA/hardware anything, the paper itself (report ≠ paper).

---

## Load-bearing design decisions (unchanged from v1)

1. **Copula-AR(1) correlation**: Gaussian AR(1) mapped through `r = 2Φ(g) − 1` keeps the
   marginal exactly Uniform[−1,1] — degrades only independence, not activation shape.
2. **Aggregation**: average logits over S passes; E[m] = tanh(z) ⇒ converges to the
   deterministic net's logits, giving sanity checks a known limit.
3. **Primary route**: frozen-weight deterministic training, p-neurons swapped at inference.
4. **Honest reporting**: E2 tolerance expected undramatic; don't manufacture a knee. If H3
   collapse doesn't appear, report what does. Claim only the reliability study (p-DNN is [2],
   configurable activations are [1]).
5. **Sanity gate before any defect sweep**: with ideal noise, accuracy(S→16) must approach
   deterministic accuracy AND the S=1→S=16 gain must be ≥ ~3 pts on at least one trained
   net (else the collapse is unmeasurable). Contingency: narrower nets create the gain.

## Expansion over v1 (what "next logical steps" adds)

| Axis | v1 (mini) | v2 (overnight) |
|------|-----------|----------------|
| Defect families | precision, bias(E1 only), correlation | + **distribution mismatch** (triangular/truncated/Gaussian-for-uniform), + **real LFSR** source, + bias on MNIST |
| RNG rigor | none | `pneuron/rng.py`: Galois LFSR (8/16-bit), **stat tests** (period, autocorrelation, histogram χ²) gating every generator before use |
| E2 grid | k ∈ {1,2,4,8}, S ∈ {1,16} | k ∈ {1..8,12,16}, S ∈ {1,2,4,8,16,32} |
| E3 grid | ρ ∈ {0,.9,1}, S ≤ 16 | ρ ∈ {0,.25,.5,.75,.9,.95,.99,1}, S ∈ {1..64}; **two correlation axes**: across-samples AND across-neurons(shared within layer) |
| E4 grid | 8 conditions, 100k sweeps | full k×ρ×bias grids, 1M sweeps, + LFSR condition; quantitative MNIST-vs-triangle sensitivity overlay |
| Operating points | 1 net | **3 widths** (784-256-128-10, 784-64-32-10, 784-32-16-10) × **3 training seeds** — shows how sampling gain & its collapse depend on operating point |
| Eval | 2k test subset, 3 noise seeds | **full 10k test set**, 5 noise seeds per condition |
| Calibration | skipped | **ECE figure**: multi-sample softmax ECE vs noise quality (ideal / quantized / correlated) |
| Training | deterministic only | + **sample-aware straight-through training** ([2]'s alternative route): does training WITH noise buy robustness to defective noise? (H5, exploratory) |
| Report | 4 figures | **8–10 figures**, REPORT.md + HTML artifact, limitations section |

Compute stays CPU-feasible: everything is small-MLP inference; the full grid is
~1–2 h of actual compute spread across the night. Agent work (coding, debugging,
validation, figures, writing) is the dominant cost — ≥4 h total.

---

## Orchestration plan (Fable = conductor only)

Phases run sequentially with validation gates; agents within a phase run in parallel
where files are disjoint. All agents get: pointer to HANDOFF.md, the relevant plan
excerpt, exact file contracts (paths in/out), and the honest-reporting guardrails.
Results = JSON in `results/` (seeds logged); figures = PNG in `plots/`.

### Phase A — Foundation (Sonnet, ~2 agents, sequential-ish)
- **A1 `models/net.py` + training**: MLP class (parametric widths), train 3 widths × 3 seeds
  (9 checkpoints, each <1 min), save `results/models/*.pt` + `results/baseline_acc.json`
  (full 10k test accuracy each).
- **A2 `models/pdnn.py`**: torch p-bit inference wrapper. Noise tensor (S, B, H) with
  k-bit quantize / bias / copula-AR(1) along the **sample** axis / shared-across-**neuron**
  mode / LFSR-backed mode. Deterministic given seed. Unit tests: E[m]→tanh, marginal
  uniformity under ρ>0, quantile check on quantized levels.
- **A3 `pneuron/rng.py`**: Galois LFSR + stat tests (period, lag-autocorrelation, χ²).
  Tests must PASS for PCG64 and FAIL (period/autocorr) for a short LFSR — that contrast
  is itself reportable.
- **GATE A** (Fable checks): unit tests green; baseline accuracies sane (≥95% big net);
  sanity curve: ideal-noise accuracy vs S climbs to within 1 pt of deterministic, S=1→16
  gain ≥3 pts on at least one width. Else: one debug round (escalate to Opus).

### Phase B — Experiments (Sonnet, 4 parallel agents on disjoint files)
- **B1 E1** (`experiments/e1_activation.py`): single-neuron measured vs IST-predicted
  activation for ALL defect families incl. mismatch + LFSR. Output: JSON + max-abs-error
  table per condition.
- **B2 E2 + bias** (`experiments/e2_precision.py`): full k×S grid + bias sweep
  b ∈ {0,.05,.1,.2,.3,.5} on MNIST, 3 widths × 3 training seeds × 5 noise seeds.
- **B3 E3** (`experiments/e3_sampling_fragility.py`): full ρ×S grid, both correlation
  axes, plus the "coarse-but-independent" (k=2, ρ=0) contrast curves; + ECE computation
  (ideal vs k=2 vs ρ=0.9 vs ρ=1).
- **B4 E4** (`experiments/e4_cross_task.py`): triangle TV vs k, ρ, bias grids + LFSR,
  1M sweeps × 5 seeds; normalized sensitivity overlay data (TV-vs-defect alongside
  MNIST Δaccuracy-vs-defect on shared defect axes).
- **GATE B** (Fable checks): each JSON exists, is self-consistent (monotonicity spot
  checks, e.g. E3 ρ=0 curve ≈ sanity curve; E4 ρ=0/k=16 TV ≈ good-RNG TV), seeds logged.

### Phase C — Extension: sample-aware training (Opus, 1 agent)
- **C1** (`models/ste_train.py` + `experiments/e5_ste_robustness.py`): straight-through-
  estimator training of the mid-width net WITH p-bit noise (S=1 and S=4 during training);
  then re-run the E2/E3 core conditions on the STE net vs the frozen-weight net.
  Question (H5): does noise-aware training buy robustness to quantized/correlated noise?
  Timebox: if STE training doesn't converge in 2 attempts, report as negative/incomplete
  and move on — the primary route doesn't depend on it.
- **GATE C**: STE net trains to ≥90% (else invoke timebox), comparison JSONs present.

### Phase D — Figures + report (Sonnet for figures w/ dataviz skill; Opus for synthesis)
- **D1 Figures** (~8–10, from JSONs only — no recomputation): E1 panel (2×3 defects),
  RNG stat-test table/figure, E2 precision curves + knee, bias curve, E3 headline
  (the money plot), E3b cross-neuron variant, ECE curve, E4 TV curves + cross-task
  overlay, H5 STE comparison, triangle validation bar chart (appendix).
  MUST load dataviz skill before writing plotting code.
- **D2 Report**: `REPORT.md` (methods recap, per-hypothesis findings H1–H5, honest
  headline sentence, limitations: 1 dataset, MLP-only, parametric-not-device noise,
  training-seed count) + HTML artifact version. Written from the JSONs + figures;
  every claim traceable to a results file. Fable reviews for overstatement before done.

### Cross-cutting rules for all agents
- Never call a black-box Bernoulli — noise injected at the comparator only.
- Fresh independent RNG state per (condition, seed) — never share state across the
  "independent" baselines.
- Frozen weights across all noise conditions (except Phase C, which is explicitly about
  retraining).
- `requirements.txt` written in Phase A; every script runnable as
  `.venv/bin/python experiments/<name>.py` idempotently.

## Timeline (overnight, wall-clock)
- Phase A ≈ 60–75 min (incl. gate + possible debug round)
- Phase B ≈ 90–120 min agent work, parallel ⇒ ~60–75 min wall-clock; compute inside
- Phase C ≈ 45–60 min
- Phase D ≈ 60–75 min
- Total ≈ 4–5.5 h wall-clock; agent-work sum comfortably ≥ 4 h.

## Verification (definition of done)
1. All unit tests + RNG stat tests green; triangle gate re-verified at 1M sweeps.
2. Sanity gate numbers recorded in results (ideal-noise convergence + gain size).
3. Every figure regenerable from committed JSONs; seeds logged everywhere.
4. REPORT.md answers H1–H5 explicitly, each with figure + effect size, including any
   null/contrary results; headline sentence matches the data, not the hypothesis.
