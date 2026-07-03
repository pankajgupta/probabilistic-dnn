# COLLABORATOR HANDOFF — Independent Verification of the p-DNN Noise-Reliability Study

> You are a Claude Code instance helping a collaborator **independently verify** a
> completed study, then propose extensions. This file is self-contained: it has the
> background, every instruction, the exact expected numbers with tolerances, and the
> open questions. Read it fully before running anything. Your two jobs, in order:
> **(1) re-run and re-verify every result below, reporting discrepancies honestly;
> (2) propose and (if the collaborator agrees) prototype further directions.**

---

## 0. What this project is

A p-bit neuron fires by comparing its input against a random draw:
`m = sign(tanh(I) − r)`, `r` fresh every pass. Since `m=+1` iff `r < tanh(I)`, the
neuron fires with probability `CDF_r(tanh(I))` — **the noise distribution IS the
activation function** (inverse sampling theorem, "IST"). Two anchor papers assume
ideal randomness: [1] Bunaiyan et al. (arXiv:2601.18943, configurable p-neurons —
the primitive) and [2] Ghantasala et al. (arXiv:2507.07763, feedforward p-DNNs +
multi-sample inference benefit — the method template). This study asks what neither
does: **how good does the randomness have to be?** Defects injected at the
comparator: bit-depth quantization `k`, bias `b`, serial correlation `ρ`
(Gaussian-copula AR(1), marginal kept exactly Uniform[−1,1] to isolate independence
from activation shape), distribution mismatch, and a real Galois LFSR.

**Claimed headline (verify it!):** a p-DNN tolerates 1–2-bit randomness almost for
free, but the multi-sample gain requires independent draws — ρ=1 erases it, ρ=0.9
defers it past practical sample budgets; STE (noise-aware) training mitigates via
better single-sample accuracy, not defect immunity. Independence > precision.

## 1. Get the code

```bash
git clone https://github.com/pankajgupta/probabilistic-dnn.git
cd probabilistic-dnn
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy, torch, torchvision, matplotlib, scipy, pypdf
.venv/bin/pip install pytest
```

Everything is CPU-only. Reference machine: Apple M3 Pro, 12 cores; total compute for
the full verification below is roughly 45–60 min. MNIST downloads to `./data` on
first use. Read `README.md` (motivation), `HANDOFF.md` (original build brief),
`reports/REPORT.md` (the claims you are checking). The committed `results/*.json` are the
original run's outputs — **do not overwrite them; write your rerun outputs elsewhere
and diff** (e.g. `git stash` your results or copy `results/` aside first).

## 2. Reproducibility ground rules (read before comparing numbers)

- Every experiment logs its seeds in the JSON `_meta`/meta blocks (train seeds 0–2,
  noise seeds 2000–2004, triangle seeds 0–4, E1 seeds in-file).
- **Bit-exact reproduction is NOT guaranteed across machines/torch versions** —
  training uses torch CPU ops whose thread scheduling varies. Expect trained-model
  accuracies within ~±0.3 pp of ours, and everything downstream within the stated
  tolerances. NumPy-only results (E1, E4, RNG tests) should reproduce to high
  precision given the same seeds.
- The verdicts, not the third decimal, are what you are verifying. A discrepancy
  that flips a verdict matters; a 0.1 pp wiggle does not. Report both kinds, but
  distinguish them.

## 3. Verification protocol (run in this order; each step has expected numbers)

### V1 — Unit tests (~1 min)
```bash
.venv/bin/python -m pytest tests/test_pdnn.py -q    # expect: 12 passed
.venv/bin/python tests/test_rng.py                  # expect: all 17 tests passed
```
Covers: E[m]→tanh; copula marginal uniformity at ρ=0.9; lag-1 correlation ≈ρ on the
Gaussian scale; ρ=1 sharing; quantization level centers; seed determinism; S=512
convergence to deterministic logits; LFSR periods 255/65535 exact; PCG64 passes all
stat tests; AR1(ρ=.9) fails only autocorrelation.

### V2 — Triangle ground-truth gate (~1 min quick, ~15 min full)
Quick gate (script your own or use python -c): run
`groundtruth.triangle.gibbs(UniformNoise(seed=1), sweeps=100_000, burn=2_000)`
against `exact_dist()`. Expected: frustrated states ≈0.16566 each, aligned ≈0.00303
each, **TV < 0.01** (we measured 0.0026 at this config; B4's pre-check got 0.00397
with its own seed). The full 1M-sweep 5-seed ideal TV in
`results/e4_cross_task.json` is **0.0009 ± 0.0002**.

### V3 — Baselines + sanity (~5 min)
```bash
.venv/bin/python models/train_baselines.py
.venv/bin/python experiments/sanity_check.py
```
Expected (ours, mean over 3 train seeds; yours within ~±0.3 pp): deterministic test
acc large 97.21%, mid 96.41%, small 95.44%. Sanity (ideal noise, full 10k test):
accuracy climbs monotonically with S; at S=32 the gap to deterministic is ≤0.4 pp
(large 0.15 / mid 0.25 / small 0.38 — the small Jensen gap from two stacked p-bit
layers); the S=1→32 sampling gain is **large 5.40 / mid 13.68 / small 16.96 pp**.
This gain existing is the precondition for E3 meaning anything.

### V4 — E1 activation deformation, H1 (~1 min)
```bash
.venv/bin/python experiments/e1_activation.py
```
Expected: every condition's measured-vs-predicted max-abs-error in **0.004–0.008**
(MC floor √(0.25/20000)=0.0035): ideal .0063, k1 .0077, k2 .0071, k3 .0077,
b0.1 .0070, b0.3 .0074, AR1-ρ0.9 .0077 (vs the IDEAL curve — negative control),
triangular .0047, truncgauss .0051, LFSR8 .0037 (also vs ideal). Anything >0.02
is a real mismatch — investigate before proceeding.

### V5 — E2 precision + bias + LFSR, H2 (~15 min)
```bash
.venv/bin/python experiments/e2_precision.py
```
Expected key numbers (mid width, 3-train-seed aggregate, ±~0.3 pp):

| k  | S=1    | S=32   |   | b    | S=32   |
|----|--------|--------|---|------|--------|
| 1  | 89.78% | 95.89% |   | 0.02 | 96.17% |
| 2  | 85.71% | 96.06% |   | 0.1  | 95.99% |
| 4  | 83.06% | 96.19% |   | 0.2  | 95.40% |
| 8  | 82.57% | 96.17% |   | 0.3  | 94.13% |
| 16 | 82.55% | 96.18% |   | 0.5  | 86.68% |

Verdicts to confirm: **no knee down to k=2**; k=1 costs ≤0.5 pp at S=32 (all
widths); **the S=1 inversion is real** (k=1 beats k=16 by ~+7 pp — coarse noise
helps single samples); bias ≫ quantization at matched displacement; LFSR8/LFSR16 at
S=32 within ~±0.1–0.4 pp of ideal (96.16%).

### V6 — E3 sampling fragility + ECE, H3 headline (~15 min)
```bash
.venv/bin/python experiments/e3_sampling_fragility.py
```
Expected (mid, aggregate; the two internal KEY CHECKS must pass: ρ=0 reproduces the
sanity curve within noise std; ρ=1 flat at the S=1 level within ~0.2 pp):

| ρ    | S=1   | S=4   | S=16  | S=64  | gain (% of ideal) |
|------|-------|-------|-------|-------|--------------------|
| 0.0  | 82.55 | 93.75 | 95.89 | 96.30 | 13.75 pp (100%)    |
| 0.9  | 82.55 | 87.53 | 91.71 | 94.94 | 12.39 pp (90%)     |
| 0.99 | 82.55 | 84.54 | 86.49 | 89.31 | 6.77 pp (49%)      |
| 1.0  | 82.55 | 82.59 | 82.52 | 82.38 | ~0                 |

Headline contrast at S=64: **k=2-independent 96.16% vs ρ=0.9-correlated 94.94%**
(ideal 96.30). Cross-neuron sharing: S=1 drops to ~76.3% but converges to ~96.22%
(different failure mode — gain intact). ECE: rises with S under ideal noise
(~0.05→~0.17), stays ~0.05 at ρ=1 — verify the direction, it's counterintuitive.
**Nuance to verify, not just the slogan:** ρ=0.9 delays rather than caps the gain
(its curve is still rising at S=64); only ρ=1 truly collapses it.

### V7 — E4 triangle cross-task, H4 (~15 min)
```bash
.venv/bin/python experiments/e4_cross_task.py
```
Expected TV (mean±std over 5 seeds): ideal .0009±.0002; **k=1..4 all EXACTLY
0.006068 with std 0** (mechanism: top quantized level 0.9375 < tanh(2)=0.9640 ⇒
aligned states unreachable ⇒ TV = their exact mass 2·e⁻⁴/(2(3+e⁻⁴)); verify the
arithmetic yourself); k=6 .0013; bias monotone .0118 (b=.02) → .2710 (b=.5);
correlation non-monotone plateau .021–.032 for ρ∈[.25,.99]; **LFSR8 .0739±.0043
(~82× ideal), LFSR16 .0135**. The LFSR8-vs-MNIST contrast (fine on one task, 82×
off on the other) is the H4 centerpiece — confirm both halves.

### V8 — E5 STE extension, H5 (~5 min)
```bash
.venv/bin/python models/ste_train.py
.venv/bin/python experiments/e5_ste_robustness.py
```
Expected (2 train × 5 noise seeds; STE nets ~95.1–96.1% stochastic eval): at S=64,
ideal: frozen 96.31 / ste_S1 95.29 / ste_S4 96.28; **ρ=1.0: frozen 82.75 / ste_S1
92.69 / ste_S4 91.05** (+8–10 pp, the one decisive STE win); precision defects: no
established benefit; mechanism claim to scrutinize: STE's edge ≈ its S=1 edge
carried into the regime ρ=1 forces (~constant +9.9 pp across S).

### V9 — Figures & report cross-check (~5 min)
```bash
.venv/bin/python experiments/make_figures.py
```
Regenerates all 9 figures from JSONs. Then spot-check reports/REPORT.md claims against your
rerun JSONs — especially every bolded number above. The original run caught (and
fixed) one aggregation error (a seed-0-only value quoted as the 3-seed aggregate);
assume more such errors could exist and hunt for them.

## 4. What to report back

Produce `VERIFICATION.md` with: (a) a table — each V-step, pass/fail, your numbers
vs expected, tolerance verdict; (b) any verdict-flipping discrepancies, investigated
to root cause; (c) anything in the code you consider a correctness risk even if the
numbers match (independence hygiene between "independent" cells, the copula
implementation, the ECE definition, the logit-averaging choice); (d) your honest
view of whether the headline sentence is supported as worded.

## 5. Known soft spots (attack these first)

1. **Jensen gap**: convergence-to-deterministic is exact only per layer; with two
   p-bit layers the S→∞ limit sits slightly below deterministic. Small here
   (≤0.4 pp) — but is it width-dependent in a way that matters?
2. **S_eff underprediction**: naive effective-sample-size S(1−ρ)/(1+ρ) UNDERpredicts
   measured accuracy at ρ=0.9 (measured S=64 ≈ ideal S≈6–7, predicted ≈3.4). Why?
   Unexplained in the report.
3. **ECE**: 15-bin equal-width ECE on averaged softmax; the rise-with-S is an
   underconfidence artifact of averaging near-one-hot samples. Would NLL/Brier or
   temperature scaling change the story? Single train seed only.
4. **E5 is 2-seed**: the STE deltas are large but the seed pool is thin.
5. **Correlation channels differ across tasks** (across-sample in E3 vs
   Gibbs-update in E4) — the report disclaims equivalence; check we didn't
   accidentally imply it anywhere.
6. **The k=1 S=1 improvement mechanism** (variance reduction) is inferred, never
   measured. A logit-variance decomposition would settle it.

## 6. Further directions (rough priority; propose your own too)

1. **Analytic effective-sample-size model**: variance of AR(1)-averaged logits is
   closed-form; test whether all (ρ,S) cells collapse onto one effective-variance
   curve — and resolve soft spot #2.
2. **Antithetic sampling** (r, −r pairs across samples): negative correlation might
   BEAT iid at small S — would refine the headline to "positive correlation is the
   enemy."
3. **Cheap decorrelation**: XOR-mixing weak streams / per-neuron phase offsets on a
   shared LFSR; how much of the gain does each buy back? (Design-rule material.)
4. **LFSR-driven across-sample correlation on MNIST**: produce the H3 collapse from
   a real generator's reuse pattern, not parametric AR(1).
5. **Measure the inferred mechanisms** (soft spots #6 and the STE carry-through)
   via logit bias/variance decomposition.
6. **Bias calibration loop**: measure comparator offset, subtract, show b=0.2 is
   recoverable.
7. **Fashion-MNIST drop-in** for task-difficulty robustness (CIFAR/CNN/RBM remain
   out of scope).
8. Nonstationary defects: bias drift, telegraph/burst noise.

## 7. Repo map + provenance

| Path | What |
|------|------|
| `pneuron/{neuron,noise,rng}.py` | primitive, defect sources (copula AR(1)!), LFSR + stat tests |
| `groundtruth/triangle.py` | Gibbs sampler, exact dist, TV |
| `models/{net,train_baselines,pdnn,ste_train}.py` | MLP, training, p-DNN wrapper, STE |
| `experiments/e1..e5, sanity_check, make_figures` | one runner per hypothesis |
| `results/*.json` | original outputs, seeds logged (do not overwrite) |
| `plots/`, `reports/REPORT.md`, `reports/report.html` | figures and the report under test |

Provenance: built 2026-07-02/03 in a multi-agent Claude Code session (git history
`cfdb0d8..37f103f` shows the phase gates). Independent verification is the point of
this handoff — trust the JSONs and the code, not the narrative, and say so where
they disagree.

[1] S. Bunaiyan et al., arXiv:2601.18943. [2] L. A. Ghantasala et al., arXiv:2507.07763.
