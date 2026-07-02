# HANDOFF — p-bit Neural Network Noise-Reliability Study

> Engineering handoff for a fresh Claude Code session. This is a 4-week high-school
> summer-research project (SRA Track 1, Probabilistic Computing). Language: Python
> (NumPy + PyTorch). Deliverables: a small codebase, ~5 figures, a paper, a talk.
> Read this whole file before writing code.

---

## 0. One-paragraph summary

We build a **feedforward neural network whose neurons are p-bits** (probabilistic bits)
and study **how the quality of the injected randomness affects reliability**. A p-bit
fires by comparing its input against a random number, so the *distribution* of that
random number *is* the neuron's activation function (the inverse sampling theorem, "IST").
Two recent papers frame the work; **neither studies degraded randomness**, and that is our
contribution. We inject controlled, hardware-motivated randomness defects and measure where
accuracy — and especially the multi-sample accuracy benefit — breaks down.

**The one finding to aim for (headline hypothesis H3):** a p-DNN tolerates *low-precision*
randomness but its multi-sample accuracy gain **collapses under *correlated* randomness** —
i.e. for the sampling advantage, noise **independence** matters more than noise **precision**.

---

## 1. The two anchor papers (read these)

- **[1] Configurable p-Neurons Using Modular p-Bits.** Bunaiyan, Alsharif, **Abdelrahman**
  (our instructor), ElSawy, Cheema, Fahmy, Camsari, Al-Dirini. arXiv:2601.18943 (2026),
  IEEE ISCAS 2026. *Contribution:* decouple the p-bit noise path to get **configurable
  activations** (p-Tanh, p-Sigmoid, p-ReLU) selected by the injected **noise distribution**.
  Validated at single-neuron + 3-neuron logic-gate level. **No trained network, no task.**
  This is our instructors' own paper; the primitive and the "noise distribution = activation"
  idea come from here.

- **[2] Improving deep neural network performance through sampling.** Ghantasala, Li, Jaiswal,
  Behin-Aein, Makin, Sen, **Datta**. arXiv:2507.07763 (2025), npj Unconventional Computing (2026).
  *Contribution:* p-bit neurons work in **feedforward** DNNs (not just Boltzmann machines);
  **averaging a few samples at inference raises accuracy** (on CIFAR-10, 1 sample of a 1-bit
  p-DNN ≈ deterministic, 2 samples beat it, ~10 samples ≈ 3-bit deterministic); it's
  energy-efficient (FPGA MNIST generator). **Assumes ideal randomness.** This is our
  *method template* — follow its recipe.

**Our gap = the intersection neither covers:** a *feedforward p-DNN* (from [2]) whose
*randomness quality* is degraded (our axis), explained through the *IST/configurable-activation*
lens (from [1]), validated against *exact ground truth*.

**Honesty for scoping (important):** building a p-DNN is [2]; configurable activations are [1].
Our claim is ONLY the reliability-under-imperfect-randomness study + the H3 correlation finding.
Do not let the code or the write-up drift into claiming we invented the p-DNN or configurable
activations.

---

## 2. The core primitive (get this exactly right)

A p-neuron output:

```
m = sign( f(I) − r )          # r drawn fresh EVERY forward pass
```

- `f(I)` is the deterministic pre-activation (in [2] it's `tanh(W·x + b)`; you can also use raw `I`).
- `r` is a random draw from a chosen distribution. **This is the injection point for every experiment.**
- Because `m = +1` iff `r < f(I)`, the neuron fires +1 with probability `CDF_r( f(I) )`.
  **So the noise distribution's CDF is the activation function.** This is IST and it is the
  spine of the whole project.

Concrete activation correspondences (verify empirically in Phase 1):
- Uniform `r` on raw input `I` → **ramp** (hard-sigmoid / ReLU-like).
- `m = sign(tanh(I) − r)` with `r ~ Uniform[-1,1]` → fires with prob `(1 + tanh(I))/2`,
  which is a **smooth S-curve (sigmoid-like), NOT a ramp** (the `tanh` pre-squash adds a second
  shaping step). **This is a classic trip-up — do not expect a ramp here.**
- Gaussian `r` → probit S-curve. Logistic `r` → exact sigmoid.

Implement the three sub-operations explicitly (activation / RNG / compare) and make the RNG a
**swappable, seeded object**, not a call into a global/black-box generator. You must be able to
substitute a defective generator at the comparator.

---

## 3. What to build (components)

1. **`pneuron`** — the primitive above, with a pluggable noise source. Vectorized (operate on
   whole layers/batches at once).
2. **Noise-defect library** — the independent variables. Four families:
   - **bit-depth `k`**: quantize `r` to `2^k` levels on its support; sweep `k = 16 … 1`.
   - **bias `b`**: shift the mean of `r`.
   - **correlation `ρ`**: short-period LFSR, or AR(1) `r_t = ρ·r_{t-1} + sqrt(1−ρ²)·ε`, or
     reuse the same `r` across neurons/samples. (This drives the headline H3.)
   - **distribution mismatch**: inject triangular/truncated/Gaussian where uniform was intended.
3. **RNG utilities** — an LFSR, an AR(1) correlated source, wrappers for good baselines
   (NumPy PCG64), plus **statistical tests** (period, autocorrelation, histogram) to validate a
   generator BEFORE plugging it into the network.
4. **Ground-truth harness** — the frustrated-triangle sampler (Section 5). Exact target known
   in closed form; used to *prove* deviations.
5. **p-DNN on MNIST** — small MLP or compact CNN. **Primary training route (do this first):**
   train a normal deterministic net with ordinary backprop, then at inference replace each
   activation with the p-neuron and inject the noise under test (the "add noise to a trained
   model" route in [2]). Frozen weights ⇒ any accuracy change is caused by the randomness alone.
   **Optional extension:** sample-aware training with a straight-through estimator (also in [2]).
6. **Experiment runners + plotting** — one script per hypothesis, all writing figures to `plots/`.

---

## 4. Experiments → hypotheses → target figures

| Exp | Hypothesis | What to run | Target figure |
|-----|-----------|-------------|---------------|
| E1 | **H1** mechanism | Single p-neuron; for each defect, overlay *measured* activation vs *IST-predicted* deformed CDF | Activation-deformation panel |
| E2 | **H2** precision tolerance | MNIST p-DNN; sweep RNG bit-depth; accuracy vs `k` | Precision curve w/ failure knee (expect tolerant to few bits) |
| E3 | **H3** correlation fragility (**headline**) | Accuracy vs # inference samples, for **independent vs correlated** noise | Correlation flattens the sampling gain → "independence > precision" |
| E4 | **H4** task dependence | Same defects on the frustrated triangle; TV distance from exact dist vs defect strength, overlaid with MNIST accuracy sensitivity | Cross-task sensitivity (provable half) |
| — | uncertainty | Multi-sample calibration (ECE) vs noise quality | Calibration curve |

**Metrics:** test accuracy; accuracy-vs-#samples; expected calibration error (ECE);
single-neuron activation fidelity (measured vs predicted); total-variation distance on the triangle.

---

## 5. Ground-truth harness — frustrated triangle (exact, self-contained)

Three p-bits `m1,m2,m3 ∈ {−1,+1}`, antiferromagnetic couplings `J12=J13=J23=−1`, no biases.
Energy `E(m) = m1·m2 + m1·m3 + m2·m3`. Target `π(m) ∝ exp(−E(m))`.

- 2 aligned states `(+,+,+),(−,−,−)`: `E=3`, weight `e^{−3}`.
- 6 mixed ("frustrated") states: `E=−1`, weight `e^{1}`.
- `Z = 6e + 2e^{−3}`.
- **Exact targets:** each frustrated state `= 1/(2(3+e^{−4})) ≈ 0.16565`;
  each aligned state `= e^{−4}/(2(3+e^{−4})) ≈ 0.00303`.

Gibbs update per p-bit: `I_i = Σ_j J_ij m_j` (h=0); `m_i = sign(tanh(I_i) − r)`, `r ~ Unif[-1,1]`.
Reference loop: `J = [[0,-1,-1],[-1,0,-1],[-1,-1,0]]`, ~100k sweeps, ~2k burn-in.

**Validation gate:** with a good RNG, empirical frequencies must converge to the numbers above
(TV → 0). This is the correctness test for the whole primitive. Then degrade the RNG and watch
TV rise — the provable core of H4.

---

## 6. Suggested repo layout

```
pdnn-noise-reliability/
  HANDOFF.md                 # this file (or rename CLAUDE.md)
  requirements.txt           # numpy, torch, torchvision, matplotlib, scipy
  pneuron/
    neuron.py                # p-neuron primitive, pluggable noise
    noise.py                 # defect families: bitdepth, bias, correlation, mismatch
    rng.py                   # LFSR, AR(1), PCG64 wrapper + stat tests
  groundtruth/
    triangle.py              # sampler + exact distribution + TV distance
  models/
    net.py                   # small MLP/CNN for MNIST
    pdnn.py                  # wrap trained net -> p-neuron activations at inference
  experiments/
    e1_activation.py  e2_precision.py  e3_sampling_fragility.py  e4_cross_task.py
  plots/
```

---

## 7. Build order (dependency-sensible; do NOT jump ahead)

1. **Primitive + triangle validation.** Prove correctness against the exact numbers in §5. Nothing
   else is trustworthy until TV→0 with a good RNG. *(Course milestone: p-bit lecture ~Jul 9.)*
2. **Noise-defect library + RNG stat tests.** Validate each generator's period/autocorrelation
   before use.
3. **E1 single-neuron activation deformation (H1).** Measured vs IST-predicted curves.
4. **Baseline MNIST net → wrap as p-DNN.** Reproduce the multi-sample accuracy benefit from [2]
   with a *good* RNG first (sanity check).
5. **E2 precision sweep (H2).**
6. **E3 sampling fragility — independent vs correlated (H3).** The headline.
7. **E4 cross-task TV-vs-accuracy (H4).**
8. **Figures + write-up.** Multiple seeds, error bars.

---

## 8. Pitfalls & correctness checks (read before coding each part)

- `(1+tanh(I))/2` is a **smooth S-curve, not a ramp.** A true ramp needs uniform noise on the
  **raw** (un-`tanh`'d) input. Expecting a ramp from the `tanh` form = phantom "bug".
- **Inject at the comparator.** Draw uniforms from an explicit seeded generator and compare to
  `f(I)` yourself; don't call a black-box Bernoulli — you can't corrupt what you can't reach.
- **Freeze trained weights** across all noise conditions (primary route). Any change must be
  attributable to randomness, not retraining.
- **The multi-sample benefit needs INDEPENDENT draws.** That independence is precisely what H3
  removes — so the "correlated" condition is implemented by deliberately sharing/serially-correlating
  the RNG state. Don't accidentally do this in the *independent* baseline (would corrupt E2/E4).
- **Seeds + error bars.** Training/sampling variance can swamp the effect; run several seeds per
  condition and plot spread.
- **MNIST small first** (subset of classes or downscaled) for fast iteration; scale up once the
  pipeline is stable. **Do NOT chase CIFAR-10 or build an RBM** — out of scope.
- Keep the network fixed/small so results are interpretable.

---

## 9. Environment

- Python 3.x, `numpy`, `torch`, `torchvision` (MNIST), `matplotlib`, `scipy`.
- CPU is fine (small nets, small MNIST). No GPU assumptions.
- Everything seeded and reproducible; log seeds with results.

---

## 10. Open questions to resolve with the TA (flag to the human — don't block on these)

1. **Training route:** add-noise-to-a-trained-model (recommended, isolates the noise variable) vs
   sample-aware straight-through training. Default to the former; build the latter only if time allows.
2. **Headline:** H3 (correlation-vs-precision) vs H4 (cross-task reliability threshold). Both are
   worth producing; confirm which leads the paper.
3. **Scope of the configurable-activation angle from [1]:** is "choose the noise distribution to get
   different activation *families* and test whether the choice matters for the task" in scope, or
   reserved for the instructors' own follow-up? Affects whether E1 also sweeps *distribution family*,
   not just defects.

---

## 11. Definition of done (for the project, so the agent knows the target)

- Primitive validated against the exact triangle distribution (TV→0 with good RNG).
- Reusable defect library covering the four families, with generator stat-tests passing.
- The five figures in §4 produced with error bars.
- One honest headline sentence supported by the data — e.g. either
  *"p-DNNs tolerate coarse randomness but the sampling advantage collapses under correlation
  (independence > precision),"* or, if that's not what the data shows, an equally honest alternative.
- Paper (SRA template) + 10–13 min capstone talk. Cite [1] and [2] as the anchors; [1] includes
  the course instructor and a TA as coauthors.

---

## 12. Companion documents (from the planning session)

- `Proposal_Noise_Quality_Reliability_pDNN.pdf` — full proposal (methodology, phases, results-to-drive-toward).
- `Related_Work_pbit_noise_activation.pdf` — related-work map + gap table (includes the RBM cluster:
  Zeng et al. 2024 is related but studies RBM activation deformation, not feedforward-p-DNN noise quality).
