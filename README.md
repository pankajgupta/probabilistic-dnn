# How Good Does the Randomness Have to Be?

Noise quality and the reliability of feedforward probabilistic (p-bit) neural networks.

## The problem

A **p-bit neuron** fires by comparing its input against a random number:

```
m = sign( tanh(I) − r )        # r drawn fresh every forward pass
```

Because `m = +1` exactly when `r < tanh(I)`, the neuron fires with probability
`CDF_r(tanh(I))` — **the distribution of the injected randomness *is* the activation
function** (the inverse sampling theorem). This is attractive for hardware: cheap
physical noise sources (sMTJs, LFSRs) replace expensive deterministic arithmetic, and
averaging a few stochastic forward passes at inference recovers — or beats —
deterministic accuracy.

Two anchor papers frame the setup. Bunaiyan et al. [1] introduce configurable
p-neurons whose activation is selected by the injected noise distribution. Ghantasala
et al. [2] show p-bit neurons work in feedforward DNNs and that multi-sample
averaging at inference raises accuracy. **Both assume ideal randomness.** Real
generators are not ideal: they have finite precision, offset bias, serial
correlation, and wrong-shaped distributions.

**This repo asks the question neither paper answers: how good does the randomness
actually have to be?** We freeze a conventionally trained MNIST MLP, replace its
activations with p-neurons at inference (so any accuracy change is attributable to
the randomness alone), inject controlled hardware-motivated defects at the
comparator, and measure where accuracy — and especially the multi-sample benefit —
breaks down. An exact-ground-truth sampling task (the frustrated antiferromagnetic
triangle, whose target distribution is known in closed form) anchors the softer
classification metrics.

## Headline finding

> A feedforward p-DNN tolerates even 1–2-bit randomness with almost no accuracy
> loss, but the multi-sample advantage requires independent draws: fully correlated
> noise (ρ=1) erases it, high correlation (ρ=0.9) defers it well past practical
> sample budgets, and noise-aware (STE) training mitigates but does not remove the
> penalty — **independence across samples, not precision, is the binding
> constraint.**

Supporting results: a 2-bit *independent* generator beats a full-precision
*correlated* one (96.16% vs 94.94% at 64 samples); bias — a systematic, non-zero-mean
defect — is the per-draw corruption averaging cannot undo; and the same 8-bit LFSR
that is statistically indistinguishable from ideal on MNIST sits ~82× above the ideal
total-variation floor on the exact sampling task, so "good enough" is task-dependent.

Full write-up with all nine figures: [`REPORT.md`](REPORT.md) or the self-contained
[`report.html`](report.html).

## Repo layout

```
pneuron/       p-neuron primitive, noise-defect library, LFSR + RNG stat tests
groundtruth/   frustrated-triangle Gibbs sampler + exact distribution + TV distance
models/        MLP, training scripts (deterministic + STE), p-DNN inference wrapper
experiments/   one runner per hypothesis (e1–e5), sanity check, figure generation
results/       all experiment outputs (JSON, seeds logged) + model checkpoints
plots/         the nine report figures (regenerable from results/ JSONs)
tests/         unit tests for the noise wrapper and RNG suite
```

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python models/train_baselines.py        # 9 checkpoints, ~1 min CPU
.venv/bin/python experiments/sanity_check.py      # multi-sample benefit, ideal RNG
.venv/bin/python experiments/e2_precision.py      # (similarly e1, e3, e4, e5)
.venv/bin/python experiments/make_figures.py      # rebuild plots/ from results/
```

Everything is seeded; every number in the report is traceable to a `results/*.json`
field. CPU-only, small nets — the full experiment grid is under an hour on a laptop.

## References

[1] S. Bunaiyan et al., *Configurable p-Neurons Using Modular p-Bits*,
arXiv:2601.18943 (2026); IEEE ISCAS 2026.

[2] L. A. Ghantasala et al., *Improving deep neural network performance through
sampling*, arXiv:2507.07763 (2025); npj Unconventional Computing (2026).

The p-DNN method is [2]'s and the configurable-activation primitive is [1]'s; this
repo's contribution is only the reliability-under-imperfect-randomness study.
