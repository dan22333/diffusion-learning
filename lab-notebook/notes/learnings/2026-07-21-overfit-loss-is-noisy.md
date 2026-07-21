# Overfit-one-batch loss is a noisy single-sample estimate
Date: 2026-07-21
Status: supported
Source:
- [Phase 0- overfit experiment](../../../experiments/phase0-overfit/README.md)

## Claim

The per-step DDPM training loss is a high-variance, single-sample Monte-Carlo
estimate (one random timestep + one noise draw per image), so it bounces even
while the model is converging. Judge "did it overfit?" on the trend / a
low-variance eval (loss averaged over many timestep draws), never on one step.

## Evidence

In the Phase 0- run the per-step loss reached ~0.017 at its min but spiked back to
~0.29 on individual later steps; the averaged eval loss was stable and low. The
overfit unit test only passed reliably after switching its assertion from
`losses[-1]` to a 50-draw averaged eval loss.

## Why it matters

Directly shapes how you assert correctness in tests and how you read a loss curve.
Reusable in every later phase (distillation/quantization ask "did quality drop?"
— same need for a low-variance measurement, not a noisy point sample).

## Confidence

High

## Caveats

Variance shrinks with larger batches and with more timestep samples per eval.

## Follow-up

## Related

## Relationships
