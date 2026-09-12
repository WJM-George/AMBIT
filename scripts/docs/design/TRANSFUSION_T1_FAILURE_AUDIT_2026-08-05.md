# Transfusion T1 failure audit (2026-08-05)

## Outcome

The 50k `t2a_transfusion_t1_invented15` listen set was generated with a
training/inference mismatch. The checkpoint is not intrinsically collapsed and
the frozen VAE is not the cause.

T1 trains both continuous modalities at one shared rectified-flow time:

```text
Qwen prefix -> trajectory(t) -> FOA latent(t)
```

The old evaluator instead ran the upstream sampler twice:

```text
Qwen prefix -> trajectory(t: 0 -> 1)
Qwen prefix -> clean trajectory(t=1) -> FOA latent(t: 0 -> 1)
```

The second sequence never occurs in T1 training. T1 must integrate trajectory
and latent together with the same time while retaining the causal attention
order. A future model trained with clean previous-stage prefixes (the Spatial-
CoT Renderer) must use sequential sampling instead.

## Evidence

### Listen-set collapse

Across the 15 different prompts and seeds:

| Output | Mean absolute waveform correlation | Mean spectral cosine |
|---|---:|---:|
| T1 50k, old sequential sampler, CFG 6 | 0.898 | 0.9997 |
| DiT 50k | 0.0066 | 0.5951 |

Therefore the failure cannot be explained by comparing a 50k Transfusion
checkpoint with a later DiT checkpoint.

### Training curve

Median T1 losses from the W&B offline history:

| Steps | total | trajectory flow | FOA-latent flow |
|---|---:|---:|---:|
| 10k-20k | 1.5285 | 0.1587 | 1.3673 |
| 40k-50k | 1.4150 | 0.1406 | 1.2496 |
| 60k-70k | 1.4016 | 0.1387 | 1.2354 |
| 70k-73k | 1.3959 | 0.1386 | 1.2283 |

The curve improves slowly and is not numerically divergent. It did not explain
the near-identical generated waveforms by itself.

### Fixed-time checkpoint probe

Using the 72,776-step EMA checkpoint on a real training batch:

| Condition presented to latent block | Latent flow MSE |
|---|---:|
| shared trajectory/latent time, low-time region | about 1.03-1.14 |
| clean trajectory prefix (`t_traj=1`) | about 2.06-2.07 |

The clean-prefix result is approximately the unlearned baseline. This directly
identifies the sequential sampler as out-of-distribution for joint T1.

### CFG range

The training-cache latent standard deviation is about 1.02. With the corrected
structural CFG prompt but the still-mismatched sequential sampler:

| CFG | Generated latent RMS |
|---:|---:|
| 1 | 1.06 |
| 3 | 3.29 |
| 6 | 7.83 |

CFG 6 sends the frozen VAE a severely out-of-distribution latent. T1 now
defaults to CFG 1.5.

### Matched coupled sampler

For two different music prompts at 72,776 steps and 30 midpoint steps:

| Sampler | CFG | Cross-prompt waveform correlation | Latent RMS |
|---|---:|---:|---:|
| old sequential | 1 | 0.968 | 1.06 |
| matched joint-coupled | 1 | 0.014 | 1.05-1.06 |
| matched joint-coupled | 1.5 | 0.016 | 1.06-1.08 |

The joint samples also recover music-like RMS and crest-factor ranges comparable
to the DiT listen samples.

The complete 15-prompt joint listen set at step 72,776 gives mean pairwise
waveform correlation 0.0057 (DiT: 0.0044). Its category-conditioned spectral
statistics track the DiT split rather than the old single-texture collapse:

| Category | T1 joint spectral centroid | DiT spectral centroid |
|---|---:|---:|
| music | 391 Hz | 670 Hz |
| sound | 710 Hz | 1009 Hz |
| speech | 8503 Hz | 8919 Hz |

These are diagnostic distribution statistics, not perceptual quality metrics,
but they verify that the corrected sampler recovers prompt/category-dependent
outputs and keeps latent RMS near the VAE training range.

## Code contract

- `continuous_traj`: `shared_uniform` training + `joint_coupled` sampling.
- `spatial_cot`: alternating clean-prefix Renderer training + `sequential`
  sampling.
- CFG unconditional prompts preserve modality META/shape/SOM/EOM control tokens
  and null only the semantic text/plan prefix.
- Configuration validation rejects a route whose training-time and sampling-
  time contracts disagree.

The old 72,776-step T1 checkpoint remains resumable because these corrections
change sampling, not parameter shapes or the training objective.
