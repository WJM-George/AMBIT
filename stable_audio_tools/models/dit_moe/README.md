# P10-v11 Dense DiT upgrade staging

This directory contains the default-off DiT upgrade that should be evaluated
after P11 is complete.  It is an upcycle of the canonical P10-v11 150k Dense
DiT, not a replacement architecture.

## Chunk-MoE

Only Transformer blocks 11--14 receive a sparse delta beside their existing
Dense SwiGLU FFN:

```text
FFN(x) = native_dense_ffn(x) + 0.5 * top2_chunk_delta(x)
```

- Four width-1024 delta experts; output projections are initialized to zero.
- One route per four latent frames (about 92.9 ms at a 1024-sample hop).
- A chunk restarts whenever any of the four ScenePlan activity tracks changes.
- The prior uses timestep, active-source caption summaries, and source IDs; the
  evidence path uses post-attention audio hidden states.  A learned conflict
  gate mixes their routing logits.
- Dispatch is tensorized over chunks/tokens.  The only Python loop is the fixed
  four-expert loop, and even unselected experts remain in the autograd graph for
  `ddp_static`.
- Routing always acts on a complete hidden token.  FOA/latent channels are never
  assigned to different experts.

The native `TransformerBlock.ff` object and all of its state-dict names remain
unchanged.  Loading P10-v11 therefore leaves only `sceneplan_moe.*` parameters
missing, exactly as intended.  At the production dimensions this MoE arm adds
73,535,524 trainable parameters (50,380,800 in experts and 23,154,724 in the
four routers); it therefore needs an explicit memory/throughput gate, not just
a quality comparison.

## Soft-block caption attention

Every caption token stays visible to the native global cross-attention.  The
adapter builds two binary frame-to-token maps from existing ScenePlan metadata:

- active source matches event/speaker-description tokens;
- active source matches exact-transcript tokens.

Each selected Transformer layer owns two bounded scalar score-bias gates,
`max_bias * tanh(raw_gate)`.  Both raw gates start at zero.  Role `0` remains a
global-only token and role `-1` is the explicit CFG-unknown state; neither gets
a local bias.  No word timestamp, forced aligner, duration prediction, or hard
attention mask is introduced.

## Experiment arms

All configs inherit the same 64-channel FOA latent, 256-channel ScenePlan 4+4,
1024-wide, 15-block P10-v11 contract:

- `qwen35_0p8b_300m_model_sceneplan_44_upgrade_v12_base.json`
- `qwen35_0p8b_300m_model_sceneplan_44_softblock_v12.json`
- `qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_v12.json`
- `qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_softblock_v12.json`
- `qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv2_v12.json`
- `qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv2_softblock_v12.json`

The original `chunkmoe_v12` pair is retained to reproduce the router-v1
step-500 diagnostic.  New pilots use `routerv2`: lower early load-balance
weight, temperature-sharpened Top-2 routing, a bounded prior/evidence gate, and
raw conflict-logit regularization plus collapse diagnostics.

Start each non-base arm from the same canonical checkpoint using the model-only
warm-start path (`--pretrained-ckpt-path ... --pretrained-route-weights ema`).
Do not use Lightning full-state resume: optimizer, scheduler, and EMA must be
fresh because the destination architecture contains new parameters.

The implementation reports load-balance loss, per-expert dispatch, router
entropy, conflict-gate mean, chunks per layer, routed chunk evaluations, and
active MoE layer count.  The
contract tests cover bit-exact step-zero upcycling, source-boundary chunking,
padding exclusion, explicit CFG `-1` roles, and gradients through attention,
experts, and routing.

## Suggested execution order

1. Run the base and attention-only arms first; the latter adds only 30 scalar
   gates and isolates whether source-local caption scores help at all.
2. Before training MoE, measure one real 432-frame and one 648-frame
   forward/backward under the intended per-GPU batch size.  Record peak memory,
   examples/second, expert dispatch, entropy, and gate distribution.
3. Compare Dense continuation, attention-only, MoE-only, and combined arms from
   the same 150k EMA checkpoint with the same rows, seeds, optimizer-step budget,
   CFG law, and frozen evaluation panels.
4. Evaluate WER/UTMOS, CLAP/FAD, DOA, activity IoU, silence, and multi-source
   binding.  Do not promote an arm from loss or routing balance alone.  Run the
   combined arm last, after the two single-module effects are attributable.
