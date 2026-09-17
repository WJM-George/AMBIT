# Transfusion OPSD

Post-training on-policy self-distillation for the shared AR + DiT Transformer.
The current line is **Editing**: STE off, hidden request-side GT, fixed 40k
reference retention, same-plan RF teachers, and paired FOA auxiliary loss.
Generation keeps the same kernels; it is not the active recipe.

See [docs/OPSD.md](../../../../docs/OPSD.md) for the method, batch recipe, and
evaluation contract.

| Role | Modules |
| --- | --- |
| Joint load / stream | `editing_stream`, `scripts/t2a/experiments/ar_structured_v1` |
| Native plans | `native_token_alignment`, `native_greedy_fastpath` |
| Request binding | `editing_request_constraints`, `request_grounded_text_retention` |
| Fixed-reference hold | `editing_spatial_retention`, `native_coarse_choice_retention` |
| Teachers | `execution_teacher_selection`, `native_paired_rf`, `supported_teacher` |
| Paired FOA / space | `editing_binaural_retention`, `foa_spatial_field_repair` |
| Nine-metric panel | `editing_nine_metrics`, `top_checkpoints` |
| Historical ST credit | `native_decision_condition`, `native_decision_credit` (unused in the current recipe) |

STE is off: DiT residual loss does not enter AR logits through discrete argmax.
Shared parameters still train once. Frozen Qwen, VAE, and CLAP stay frozen.

A bidirectional planner–renderer loop is not established. Do not treat a
development-panel win as a replacement for the released 40k baseline.
