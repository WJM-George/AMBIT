# Transfusion OPSD

Physical GPU0 develops Generation AR＋DiT; GPU1 develops Editing AR＋DiT.
Both use the common decision-gradient and bounded joint-update kernels.
GRPO is stopped and archived. A complete saved checkpoint at 20k updates of the
current structured Editing joint SFT is now saved and has passed the native
loader/adapter boundary check. The user does not require waiting for all 50k
updates. Check actual editing/content preservation before OPSD training. Editing SFT and
other tasks' evaluation are untouched.

[Current method and evidence](${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace/docs/transfusion_opsd.md) · [Editing 20k milestone](${AMBIT_CKPT_ROOT}/transfusion_opsd/editing_structured_20k_opsd_pilot_v1/README.md)

| Role | Current components |
| --- | --- |
| Complete native model | `event_native_policy`, `generation_event`, `event_dit_precision`, `event_qwen_norm_runtime` |
| Native probabilities and actual-execution teachers | `native_stochastic_policy`, `native_prefix_teacher`, `native_timing_execution`, `native_reward_execution_teacher` |
| Decision-to-execution training connection | `native_decision_condition`: exact hard forward with stopped alternative encodings and declared conditional-support ST derivative to native logits |
| Credit assignment | `native_decision_credit`: routes total-repair derivative to logits, excludes direct executor derivatives of this term; ordinary plan-matched residual losses train DiT |
| Bounded native behavior update | `guarded_joint_update`: one Adam proposal, finite declared positive group scales, actual planning validation, in-memory rollback; original moments with explicitly scaled parameter displacement |
| Native timing intervention | `native_timing_probe`: a native categorical intervention, local decoder scope, no production inference change |
| Joint objectives and retention | `native_joint_objectives`, `native_velocity_distillation`, `native_decision_retention`, `native_paired_rf`, `identity_preserving_kl`, `auditory_retention` |
| Native FOA-latent evaluation | `generation_output_observer`, `native_latent_clap`, `native_clap_level_content`, `native_clap_text_cache`, `lexical_content_evidence` |
| Request constraints and actual output | `event_rewards`, `request_coarse_spatial_reward`, `request_semantic_time_reward`, `event_counterfactual`, `native_execution_identity` |
| Editing 20k interface | `editing_clap44_adapter`: native source/slot/plan/gain conditioning; `editing_native_decision`: actual student-prefix logits and grammar/task-checked alternatives for the common credit kernel. Native audio baseline and local gradient connection verified; no Editing OPSD update or gain claim |
| Model-only recoverable controls | `native_model_overlay`: parent-bound native parameters, no optimizer continuation claim |

V68 completed four joint updates per connected/detached arm and new-request
confirmation. V69 completed matched RF-only and AR＋RF controls plus discrete
branch interventions. There are small actual lexical gains, with retained
individual failures and a slight semantic tradeoff. C0 remains unchanged.

V70 expanded the native actual-execution teachers to three requests and measured
all legal onset/offset recombinations at the original durations. Both arms
stopped after two updates when a taught request's native duration left the
measured plan set. Content protection failed; no endpoint was stored or new
confirmation opened. Changed-duration normal outputs have different reference
noise geometry from C0; fixed-C0-plan comparisons remain paired. Refreshed
executor feedback was measured but did not produce a new training round.
[V70 findings](${AMBIT_CKPT_ROOT}/transfusion_opsd/generation_event_coupled_v70_20260911/NATIVE_JOINT_UPDATE_FINDINGS_V1.md).

V71 controls the observed Generation native-duration drift using actual finite
behavior checks. Development CTC improves slightly; new-request Whisper improves
but independent CTC regresses slightly. Confirmation fails its declared content
condition; no scale-up or C0 replacement. All 108 development/confirmation plans
are unchanged, so branch interaction is zero. Keep the update fix and local
output evidence, not the unchanged timing-teacher recipe.
[V71 closeout](${AMBIT_CKPT_ROOT}/transfusion_opsd/generation_event_coupled_v71_20260911/UNIFIED_BRANCH_FINDINGS_V1.md).

Editing20k produces 64 normal/teacher/control audios on eight existing training
rows. Four speech plans have wrong transcripts; extra verified-content text
does not fix them, although source-conditioned audio sometimes preserves the
words. Do not qualify teachers solely from audio scores or preserve false plans.
One correct sound request at two visited states verifies the common ST kernel
reaches native token logits, slot operation and the shared Transformer with
bitwise hard-forward parity. This paired RF gradient probe makes zero updates
and is not a verified self-distillation teacher.
[Unified method](${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace/docs/transfusion_opsd_unified.md).

The new decision derivative passes actual-state and CPU checks but has no
proven added normal-generation benefit over detachment. The native Generation
timing head uses request context, not the shared Transformer's AR query; do not
claim every planning decision or this entire credit path traverses that backbone.
The proxy is biased and its routing is explicitly not the full gradient of the
sum of all displayed diagnostic scalars. Do not repeat the unchanged recipe.

RF-only improves execution under fixed C0 plans but its normal planning drift
hurts the combined result. The measured AR-teacher contribution mainly avoids
that drift. DiT repair has local lexical benefit and semantic tradeoffs. This
does not establish that refreshed DiT taught AR to exploit a new capability.
[Matched findings and limitations](${AMBIT_CKPT_ROOT}/transfusion_opsd/generation_event_coupled_v69_20260911/MATCHED_COMPONENT_FINDINGS_V1.md).

Current native FOA VAE-latent CLAP is pinned to additional 10k, separate from
original 20k initialization and from the Editing joint-training 20k milestone.
Keep historical 40k reports unchanged. All shared trainable parameters appear
only once in a joint optimizer; Qwen, CLAP and VAE observers remain frozen.

Useful older teacher/bridge/retention components remain available; not all are
active research routes. Original evidence and retirement requirements remain in
[history](${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace/docs/transfusion_opsd_history.md) and
[source retirement](${AMBIT_CKPT_ROOT}/transfusion_opsd/generation_event_coupled_v67_20260911/retirement_v1/RETIREMENT.md).
