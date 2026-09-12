# Text-to-audio model configs

`t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json`
is the only canonical P11 planner config. Its runtime identity is
`model_type=sceneplan_p11_v4` and `route_id=sceneplan_p11`; no compatibility
alias or legacy P11 fallback is launchable.

## Current mainline

P11 has three single-turn views over one shared Qwen3.5-0.8B planner and one
non-overlapping semantic/numeric state boundary:

```text
Generation:    raw text -> SceneSketch + ExecutionState -> ScenePlan
Understanding: FOA latent + CLAP + gated reliable ASR -> posterior sketch/state -> ScenePlan
Editing:       instruction + current plan -> DeltaSketch + DeltaExecutionState -> atomic patch
```

Only the deterministic assembler may create a complete ScenePlan. The frozen
P10-v11 step-150000 `ScenePlan -> DiT -> FOA` renderer remains external; P11
does not contain or jointly train another waveform renderer.

The current idea-validation route uses manifest schema v6, ScenePlan codec v4,
the 648-frame/15-second P10 envelope, and the frozen pair-aware 810-row pilot
curriculum. Training uses batch 8 per GPU, BF16, EMA, ordered non-shuffling
sampling, and full-state resumable checkpoints.

## Operations

All active build, validation, training, and evaluation commands are documented
in `scripts/t2a/README.md`. The canonical entrypoints are:

```bash
scripts/t2a/data/run_sceneplan_p11_semantic_cache.sh train
scripts/t2a/train/run_sceneplan_p11.sh preflight
scripts/t2a/train/run_sceneplan_p11.sh pilot
.venv/bin/python scripts/t2a/eval/evaluate_sceneplan_p11_v4_challenge.py --help
```

The launcher exposes only the canonical candidate and two required comparison
baselines; inheritance-only and negative-ablation configs are not launch
targets. Retired route hashes live in
`artifacts/sceneplan_p11/p11_active_tree_retirement_manifest_v2_20260902.json`.
