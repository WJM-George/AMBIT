# scripts/

AMBIT helper scripts, grouped by purpose. Everything lives in a
typed subfolder — there are no flat top-level scripts, so always reference the
full path (e.g. `scripts/vae/eval/run_vae_sweep.sh`).

## Layout

```
scripts/
├── README.md                 ← this file
├── setup_uv.sh               ← env bootstrap (stays top-level)
├── _repo.py                  ← shared repo-root helper for nested scripts
├── docs/                     ← runbooks & design notes
├── vae/
│   ├── eval/                 ← heldout / FOA recon / sweep / compare / ablation
│   ├── train/                ← 8-GPU launch config builders
│   └── loss/                 ← phase/SCM loss weight calibration + sanity tests
├── t2a/                      ← text-to-audio eval / Qwen checks
├── v2a/                      ← video-to-audio demos
└── utils/                    ← one-off migration / ckpt helpers
```

## Quick map

| Need | Go to |
|---|---|
| VAE checkpoint sweep / heldout metrics | `vae/eval/run_vae_sweep.sh`, `vae/eval/summarize_vae_sweep.py` |
| Mixed-15 FOA recon export + table | `vae/eval/export_vae_checkpoint_series_foa.py`, `vae/eval/build_vae_mixed15_manifest.py` |
| Decode pre-encoded latents | `vae/eval/decode_latents_4ch.py` |
| Matched-step ablation compare (base vs phase+SCM) | `vae/eval/compare_abl_850k.py` |
| Build HF-overshoot / decay train launches | `vae/train/build_vae_hf_*.py` |
| Calibrate new phase/SCM loss weights + sanity tests | `vae/loss/calibrate_new_loss_weights.py`, `vae/loss/test_new_spatial_losses.py` |
| Phase / spatial loss design notes | `docs/design/LOSS_DESIGN_SPATIAL_PHASE.md`, `docs/design/LOSS_THEORY_GROUNDING.md` |
| Overall pipeline / UV how-to | `docs/pipeline/WORKFLOW.md`, `docs/runbooks/RUN_UV.md` |
| Spatial-CoT method and exact-pair contract | `docs/design/SPATIAL_COT_AUDIOCHAT_METHOD.md` |
| Build Spatial-CoT artifacts/catalog | `t2a/data/prepare_spatial_cot_v1_artifacts.sh` |
| Render + stream-preencode Spatial-CoT data | `t2a/data/run_spatial_cot_data.sh` |
| Train final Spatial-CoT model | `t2a/train/run_t2a_spatial_chat_500m_8gpu.sh` |
| Transfusion OPSD | [`docs/OPSD.md`](../docs/OPSD.md), launcher `t2a/rl/launch_editing_opsd_repaired_fresh.py` |
| Other Qwen T2A training/eval helpers | `t2a/` |
| V2A demos | `v2a/` |

## Common commands

```bash
# From the repository root

# VAE heldout sweep
MODE=heldout GPU=4 bash scripts/vae/eval/run_vae_sweep.sh

# Mixed-15 FOA series export
uv run python scripts/vae/eval/export_vae_checkpoint_series_foa.py --help

# Spatial-CoT nested data gates (do not start the 1M build implicitly)
scripts/t2a/data/prepare_spatial_cot_v1_artifacts.sh
scripts/t2a/data/run_spatial_cot_data.sh status

# Final full-depth optimizer-allocation smoke
BENCHMARK=1 MAX_STEPS=4 BATCH_SIZE=6 \
  scripts/t2a/train/run_t2a_spatial_chat_500m_8gpu.sh
```
