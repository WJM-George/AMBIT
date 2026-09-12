# Retired P10 routes

As of 2026-08-31, the only active P10 executor is:

`sceneplan_dit_v11_semantic_v2_protected_resume_150k/checkpoints/epoch=48-step=150000.ckpt`

In the cleaned AMBIT source distribution, the retired launcher paths only print
a retirement notice and exit with status 64. `P10_ALLOW_RETIRED_ROUTE=1` no
longer enables them. Their full historical implementations remain in the
original workspace; source paths and hashes are recorded in
`docs/SOURCE_MANIFEST.json`. See `docs/CLEANUP.md` for the cleanup scope.

Their checkpoints, logs, evaluations, and samples were indexed in:

`/mnt/sdc/stable-audio-tools-workspace/artifacts/archive_manifests/P10_P11_CHECKPOINT_ARCHIVE_20260831.md`

Cold artifacts live at:

`/mnt/sdb/model_archives/p10_pre_v11_20260831`

Do not use a retired route as a P11 executor or as the target capability contract.
