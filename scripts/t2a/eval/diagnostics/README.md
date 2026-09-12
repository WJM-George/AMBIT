# Spatial-CoT renderer diagnostics

These scripts test causal control rather than treating training loss as a
renderer-quality metric.

- `calibrate_spatial_anchor.py` compares the deterministic ScenePlan anchor to
  retained validation FOA and optional frozen-VAE reconstructions. It reports
  single-source and overlap frames separately and audits all signed axis
  permutations.
- `diagnose_spatial_conditions.py` performs a same-noise `Plan x anchor`
  rotation intervention and tests correct, absent, and foreign previous-FOA
  contexts.
- `run_spatial_condition_diagnostics.sh` runs step0 and step300 in the fixed
  family274 -> family8179 order.
- `summarize_spatial_conditions.py` produces the cross-family causal summary.
  A previous-FOA context counts as capture only when it moves the same-noise
  output at least one degree toward that supplied context relative to the
  `context=none` control; static reference similarity is not treated as a
  causal effect.
- `diagnose_semantic_conditions.py` holds noise and geometry fixed while
  independently swapping caption semantics and ScenePlan event semantics.
  `score_semantic_conditions.py` scores those interventions against both the
  correct and donor captions with CLAP and retains target-relative latent,
  waveform, multi-resolution spectral, spatial-field, and target-silence
  diagnostics in the same report. `score_semantic_speech.py` adds
  transcript-specific ASR evidence. CLAP is an event/music/speech semantic
  diagnostic; it does not replace listening checks. When source-location stems
  are present, the ASR report also records which persistent trajectory contains
  each reference transcript.
- `score_source_location_semantics.py` uses the pinned ScenePlan to demix
  target and generated FOA into activity-gated directional stems. It reports
  generated-to-target-source and source-caption CLAP matrices, but only counts
  assignments whose retained target stem is itself discriminable. This closes
  the gap between "the mixture contains the event" and "the event occupies the
  correct persistent source trajectory." It accepts both semantic-intervention
  reports and canonical one-turn checkpoint `RESULT.json` files; the fixed
  family capacity sweep runs it automatically after generation.
- `score_source_presence_openflam.py` adds target-calibrated frame-level event
  presence and inactive-window leakage for each ScenePlan source. It is an
  optional non-commercial research evaluator under the Adobe Research License;
  its isolated package/weight paths are recorded in every output report and it
  is never imported by training.
- `diagnose_source_binding.py` holds the global 4-D field anchor fixed and
  swaps persistent source slots, exposing whether source identity reaches the
  Renderer rather than collapsing into mixture steering. When explicit
  source-semantic fusion is enabled, its residual intervention uses the same
  exact Qwen source-span summaries as normal rendering.

All launchers require an explicit output root and write their protocol next to
the results. After the ordered spatial sweep, expensive text/semantic
diagnostics follow the common eligible checkpoint with the lowest combined
angular score. If promotion is already blocked, they instead inspect the
least-bad observed step for failure analysis without changing that BLOCK.
Promotion decisions must use the validation latent root explicitly.
