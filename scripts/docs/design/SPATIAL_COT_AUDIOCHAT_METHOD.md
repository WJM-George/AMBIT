# Spatial-CoT: AudioChat-style persistent state for metric-3D FOA

Status (2026-08-10): the final method, train1m family store, retained test2k /
validation10k splits, and fail-closed full-data audits are complete. The
production-shaped 8-GPU gate selected batch 6/GPU. A fixed-panel
Boundary-Anchored Transfusion Forcing comparison is running from the exact
step-2560 warm start before promotion to the 5k pilot. Promotion now includes
source-disjoint Planner/Understanding counterfactual and free-decode evidence,
so the very small bounded-overfit CE is not treated as generalization by
itself. For the current code authority and maintenance boundary, see
`SPATIAL_COT_CODE_STRUCTURE.md`.

## 1. Final decision

We keep AudioChat's core idea—persistent source state, turn-level diffs,
previous-audio context, and independently noised audio spans—but implement it
inside our frozen-Qwen + Transfusion + 4-channel FOA system:

```text
original caption / edit instruction
              │
        frozen Qwen3.5-0.8B
          ┌───┴────────────┐
          │                │
   semantic prefix    ScenePlan AR
          │                │
          │        deterministic compiler
          │             [32, T] tracks
          └───────┬────────┘
                  │
      depth-24 Transfusion Renderer
                  │
             FOA latent flow
```

This is not a copy of AudioChat's stereo representation or its 3.6B model.
AudioChat provides the conversational state/data principle. Our contribution
is a compact, editable state for metric azimuth/elevation/distance, arbitrary
source activity and trajectories, persistent source identity, room state, and
FOA rendering.

The resolved final model has 505,101,472 trainable parameters, plus frozen
Qwen3.5-0.8B. The older “494M” number described the two-modality T1 depth-24
model; the final plan table and new context/control projections add about 11M.

## 2. What is retained from AudioChat

[AudioChat](https://arxiv.org/abs/2602.17097) motivates five invariants:

1. Every turn stores a complete source state, not only a prose delta.
2. Source IDs persist across turns; the diff names added, removed, and changed
   sources.
3. Previous audio is an explicit model context for editing.
4. Previous and target audio spans use independent diffusion/flow times during
   training, preventing a same-noise copying shortcut.
5. Generation, understanding, and editing are views of one multimodal model.

We do not copy AudioChat's stereo panning fields, large LM vocabulary, audio
tokenizer, or model scale. ScenePlan and `[32,T]` controls replace those pieces.

## 3. Two controls, deliberately separated

The semantic caption and ScenePlan are not redundant captions:

- `semantic_caption` says *what is audible*: event class, timbre-level wording,
  and speech transcript. It excludes coordinates, activity times, gain, and
  trajectory. Frozen Qwen converts it into a continuous semantic prefix.
- `ScenePlan` says *which persistent source, where, when, how it moves, and how
  it is mixed*. It is the authoritative discrete spatial state.
- `planner_prompt` is the initial full request or the current edit instruction.
  It is used to predict the next ScenePlan, not as a second renderer shortcut.

The renderer receives the semantic Qwen prefix and ScenePlan/tracks. Removing
Qwen would make exact speech/content unnecessarily depend on a tiny plan BPE;
removing ScenePlan would make editing and metric control implicit again.

## 4. Canonical turn record

Each four-state family contains one creation turn followed by three cumulative
edits. A compact training record is:

```json
{
  "family_id": "spcot_train_0000001_...",
  "turn_index": 2,
  "task": "spatial_audio_edit",
  "planner_prompt": "Move source_1 to the left and start it later.",
  "semantic_caption": "A spatial audio scene containing speech saying ... and footsteps.",
  "before": {
    "audio_path": ".../turn_001_WYZX_4ch.flac",
    "scene_plan": {"scene": {"sources": []}}
  },
  "after": {
    "audio_path": ".../turn_002_WYZX_4ch.flac",
    "scene_plan": {"scene": {"sources": []}},
    "source_track_refs": []
  },
  "diff": {"added": [], "removed": [], "changed": []},
  "edit": {"type": "move_source", "target_source_id": "source_1"}
}
```

One source state contains:

```json
{
  "source_id": "source_1",
  "event": {"label": "speech", "category": "speech"},
  "content": {"transcript": "...", "source_audio_id": "..."},
  "activity": {"onset_sec": 1.25, "offset_sec": 6.40},
  "motion": {
    "type": "linear",
    "time_basis": "full_clip",
    "keyframes": [
      {"t_norm": 0.0, "position": {"azimuth_deg": -70, "elevation_deg": 0, "distance_m": 1.8}},
      {"t_norm": 1.0, "position": {"azimuth_deg": 80, "elevation_deg": 25, "distance_m": 2.7}}
    ]
  },
  "acoustics": {"gain_db": -1.5}
}
```

TTS uses `event.label=speech`; transcript is stored once under `content`.
Generative IDs are the small atomic slots `source_0...source_3`; random lineage
hashes are never LM targets.

Transcript supervision is fail-closed. If a transcript-bearing utterance fits
inside the 10.031-second model window, the full utterance is taken from native
sample zero and its complete transcript is retained. If the source is longer
and no word-level alignment exists, recipe v1.4 still takes the beginning but
sets `transcript=null` and
`transcript_quality=omitted_unaligned_long_source`; both planner prompt and
semantic prefix then honestly request generic speech. We do not pair a random
middle crop with the full sentence.

## 5. Compact plan and deterministic controls

`spatial_plan_codec_v2` has a 4096-token vocabulary:

- atomic grammar/field/source tokens;
- enum tokens for room, event class, direction, and quality;
- quantized time, azimuth, elevation, distance, and gain;
- a small BPE only for event labels and transcript text;
- weight-tied input embedding / output head;
- FSM-constrained decoding for a valid one-to-four-source ScenePlan.

The tied table is explicitly initialized at `std=0.02`. Upstream
`nn.Embedding` otherwise starts near `N(0,1)` and produced about 300 nats of
initial CE; after the fix the real-Qwen wiring smoke starts at 8.55–8.63,
close to `ln(4096)=8.32`.

The compiler maps each plan to `[32,T]`, eight values for each of four slots:

```text
[activity, direction_x, direction_y, direction_z,
 inverse_distance, gain_linear, geometry_confidence, activity_confidence]
```

At `T=432`, this remains one control token per FOA-VAE frame. Editing a plan
immediately recompiles the affected trajectory without asking the flow model
to infer coordinates from prose.

## 6. One model, three views

All objectives share one depth-24 Transfusion.

### Persistent-state Planner

```text
Qwen(edit instruction) + previous FOA + previous ScenePlan
    -> current ScenePlan token CE
```

The creation turn uses an explicit zero previous-FOA context and no previous
plan. Field-group weighting keeps grammar supervised at low weight while
metric position, motion, semantics, and speech receive stronger weights.

### Understanding

```text
current FOA + fixed understanding request -> current ScenePlan token CE
```

The target latent is reused as clean context; no duplicate waveform/latent is
stored. This gives the unified model an audio-to-structured-state view.

### Renderer

```text
semantic Qwen prefix + previous FOA + current ScenePlan + clean [32,T] tracks
    -> current FOA latent rectified flow
```

Previous FOA and target FOA receive independently sampled uniform RF times.
Tracks remain clean. Initial-turn zero context stays clean. Semantic CFG drops
only the Qwen prefix; it never deletes the authoritative ScenePlan or tracks.

Resolved objective weights are 0.25 Planner, 0.15 understanding, and 1.0
Renderer. These are one final method, not separate T1/T2/T3 model trees.

## 7. Final dataset scale

Counts refer to persistent edit families, not isolated clips:

| split | families | states/turns | rendered FOA retention |
|---|---:|---:|---|
| train | 1,000,000 | 4,000,000 | transient; delete after QC + VAE success |
| validation | 10,000 | 40,000 | retain |
| test | 2,000 | 8,000 | retain |

Dry-source identities are hash-split 98.5% / 1% / 0.5%. Every base and donor
source in a family comes from the same split; a dry source cannot cross splits.

Every state contains at most one speech source. Multi-source speech examples
are therefore `speech + sound` or `speech + music`, never `speech + speech`;
a family whose base has no speech may introduce at most one speech donor with
`add_source` or `replace_source`. All later states still satisfy the one-speech
limit. Base and donor semantic keys are also unique within one family, so a
replace edit cannot silently change the dry WAV while leaving both ScenePlan
and semantic conditioning unchanged. A same-event spatial change such as one
music source moving from front-right to rear-left is instead a valid
`move_source`: it preserves the dry asset and semantics and changes the metric
trajectory.

Speech classification uses transcript and conservative event-label matching.
We intentionally do not run an acoustic speech detector over sound/music at
production scale; rare completely unlabelled background speech is accepted as
metadata noise and recorded as an explicit dataset risk policy.

No-sound labels are a separate hard exclusion. AudioSet ontology strings are
parsed as comma-separated classes, so both `Silence` and composites such as
`Silence, Engine` are removed before source splitting or family selection.
Free-form captions such as “silent for a moment, followed by a crash” remain
valid because they still describe an audible event. Catalog schema v5 removed
2,695 explicit no-sound templates and independently rechecked all 841,065
retained templates.

Audibility is also family-window specific. If a selected dry playback window
falls below `1e-4`, materialization does not lower the threshold or amplify the
failed candidate into the dataset. It deterministically selects a replacement
from the same split and source kind, excludes conflicting assets/semantics,
and records the requested ref, rejected path/RMS, replacement ref, threshold,
and reason in `source_lineage`. Renderer and preencoder gates remain
independent. Validation10k exercised this path exactly once; its full audit
verified the substitution and every resulting state.

The new exact paired data must be regenerated from dry sources. This means we
return to `/mnt/sdd/audio_dataset/source_wav_cache` for AudioSet/AudioCaps and
the direct FSD50K, VGGSound, PicoAudio, and MusicCaps assets already referenced
by Spatial-FOA-v2. The paired catalog uses the 200k spatial-TTS plan rows; each
one resolves to its exact HiFiTTS/LibriTTS dry utterance through the immutable
400k Parquet locator index (which covers 200k HiFiTTS and 193,260 QC-clean
LibriTTS rows, plus the indexed remainder). It does **not** mean one million
unique WAV files: deterministic seeds reuse a source
in different rooms, activities, trajectories, mixtures, and edit families.

The 151 MB TTS locator JSONL is compiled once into an immutable SQLite index.
Before production, all 400,000 indexed utterances are materialized into
`/mnt/sdb/audio_dataset/spatial_cot_v1/source_cache/speech`. The cache contains
exactly 200,000 HiFi-TTS FLAC and 200,000 LibriTTS WAV assets plus one verified
sidecar per asset (about 78 GB). The bulk reader groups requests by Parquet
file/row group, so every group is read once instead of once per requested row.
The full prewarm took 342 seconds with eight workers; a subsequent no-op takes
about 18 seconds and rechecks the one-to-one 400k audio/sidecar inventory.

Existing 1.018957M FOA/latents remain useful reference/generation supervision;
they are not deleted. Spatial LibriSpeech and old synthetic FOA remain useful
for generation/understanding but are not falsely labelled as exact edit pairs.

## 8. Exact-pair and recoverability contract

Every persistent recipe records:

- dry direct path or stable Parquet locator, file size/mtime identity, and any
  upstream content hash available for that source;
- native rate/frames, exact crop start/count, and pad count;
- the maximum-energy playback window, its raw peak/RMS, and the deterministic
  source-loudness gain used by the renderer;
- family and turn seeds, renderer/backend version;
- room dimensions, RT60/free-field, microphone XYZ, and max image order;
- persistent source ID, exact activity, gain, and trajectory keyframes;
- source-render signature and one family-level master gain;
- complete before/after ScenePlans and explicit diff.

Invariants checked before encoding:

- every target state is independently summed from dry-source-derived tracks;
  the previous FOA is conditioning/reference only and is never an audio input
  to the renderer;
- unchanged source -> identical cached pre-mix track ID;
- gain edit -> identical track, new mix coefficient;
- position/activity edit -> only that track is rerendered;
- add/replace -> resolved donor dry source and new track;
- remove -> one track omitted;
- all four turns share one master gain;
- source playback must have raw RMS at least `1e-4`; a phase-preserving scalar
  gain targets mono RMS 0.05, is dry-peak-limited to 0.95, and is capped at
  +30 dB before the room renderer. Scene `gain_db` remains a separate,
  editable relative-mix control;
- exact 44.1kHz, 442368 samples, four channels, finite samples, peak/RMS QC;
- every output turn must have RMS at least `1e-4` and at least one active
  100 ms frame (`active_100ms_fraction >= 0.005`) before it may be written or
  VAE-encoded;
- recomputed ScenePlan diff equals the persisted diff.

Train FOA can therefore be deleted after successful preencoding. Catalog,
recipes, source lineage, source assets, renderer version, VAE config/checkpoint,
and RNG seed remain sufficient to regenerate it. Bit-identical output still
depends on preserving compatible renderer/library versions, so checksums and
versions are retained rather than claiming timeless numerical identity.

## 9. Storage and sharding

Each family latent is one float16 tensor `[4,64,432]` in a safetensors shard:

```text
1,000,000 * 4 * 64 * 432 * 2 bytes
  = 221,184,000,000 bytes
  = about 206 GiB
```

This avoids four million `.npy + .json` pairs. SQLite stores byte offsets into
metadata JSONL and safetensors keys. The smoke measured approximately 38.5 KB
training metadata and 68.7 KB persistent recipe JSON per family; production
compression and final text distribution may change those estimates.

The retained-eval smoke used about 18 MB/family including four turn mixtures
and the deduplicated source tracks. The exact 12k validation/test size depends
on FLAC compressibility and source count, but remains comfortably inside the
current 2.6 TiB free on `/mnt/sdb`. Train render staging is bounded in
`/dev/shm`: eight active 256-family render shards plus a 16-shard total
render/encode double buffer. The validated setting uses 12 render processes per
active shard (96 total). A full 2,048-family renderer-only benchmark measured
14.82 families/s at 96 processes versus 15.19 at 128; the 2.5% isolated gain
at 128 consumed nearly all CPU headroom and is not worthwhile while source
materialization and eight VAE workers run concurrently.

All durable Spatial-CoT artifacts live under
`/mnt/sdb/audio_dataset/spatial_cot_v1`: catalogs/recipes, the verified speech
cache, retained validation/test FOA and source tracks, latents, and any
overflow. Train FOA uses only transient `/dev/shm/spatial_cot_v1/staging` and
is removed after durable DONE. `/mnt/sdd` is only an input location for
existing dry-source assets; `/mnt/sdc` remains for checkpoints, inference, and
evaluation outputs.

## 10. Execution path

No command below is launched by reading this document.

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools

# Rebuild ScenePlan with TTS source locators, annotate activity, build codec,
# then create the source-disjoint 1M/10k/2k metadata-only master catalog.
scripts/t2a/data/prepare_spatial_cot_v1_artifacts.sh

# Recommended guarded path: metadata audit -> nested smoke -> recovery audit ->
# nested pilot -> pilot audit. This command has no full-data mode.
scripts/t2a/data/run_spatial_cot_prepare_pilot.sh

# The individual nested commands remain available when debugging one stage.
scripts/t2a/data/run_spatial_cot_data.sh smoke64
scripts/t2a/data/run_spatial_cot_data.sh pilot2048

# Full data, when smoke/pilot gates pass. Build retained evaluation splits
# first so their outputs can be inspected before committing to the 1M run.
scripts/t2a/data/run_spatial_cot_data.sh test2k
scripts/t2a/data/run_spatial_cot_data.sh audit-test2k
scripts/t2a/data/run_spatial_cot_data.sh validation10k
scripts/t2a/data/run_spatial_cot_data.sh audit-validation10k
scripts/t2a/data/run_spatial_cot_data.sh train1m

# The same production phases can be launched as named, resumable tmux jobs.
scripts/t2a/data/launch_spatial_cot_data_tmux.sh test2k

# Benchmark BATCH_SIZE=1 first. It means four turns/GPU and three shared-model
# objectives, not one isolated clip.
BENCHMARK=1 MAX_STEPS=4 BATCH_SIZE=1 \
  scripts/t2a/train/run_t2a_spatial_chat_500m_8gpu.sh

# Formal first epoch is only launched after the 5k pilot promotion gates pass.
scripts/t2a/train/run_spatial_cot_training_gate.sh formal1epoch
```

The data supervisor uses parallel CPU Pyroom renderers and eight long-lived GPU
preencoders. Each GPU loads the 595 MiB VAE once. A bounded double buffer keeps
transient train staging below `/dev/shm` capacity; READY/DONE markers make every
work shard restartable. Only a reopened, finite, checksum-recorded safetensors
shard may trigger train-render cleanup. A reboot loses at most the in-flight
tmpfs render shards; persistent recipes and completed latent DONE shards resume.

The validated full-machine setting is eight active shards with twelve Pyroom
processes each (96 render processes total), a 16-shard total double buffer, and
eight persistent VAE workers. With the final batched-FFT renderer, PCM24
profiles, RAM staging, source loudness alignment, and per-state audibility
gates, the production `pilot2048` completed materialize + render + preencode +
cleanup + finalization in 341 seconds, or 6.01 families/s; its full audit and
two deterministic recovery renders finished in 398 seconds total. Linear
projection is about 46.8 hours for all 1,012,000 production families; the
operational budget is roughly 50--60 hours including retained-eval writes,
full audits, and long-tail shards. The speech cache is already warm, removing
the former cold-Parquet tail. Increasing to 128 simultaneous render processes
is not the default for the CPU-headroom reason measured above.

The retained quality gate reopens every latent tensor, verifies every retained
FOA/source-track PCM24 header and SHA, checks persisted audibility statistics
for every state, samples decoded signal RMS/activity/clipping, round-trips
ScenePlan tokens through the runtime loader, and independently remixes
stratified families from their retained source tracks.
It publishes `eval_rendered/{test,validation}/QUALITY.json` only on complete
success. The `train1m` wrapper refuses to launch until both markers pass.

## 11. Completed validation evidence

Current production metadata and nested validation roots are:

```text
/mnt/sdb/audio_dataset/spatial_cot_v1/scene_plan
/mnt/sdb/audio_dataset/spatial_cot_v1/codec
/mnt/sdb/audio_dataset/spatial_cot_v1/catalog
/mnt/sdb/audio_dataset/spatial_cot_v1/catalog/generated_views/smoke64
/mnt/sdb/audio_dataset/spatial_cot_v1/catalog/generated_views/pilot2048
/mnt/sdb/audio_dataset/spatial_cot_v1/audits/smoke64.json
/mnt/sdb/audio_dataset/spatial_cot_v1/audits/pilot2048.json
```

Recovery re-render evidence is embedded in those two audit reports. Historical
migration stores and version-suffixed audit files are not retained.

Verified in this implementation pass:

- the frozen step-2560 source-disjoint text baseline covers 16 validation
  families × 4 turns: Planner/Understanding CE is `1.1148/1.1528`, FSM
  ambiguous-choice accuracy is `77.28%/76.36%`, and exact plans are `0/64` for
  both views; rotating the complete Planner condition raises CE by `0.0611`,
  while rotating the Understanding FOA raises it by only `0.000627`, making
  audio-condition sensitivity an explicit 5k promotion target rather than
  treating the bounded-overfit CE near `0.0015` as generalization;
- full metadata audit PASS for 1,018,957 ScenePlans, 841,065 audible source
  templates, and 1,012,000 source-disjoint family specifications; 2,695
  explicit no-sound templates were rejected before source splitting;
- compound speech labels are conservative speech-bearing sources; 115,790
  metadata families contain a conditional speech donor, while every rendered
  state still has at most one speech source;
- all base/donor semantic keys in one family are distinct, all replace edits
  have a planner-visible diff, and donors are consumed exactly once;
- `smoke64` PASS: 64 families / 256 states, all tensors/metadata/runtime
  turns plus deterministic delete-and-regenerate checksum recovery; every
  state has persisted audibility statistics, with minimum RMS `0.00514` and
  minimum active-100ms fraction `0.1089`;
- `pilot2048` PASS: 2,048 families / 8,192 states, all 2,048 latent tensors
  and metadata records verified, plus 128 stratified runtime families and two
  deterministic recovery renders; all 8,192 states carry signal statistics,
  with minimum RMS `0.00514`, minimum active-100ms fraction `0.04950`, and
  maximum peak `0.90000004`;
- isolated catalog-v5 render smoke PASS: all 64 families were independently
  remixed from source tracks with maximum absolute error `5.96e-8`, zero
  near-clipping, and minimum state RMS `0.00514`; the complete scratch tree was
  then deleted;
- retained `test2k` QUALITY v2 PASS: 2,000 families / 8,000 states, all 17,224
  retained audio files hashed, all state signal statistics verified, minimum
  RMS `0.005143`, minimum active-100ms fraction `0.05941`, zero near-clipping,
  and 128 independent remixes with maximum absolute error `5.96e-8`;
- retained `validation10k` QUALITY v2 PASS: 10,000 families / 40,000 states,
  all 86,726 retained audio files hashed, all state signal statistics verified,
  minimum RMS `0.004648`, minimum active-100ms fraction `0.05941`, zero
  near-clipping, and 256 independent remixes with maximum absolute error
  `5.96e-8`; its single deterministic quiet-window source substitution was
  present in lineage and passed the full structural audit;
- transcript policy verified: complete in-window speech retains exact full
  transcript from native sample zero; 107 pilot long-source occurrences omit
  unaligned transcript instead of pairing a random crop with the full text;
- full source audit: all 470,966 unique dry paths referenced by the 460,255
  Spatial-FOA-v2 manifest rows exist at their recorded direct locations;
- all 393,260 QC-clean TTS renders have an exact source-plan row, matching
  `(source_dataset, source_id)`, and a matching entry in the 400k Parquet
  source index (zero missing or mismatched joins);
- the complete 400k TTS JSONL compiled to SQLite in 7.5 seconds; the full
  400k source cache was materialized in 342 seconds and now has exactly 400k
  audio files + 400k sidecars, with a 256-asset stratified payload-SHA/header
  audit PASS;
- catalog creation and source-disjoint selection;
- direct AudioSet and exact TTS Parquet source resolution;
- four persistent states with move/remove/add/replace-capable recipes;
- Pyroom FOA rendering and unchanged-track/diff/master-gain QC;
- wdmix-1.35M VAE output `[4,64,432]` float16 family tensors;
- safetensors + metadata JSONL + SQLite finalization and reader round-trip;
- previous-turn latent recovery and deterministic `[32,432]` controls;
- real frozen-Qwen + depth-2 wiring version of the same final model completed
  all three objective forwards and backward (Renderer loss 2.33);
- after tied-embedding initialization, Planner/understanding CE starts at
  8.63/8.55;
- bounded supervisor smoke completed and deleted only its generated render
  staging after DONE; re-running a complete `smoke64` returned immutable
  `ALREADY_READY` in 0.05 seconds without allocating a GPU.

## 12. Remaining execution gates

1. Let the active resumable `train1m` data job finish; train FOA remains
   transient while recipes/latents remain durable.
2. Run the final train reader/index audit and a fresh deterministic recovery
   render after all 1M latent families are READY.
3. Benchmark the full depth-24 model at family batch 1 (then 2 only if safe)
   through backward and Adam-state allocation on eight GPUs, then start the
   300k/50k-checkpoint training run.

The large launch gate is operational, not methodological: exact source joins,
zero failed QC families, sufficient free space, and a full-depth batch smoke.
