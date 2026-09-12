# ScenePlan v2 non-speech source descriptions

This folder labels unique mono **music** and **sound** sources. Speech donors
are excluded because LibriTTS/HiFiTTS already provide authoritative metadata
and transcripts.

The contract is intentionally simple:

- `Qwen/Qwen3-Omni-30B-A3B-Instruct` at revision
  `26291f793822fb6be9555850f06dfe95f2d7e695`;
- one natural English semantic audio description, prompted at about 20--30
  words; this length is a soft target;
- no restrictions on subject wording or terms such as audio, clip, listener,
  microphone, speech, room, or position;
- store the complete direct response after whitespace compaction only;
- hard failures are limited to empty/non-English output or generation cutoff.

Build the deterministic 50-music/50-sound pilot selection:

```bash
.venv/bin/python dataset/captioning/sceneplan_a2t_v2/build_pilot_selection.py
```

Run four independent two-GPU Transformers workers:

```bash
.venv-qwen/bin/python dataset/captioning/sceneplan_a2t_v2/launch_transformers_scaleout.py \
  --log-dir ${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/audit/a2t_pilot_100/logs_transformers \
  -- \
  --input-jsonl ${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/audit/a2t_pilot_100/instruct_input.jsonl \
  --out ${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/audit/a2t_pilot_100/source_descriptions_instruct.jsonl \
  --device-map balanced \
  --attn-implementation sdpa \
  --safety-max-generation-tokens 256
```

Merge, audit, and select 10 music plus 10 sound examples:

```bash
.venv-qwen/bin/python dataset/captioning/sceneplan_a2t_v2/finalize_pilot.py
```

The pilot stops for user review. It does not start revised ScenePlans, full
annotation, P8, or P9.

After the user accepts the 50-music/50-sound descriptions, build the separate
100-record ScenePlan conditioning pilot:

```bash
.venv/bin/python dataset/captioning/sceneplan_a2t_v2/build_revised_sceneplan_pilot.py
```

This second pilot uses 50 speech and 50 no-speech scenes, covers one to four
sources, compiles the full 4+4+2 masks and structured controls, and audits the
complete captions against the proposed 512-token ceiling. It does not render
FOA, start full annotation, rewrite the 1.124M manifests, or enter P8/P9.
Its primary output is JSONL with one complete renderer-sample record per line;
the Parquet file is only a machine-reading mirror of the same 100 records.

## Frozen full-scale preflight

The production universe is frozen at exactly **873,502** unique non-speech
audio hashes: 521,982 music and 351,520 sound.  The direct Instruct response is
stored after whitespace compaction only.  `spoken_language_background` is a
separate safety label; it never rewrites the description.  Singing and rap
remain music.  A ScenePlan containing formal TTS may not use a non-speech donor
whose safety label is true, and the exact transcript belongs only to the formal
TTS source.

The frozen input and schemas live under:

```text
${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/source_annotations/nonspeech_instruct_v2/
```

The eight-GPU preflight used four independent two-GPU workers, batch size 256,
SDPA attention, and last-token-only vocabulary projection.  The 1,000-row test
completed with zero hard failures and zero generation cutoffs at a steady
aggregate 12.216 descriptions/s.  That projects to 19.86 hours for the frozen
universe.  The authoritative report is:

```text
${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/audit/a2t_throughput_1k_20260816/preflight_report.json
```

The revised 100-row ScenePlan pilot is canonical JSONL with one complete sample
per line.  It contains 50 formal-TTS and 50 no-speech scenes, has zero
formal-TTS/spoken-background violations, passes all 4+4+2 and structured-control
checks, and has no caption truncation under the 512-token contract:

```text
${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/pilots/revised_sceneplan_100_registry_v1/revised_sceneplans_100.jsonl
```

The full-scale launch command is recorded in `preflight_report.json` as
`selected_production_runner.launch_command_not_executed`.  It must not be run
until the user explicitly confirms the 873,502-source launch.  Rerunning the
same command is the supported resume path: each modulo shard skips durable
completed IDs.  The finalizer refuses to write the registry or `READY` until
all 873,502 descriptions and matching spoken-language labels are present
exactly once.  Full annotation, P8, and P9 are currently not started.
