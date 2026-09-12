# AMBIT

**Executable Scene Plans for Native Ambisonic Generation and Editing**

Official code for **AMBIT**: a Transfusion-style system that *plans* a first-order Ambisonic (FOA) scene and then *renders* it.

Spatial audio is not only “what should I hear?”, it is also *when* each source is active and *where* it moves. AMBIT makes those decisions explicit. A language request or an edit instruction is turned into a **ScenePlan**. A deterministic compiler converts that plan into renderer conditions. A shared Transformer then synthesizes native four-channel FOA with rectified flow in a frozen FOA VAE latent space.

| Generation | Editing |
| --- | --- |
| English request → ScenePlan AR → compile → FOA DiT → VAE decode | Reference FOA + instruction → ScenePlan AR (with CLAP-FOA evidence) → compile → FOA DiT (conditioned on the reference latent) → VAE decode |

Audio I/O is always **44.1 kHz, 4-channel FOA**, channel order **WYZX / ACN / SN3D**. There is no stereo proxy and no implicit source assignment.

---

## What AMBIT does

Three pieces of work sit on top of one another.

### 1. ScenePlan: an executable intermediate

A ScenePlan is a complete, checkable description of the target scene, not a free-form caption.

| Field | Constraint |
| --- | --- |
| Duration | at most 15.05 s (648 VAE latent frames at hop 1024) |
| Room | `dry`, `moderate`, `reverberant`, or `outdoor` |
| Sources | 1–4 sources; at most one speech source |
| Per source | kind (`speech` / `music` / `sound`), free-text description, optional transcript, activity interval, static or linear 3D trajectory |

Source identity is persistent across an edit: *move the cello* changes that source’s pose and leaves every other field and every other source alone. Because the plan is explicit, we can score **plan fidelity** before audio is rendered, and score **rendering fidelity** under a ground-truth plan.

The compiler produces two conditions:

- **Semantic clauses:** source descriptions and transcripts, encoded by a frozen Qwen3.5-0.8B text encoder and read with cross-attention.
- **Control track (256 x T):** four source slots times (event id + pose features), aligned to the latent frame grid. Text and gain never enter this path.

### 2. FOA-aware VAE

A four-channel Oobleck-style VAE maps WYZX waveforms to a 64 x T latent. The latent is split into a 40-channel content group and a 24-channel directional group:

- the same decoder must reconstruct W from the content group alone;
- grouped KL is asymmetric (weaker on content, stronger on direction);
- a spatial-covariance loss matches inter-channel energy, coherence, and phase.

The VAE is **frozen** for all later AR / DiT / CLAP training.

### 3. One Transformer for planning and rendering

Following Transfusion, the same 15-block DiT (width 1024, about 0.32B parameters) carries:

- a next-token objective over grammar-constrained ScenePlan tokens (**AR planner**);
- a rectified-flow objective over FOA latents (**renderer**).

**Generation.** The planner reads a qualitative English request and emits an admissible ScenePlan. Light heads recover inventory (how many sources, of which kinds), copy literal descriptions/transcripts, and map qualitative phrases (left / approaching / later) onto bins. Fields the request leaves free are completed inside the admissible region so one request maps to one reproducible plan. The renderer then starts from Gaussian noise and integrates the flow conditioned only on the compiled plan.

**Editing.** The planner additionally sees the reference latent and CLAP-FOA features that bind the instruction to one source. The renderer concatenates the noisy target, the compiled new plan, and the **clean reference latent** (64 extra channels, InstructPix2Pix-style) so unmodified sources can be copied. Five atomic operations are supported:

`event_addition` · `event_removal` · `linear_to_static` · `static_to_linear` · `stationary_spatial_relocation`

**OPSD.** On-policy self-distillation (`stable_audio_tools/training/transfusion_opsd/`) couples execution feedback on discrete plan decisions with audio-validated velocity targets for the renderer.

```text
  English request or edit + reference FOA
                    |
                    v
        ScenePlan AR (shared Transformer)
          room, duration, per-source fields
                    |
                    v
        Deterministic compiler
          Qwen semantic clauses
          + 256-channel control track
                    |
                    v
        FOA DiT (rectified flow)
          editing also conditions on z_s
                    |
                    v
        Frozen FOA VAE decode --> 44.1 kHz FOA WAV
```

---

## Installation

Python **3.10** is required.

```bash
git clone https://github.com/WJM-George/AMBIT.git
cd AMBIT
uv sync --extra train --extra spatial
# or: pip install -e ".[train,spatial]"
```

The importable package name is still `stable_audio_tools`, so existing checkpoints keep working.

Copy the environment template and point it at **your** disks:

```bash
cp .env.example .env
# then edit .env
set -a && source .env && set +a
```

| Variable | Default | What it should contain |
| --- | --- | --- |
| `AMBIT_DATA_ROOT` | `data` | ScenePlan indexes, latents, synthesized FOA |
| `AMBIT_CKPT_ROOT` | `checkpoints` | VAE, DiT, AR, CLAP, Qwen |
| `AMBIT_CACHE_ROOT` | `cache` | Hugging Face and download caches |

JSON configs expand `${AMBIT_DATA_ROOT}`, `${AMBIT_CKPT_ROOT}`, and `${AMBIT_CACHE_ROOT}`. Launchers read GPUs from `CUDA_VISIBLE_DEVICES`; they do not assume a laboratory device map.

---

## Data: where it comes from

AMBIT does **not** train on raw internet FOA as the supervision target. Every training scene is **simulated from a ScenePlan**, so the plan and the waveform are exactly paired. Public corpora are used as **mono assets** (and, for the VAE, as additional FOA clips).

### Public sources (`data_download/`)

Download into `$AUDIO_DATASET_ROOT` (defaults to `$AMBIT_DATA_ROOT`):

```bash
source data_download/scripts/env_audio_dataset.sh
huggingface-cli login          # gated sets only
uv run python data_download/scripts/download_dataset.py audiocaps
uv run python data_download/scripts/download_all.py
```

| Role in AMBIT | Corpora (catalog keys) |
| --- | --- |
| Speech assets + transcripts | LibriTTS, Hi-Fi TTS; Spatial LibriSpeech FOA (`spatial_librispeech`) |
| Sound / music assets + captions | AudioCaps, AudioSet, VGGSound, MusicCaps, PicoAudio, FSDKaggle2019, Audio-FLAN |
| Extra spatial / AV material | Sphere360, BEWO-1M, MRSAudio, MRSDrama, YT-Ambient / ViSAGe |

Gated Hub datasets need license acceptance. Sphere360 may also need `$SPHERE360_COOKIE` for yt-dlp. Details: [`docs/DATA.md`](docs/DATA.md) and [`data_download/README.md`](data_download/README.md).

### Scene construction (`dataset/` + `scripts/t2a/data/`)

1. **Index** mono assets — `dataset/indexing/`
2. **Caption** sources that have no text — `dataset/captioning/`
3. **Sample a ScenePlan** top-down (room → duration → source count → one source at a time)
4. **Render FOA** with pyroomacoustics image-source, ACN/SN3D, -23 dBFS W-channel RMS, -1 dBFS true-peak — `dataset/synthesis/`
5. **Write generation requests** as qualitative paraphrases of the plan (exact angles/times left open)
6. **Write edit pairs** by applying exactly one of the five operations to one source and re-rendering with the same assets and room — `scripts/t2a/data/`

Paper partitions (not shipped in git):

| Split | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Generation scenes | 1.6M | 32k | 8k |
| Editing pairs | 1.0M | 20k | 5k |

The VAE mix is 600k synthetic non-speech FOA + 219k Spatial LibriSpeech + 200k synthetic speech FOA, trained on 4 s crops.

Keep the official test sets frozen. Do not select checkpoints on test.

---

## Checkpoints

Put weights under `$AMBIT_CKPT_ROOT`:

```text
$AMBIT_CKPT_ROOT/
  pretrained/Qwen/Qwen3.5-0.8B/          # frozen text encoder
  compareVAE_ckpt/<vae>.ckpt             # frozen FOA VAE
  dit/<generation_renderer>/             # ScenePlan-conditioned DiT
  generation_ar/                         # Generation AR
  editing_clap44/                        # Editing CLAP-FOA
  editing_dit/  editing_ar/              # Editing renderer and planner
```

Released AMBIT weights will be linked here when they are public. Until then, pass explicit `--checkpoint` / `--release` paths. A typical Qwen snapshot is `Qwen/Qwen3.5-0.8B` from Hugging Face.

---

## Inference

Set `CUDA_VISIBLE_DEVICES` to the GPUs you want. Outputs are 44.1 kHz FOA WAVs plus the generated ScenePlan JSON.

### Text-to-FOA generation

English request → Generation AR → compiled ScenePlan → frozen renderer → FOA.

```bash
python scripts/t2a/inference/generate_foa_from_raw_english.py \
  --request "A dog barks on the left while rain falls behind me." \
  --checkpoint "$AMBIT_CKPT_ROOT/generation_ar.ckpt" \
  --snapshot "$AMBIT_CKPT_ROOT/generation_ar_snapshot" \
  --output outputs/generation_demo
```

Batch mode: `--requests requests.json` with

```json
{ "requests": [ { "id": "demo_001", "request": "A cello on the left, beatboxing moving from back to front." } ] }
```

Plan only (no audio):

```bash
python scripts/t2a/inference/generate_sceneplan_from_raw_english.py \
  --request "Two sources in a dry room: speech on the right, music static on the left." \
  --checkpoint "$AMBIT_CKPT_ROOT/generation_ar.ckpt" \
  --snapshot "$AMBIT_CKPT_ROOT/generation_ar_snapshot" \
  --output outputs/plan_demo
```

### Instruction-guided editing

Reference FOA + instruction → Editing AR (CLAP-FOA) → new ScenePlan + edited FOA.

```bash
python scripts/t2a/inference/edit_foa_with_clap44.py \
  --release "$AMBIT_CKPT_ROOT/editing_clap44_release.pt" \
  --release-sha256 <sha256 of that file> \
  --source path/to/source.wav \
  --instruction "Move the speaker behind me and keep the music unchanged." \
  --output-dir outputs/edit_demo \
  --device cuda
```

`--source` must be finite native 44.1 kHz four-channel FOA within the 648-frame limit. The run writes `edited.wav`, `NEW_SCENEPLAN.json`, and `RESULT.json`.

A versioned generation bundle (AR plus pinned renderer) can be launched with `scripts/t2a/inference/run_generation_ar_bundle.py`.

---

## Training

All commands are from the repository root with `AMBIT_DATA_ROOT` / `AMBIT_CKPT_ROOT` set. Recommended order matches the paper: **VAE → generation renderer (DiT) → generation AR → editing CLAP → editing DiT → editing AR**.

### 1. FOA VAE

```bash
python train_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
  --name vae_4ch \
  --save-dir "$AMBIT_CKPT_ROOT/vae"
```

Ablations: `stable_audio_tools/configs/model_configs/autoencoders/ablation_arms/`.

### 2. Generation renderer (ScenePlan → FOA DiT)

`train.py` is the Lightning entry. Model / data JSON live under `stable_audio_tools/configs/`. Example:

```bash
python train.py \
  --model-config stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_model_sceneplan_44.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/sceneplan_v2_train.json \
  --val-dataset-config stable_audio_tools/configs/dataset_configs/sceneplan_v2_validation.json \
  --name generation_dit \
  --save-dir "$AMBIT_CKPT_ROOT/dit"
```

Multi-GPU launchers: `scripts/t2a/train/run_sceneplan_dit_*.sh`.

### 3. Generation AR (request → ScenePlan)

```bash
python scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py \
  --mode full \
  --run-dir "$AMBIT_CKPT_ROOT/generation_ar" \
  --train-manifest "$AMBIT_DATA_ROOT/sceneplan_v2_1p124m/transfusion_shared_v1/generation_ar/train.sqlite" \
  --validation-manifest "$AMBIT_DATA_ROOT/sceneplan_v2_1p124m/transfusion_shared_v1/generation_ar/validation.sqlite" \
  --batch-size 2
```

`--mode tiny` is an 8-row smoke run.

### 4. Editing stack

| Stage | Command |
| --- | --- |
| CLAP-FOA pretraining | `python scripts/t2a/train/train_sceneplan_transfusion_editing_clap44.py --index ... --preflight ... --config stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_clap44_v1.json --output "$AMBIT_CKPT_ROOT/editing_clap44"` |
| Editing DiT | `scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh` (set `CUDA_VISIBLE_DEVICES`) |
| Editing AR (joint) | `python scripts/t2a/train/train_sceneplan_transfusion_editing_ar_joint.py` |

Editing DiT warms the generation renderer and adds 64 reference-latent channels. Editing AR shares those blocks and trains the discrete plan head on the five operations.

OPSD data prep: `scripts/t2a/rl/`. Training code: `stable_audio_tools/training/transfusion_opsd/`.

More flags and configs: [`docs/TRAINING.md`](docs/TRAINING.md).

---

## Evaluation

Official scorers are in `scripts/t2a/eval/`. They take contracts and environment paths; they do not assume a particular machine.

Paper protocol, briefly:

- **Generation (8k):** request satisfaction of predicted plans; renderer accuracy under ground-truth plans; remaining error attributed to planner–renderer shift.
- **Editing (5k):** native FOA spatial metrics on common valid frames (angle, trajectory RMSE, edit precision, keep error) plus a frozen matched content/spatial suite (CLAP, FAD, PANN, LSD, GCC, StereoCRW, FSAD).

```bash
python -m pytest -q tests
```

Tests that need a local codec or index skip if `$AMBIT_DATA_ROOT` is empty.

---

## Repository map

| Path | Role |
| --- | --- |
| `stable_audio_tools/` | VAE, ScenePlan codec, AR/DiT/CLAP, OPSD, loaders |
| `stable_audio_tools/configs/` | Model and dataset JSON (`${AMBIT_*}` placeholders) |
| `scripts/t2a/inference/` | Generation and editing CLIs |
| `scripts/t2a/train/` | AR, DiT, CLAP trainers and launchers |
| `scripts/t2a/eval/` | Paper evaluation and baselines |
| `scripts/t2a/data/` | ScenePlan / edit-pair construction |
| `dataset/` | Indexing, captioning, FOA synthesis |
| `data_download/` | Public corpus downloaders |
| `tests/` | Unit and contract tests |

---

## Citation

```bibtex
@misc{ambit,
  title  = {AMBIT: Executable Scene Plans for Native Ambisonic Generation and Editing},
  author = {Anonymous}
}
```

## Acknowledgements

AMBIT builds on [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). Upstream MIT license and third-party notices are in `LICENSE` and `LICENSES/`.

## License

MIT. See `LICENSE`.
