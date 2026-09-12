# AMBIT

**Executable Scene Plans for Native Ambisonic Generation and Editing**

Official implementation of AMBIT, a Transfusion-style framework for first-order Ambisonic (FOA) generation and instruction-guided editing.

Creating and revising a spatial scene requires deciding what each source produces, when it is active, and where it moves, then realizing those decisions acoustically while leaving everything else untouched. AMBIT represents that target with an executable intermediate, **ScenePlan**: a source-wise specification of content, optional transcript, activity interval, and 3D trajectory. A deterministic compiler turns the plan into renderer conditions. A shared Transformer predicts the plan autoregressively from a qualitative request or an edit instruction, and renders FOA latents with rectified flow.

For editing, the planner also receives contrastively aligned FOA audio–text features of the reference recording, and the renderer keeps the full reference latent sequence so unmodified sources can be preserved.

```text
request or edit instruction
        │
        ▼
   ScenePlan AR  ──compile──►  FOA DiT (rectified flow)
        ▲                            │
        │                            ▼
   optional CLAP-FOA            native FOA waveform
   reference features           (WYZX / ACN / SN3D)
```

## Highlights

- **Native FOA generation and editing** through one planner–renderer stack, rather than stereo proxies or implicit source assignment.
- **ScenePlan** as structured, executable chain-of-thought: content, timing, and trajectory are explicit and compilable.
- **FOA-aware VAE** with shared-decoder reconstruction of the omnidirectional W channel, asymmetric grouped KL, and spatial covariance supervision.
- **On-policy self-distillation (OPSD)** that couples discrete plan-decision feedback with audio-validated velocity targets for the renderer.

## Repository layout

| Path | Role |
| --- | --- |
| `stable_audio_tools/` | Models, data loaders, training, and inference library |
| `stable_audio_tools/configs/` | Model and dataset configs (paths use `${AMBIT_*}` placeholders) |
| `scripts/t2a/inference/` | Generation and editing entry points |
| `scripts/t2a/train/` | AR, DiT, and CLAP training |
| `scripts/t2a/eval/` | Official evaluation and baseline scoring |
| `scripts/t2a/data/` | ScenePlan construction and materialization |
| `dataset/` | Source indexing, captioning, and FOA synthesis |
| `data_download/` | Public-dataset downloaders |
| `tests/` | Unit and contract tests |

The Python package name remains `stable_audio_tools` so existing checkpoints and imports stay compatible.

## Installation

Python 3.10 is required.

```bash
git clone https://github.com/WJM-George/AMBIT.git
cd AMBIT
uv sync --extra train --extra spatial
```

Alternatively:

```bash
pip install -e ".[train,spatial]"
```

Copy the environment template and point it at **your** data and checkpoint directories:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `AMBIT_DATA_ROOT` | `data` | ScenePlan indexes, latents, and synthesized FOA |
| `AMBIT_CKPT_ROOT` | `checkpoints` | VAE, DiT, AR, CLAP, and pretrained backbones |
| `AMBIT_CACHE_ROOT` | `cache` | Hugging Face and download caches |

JSON configs expand `${AMBIT_DATA_ROOT}`, `${AMBIT_CKPT_ROOT}`, and `${AMBIT_CACHE_ROOT}`. Do not hard-code machine mounts in configs or launchers.

## Checkpoints

Place released or locally trained weights under `$AMBIT_CKPT_ROOT`, for example:

```text
$AMBIT_CKPT_ROOT/
  pretrained/Qwen/Qwen3.5-0.8B/
  compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt
  dit/sceneplan_dit_v11_.../checkpoints/*.ckpt
```

Pretrained AMBIT weights will be linked here when they are released. Until then, pass explicit `--checkpoint` / `--release` paths.

## Inference

**Text-to-FOA generation** (English request → ScenePlan AR → frozen renderer):

```bash
python scripts/t2a/inference/generate_foa_from_raw_english.py \
  --request "A dog barks on the left while rain falls behind me." \
  --checkpoint "$AMBIT_CKPT_ROOT/generation_ar.ckpt" \
  --snapshot "$AMBIT_CKPT_ROOT/generation_ar_snapshot" \
  --output outputs/generation_demo
```

**Instruction-guided editing** (reference FOA + instruction → new ScenePlan + edited FOA):

```bash
python scripts/t2a/inference/edit_foa_with_clap44.py \
  --release "$AMBIT_CKPT_ROOT/editing_clap44_release.pt" \
  --release-sha256 <sha256> \
  --source path/to/source.wav \
  --instruction "Move the speaker behind me and keep the music unchanged." \
  --output-dir outputs/edit_demo
```

Inputs and outputs are 44.1 kHz, 4-channel FOA in WYZX / ACN / SN3D. Set `CUDA_VISIBLE_DEVICES` to whatever GPUs you want to use.

## Training

FOA VAE:

```bash
python train_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
  --save-dir "$AMBIT_CKPT_ROOT/vae"
```

Generation AR, editing DiT / AR / CLAP, and the shared renderer use the scripts in `scripts/t2a/train/` with the configs under `stable_audio_tools/configs/`. See [docs/TRAINING.md](docs/TRAINING.md).

## Data

Public source corpora are downloaded with `data_download/`. ScenePlan construction and FOA synthesis live under `dataset/` and `scripts/t2a/data/`. Layout and commands: [docs/DATA.md](docs/DATA.md).

Generation uses 1.6M / 32k / 8k train–val–test scenes. Editing uses a separate 1M / 20k / 5k pair split.

## Evaluation

Official scorers and baseline runners are in `scripts/t2a/eval/`. They read paths from the environment and from the contracts you pass in; they do not assume a particular machine layout.

```bash
python -m pytest -q tests
```

Some tests skip unless the corresponding codec or data artifacts exist under `$AMBIT_DATA_ROOT`.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{ambit2027,
  title     = {AMBIT: Executable Scene Plans for Native Ambisonic Generation and Editing},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

## Acknowledgements

AMBIT builds on [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). The upstream MIT license and third-party notices are retained in `LICENSE` and `LICENSES/`.

## License

MIT. See `LICENSE`.
