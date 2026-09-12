# `dataset/` — spatial-audio data pipeline (by category)

Scripts are grouped by their role in the pipeline. Run everything from the repo
root with `uv run python dataset/<category>/<script>.py`.

```
indexing/    discover + label raw source audio into unified pools
captioning/  turn sources / synthesis metadata into text captions
synthesis/   build first-order-ambisonics (FOA) spatial audio with pyroom
evaluation/  measure VAE reconstruction quality (incl. spatial metrics)
features/    extract video features for later (video-conditioned) stages
```

## indexing/
| file | what it does |
|---|---|
| `build_source_index.py` | Scan AudioCaps / AudioSet / MusicCaps / SLS / MRSDrama / PicoAudio → balanced `sources_{audio,music,speech}.jsonl` mono-source pools (parquet/zip extracted to a wav cache). |
| `build_spatial_prompts.py` | SLS (parquet) + MRSDrama (data.json) native metadata → prompt jsonl. |

## captioning/
| file | what it does |
|---|---|
| `refine_caption.py` | Synthesis manifest (content + azimuth/elevation/distance/motion + room) → ONE natural caption per clip (content **and** space). Template fallback (`--no-llm`) or text-LLM refine. |
| `caption_audio.py` | Audio-only captioner (Qwen3-Omni-Captioner) for sources without text (e.g. Sphere360 audio). |
| `caption_sphere360.py` | Video / AV captioner for Sphere360. |

## synthesis/
| file | what it does |
|---|---|
| `synthesize_foa_pyroom.py` | FOA engine: az/el/distance placement, **room archetypes** (booth→cathedral→outdoor), static & dynamic (moving) sources, multi-source mixing, ACN/SN3D `[W,Y,Z,X]`. Also the AudioCaps mono→FOA path. |
| `build_spatial_dataset.py` | Orchestrator: plan ~200k clips with category balance (audio/music/speech), mix ratios (single/pair/multi), static/dynamic ratio; calls the engine; writes FOA + rich spatial manifest. |

## evaluation/
| file | what it does |
|---|---|
| `eval_vae_recon.py` | Compare VAE reconstructions vs sources: SI-SDR, LSD, + FOA spatial (intensity-DoA error, directional-energy, inter-channel correlation). Writes a `<tag>/` report folder. |

## features/
| file | what it does |
|---|---|
| `extract_videomae_features.py` | VideoMAE-v2 sliding-window features for video-conditioned stages. |

## End-to-end (spatial corpus → VAE)
```bash
# 1) source pools
uv run python dataset/indexing/build_source_index.py --out ${AMBIT_CACHE_ROOT}/spatial_sources ...
# 2) synthesize FOA (varied rooms, static/dynamic, mixes)
uv run python dataset/synthesis/build_spatial_dataset.py --sources-dir ... --out-dir ... --manifest ...
# 3) spatial captions
uv run python dataset/captioning/refine_caption.py --manifest ... --out captions.jsonl --num_gpus 8
# 4) (after VAE training + decode) evaluate reconstruction
uv run python dataset/evaluation/eval_vae_recon.py --recon-dir ... --tag <vae_setting>
```
