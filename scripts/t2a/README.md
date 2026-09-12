# Text-to-Ambisonics

Paper entry points for AMBIT generation and editing.

```text
English request or edit instruction
  -> ScenePlan AR
  -> compiled renderer conditions
  -> FOA DiT / VAE decode
```

- Inference: `scripts/t2a/inference/`
- Training: `scripts/t2a/train/`
- Evaluation: `scripts/t2a/eval/`
- ScenePlan construction: `scripts/t2a/data/`

Configs live in `stable_audio_tools/configs/`. Paths are environment placeholders, not machine mounts. See the repository [README](../../README.md) and [docs/TRAINING.md](../../docs/TRAINING.md).
