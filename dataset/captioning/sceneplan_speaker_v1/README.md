# ScenePlan speech speaker registry v1

This stage enriches the 512,000 formal TTS donors without changing their
audio, split, or exact transcript.

* LibriTTS speakers with LibriTTS-P annotations use the three official human
  voice-profile labels plus the official per-utterance pitch, speed, and
  energy tags.
* Official metadata gaps, mixed-gender LibriTTS speaker IDs, and the ten
  HiFiTTS speakers use Qwen3-Omni-Instruct on three deterministic clean
  excerpts per profile.
* The finalized Parquet registry has exactly one row per formal speech asset.
  ScenePlan builders join it by `asset_id`; no FOA or latent is regenerated.

Run `build_speaker_profile_inputs.py`, annotate the emitted JSONL with the
existing four-worker Transformers runner and `speaker_profile_prompt_v1.txt`,
then run `finalize_speaker_registry.py`.
