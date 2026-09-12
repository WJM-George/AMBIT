"""Shared lazy Qwen caption model loading for audio and audiovisual inputs."""
import logging


def build_model(model_path: str, attn: str):
    """Import heavy deps lazily so this file is importable without them installed."""
    import torch  # noqa: F401
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    logging.info("Loading %s (attn=%s) ...", model_path, attn)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", attn_implementation=attn,
    )
    # Captioner is thinker-only, but guard anyway: we never need the speech talker.
    if hasattr(model, "disable_talker"):
        try:
            model.disable_talker()
        except Exception as exc:  # noqa: BLE001
            logging.warning("disable_talker() failed (continuing): %s", exc)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor

