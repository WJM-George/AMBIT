"""Shared text-conditioning utilities for T2A data and model paths."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


CaptionRegion = Tuple[int, int, int]
TokenizerSpec = Tuple[Any, int, Optional[dict]]


def _validate_token_alignment_inputs(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError(f"offsets must be [T,2], got {tuple(offsets.shape)}")
    if attention_mask.shape != offsets.shape[:1]:
        raise ValueError(
            "attention_mask must match offsets [T], got "
            f"{tuple(attention_mask.shape)} vs {tuple(offsets.shape[:1])}"
        )


def _explicit_character_region(
    text: str,
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    start = value.get("start")
    end = value.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError(
            f"{label} has invalid half-open character span "
            f"[{start!r}, {end!r}) for text length {len(text)}"
        )
    return start, end


def _assign_character_regions(
    text: str,
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    regions: Sequence[tuple[int, int, int]],
    *,
    require_all_regions: bool,
) -> torch.Tensor:
    """Map explicit half-open character spans onto tokenizer offsets."""

    _validate_token_alignment_inputs(offsets, attention_mask)
    region_ids = torch.zeros(offsets.shape[0], dtype=torch.long)
    offsets_cpu = offsets.detach().cpu()
    mask_cpu = attention_mask.detach().cpu().bool()
    assigned = [False] * len(regions)
    for token_index, (token_start, token_end) in enumerate(offsets_cpu.tolist()):
        if not mask_cpu[token_index] or token_end <= token_start:
            continue
        overlapping = [
            index
            for index, (span_start, span_end, _) in enumerate(regions)
            if token_start < span_end and token_end > span_start
        ]
        if len(overlapping) > 1:
            # Nested regions are applied in separate passes. Two regions in
            # one pass therefore indicate corrupt/overlapping source spans.
            raise ValueError(
                "one caption token overlaps multiple explicit regions: "
                f"token=[{token_start},{token_end}) regions={overlapping}"
            )
        if overlapping:
            region_index = overlapping[0]
            region_ids[token_index] = int(regions[region_index][2])
            assigned[region_index] = True
    if require_all_regions and not all(assigned):
        missing = [index for index, value in enumerate(assigned) if not value]
        raise ValueError(
            "caption tokenization truncated or failed to align explicit "
            f"regions {missing}; increase text.max_length or repair spans"
        )
    return region_ids


def build_source_region_ids(
    text: str,
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    source_regions: Sequence[Mapping[str, Any]],
    *,
    max_sources: int = 4,
) -> torch.Tensor:
    """Build token ids for persistent Spatial-CoT source slots.

    Zero means ordinary/padding text and ``slot + 1`` denotes the caption span
    belonging to ``source_<slot>``. Exact spans come from the versioned caption
    overlay; no source identity is inferred from punctuation or a regex.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("source-regional caption text must be non-empty")
    if isinstance(max_sources, bool) or int(max_sources) <= 0:
        raise ValueError("max_sources must be positive")
    if not isinstance(source_regions, Sequence) or isinstance(
        source_regions, (str, bytes)
    ):
        raise TypeError("source_regions must be a sequence")
    if not source_regions:
        raise ValueError("source-regional caption requires at least one source span")

    spans: list[tuple[int, int, int]] = []
    seen_ids: set[str] = set()
    seen_slots: set[int] = set()
    for index, region in enumerate(source_regions):
        start, end = _explicit_character_region(
            text, region, label=f"source_regions[{index}]"
        )
        source_id = region.get("source_id")
        slot = region.get("source_slot")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"source_regions[{index}] lacks source_id")
        if source_id in seen_ids:
            raise ValueError(f"duplicate source-region id {source_id!r}")
        if (
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or not 0 <= slot < int(max_sources)
        ):
            raise ValueError(
                f"source_regions[{index}].source_slot must be in "
                f"[0,{int(max_sources)}), got {slot!r}"
            )
        if slot in seen_slots:
            raise ValueError(f"duplicate source-region slot {slot}")
        if source_id not in {f"source_{slot}", f"s{slot}"}:
            raise ValueError(
                f"source id/slot mismatch: {source_id!r} vs slot {slot}"
            )
        seen_ids.add(source_id)
        seen_slots.add(slot)
        spans.append((start, end, slot + 1))

    ordered = sorted(spans)
    for left, right in zip(ordered, ordered[1:]):
        if left[1] > right[0]:
            raise ValueError(
                "source caption regions overlap: "
                f"[{left[0]},{left[1]}) and [{right[0]},{right[1]})"
            )
    return _assign_character_regions(
        text,
        offsets,
        attention_mask,
        spans,
        require_all_regions=True,
    )


def build_explicit_speech_region_ids(
    text: str,
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    source_regions: Sequence[Mapping[str, Any]],
    transcript_regions: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Build DiT-compatible cue/transcript ids from exact overlay spans.

    Source phrases containing a transcript receive cue id 1, and the nested
    transcript receives id 2. This is used only by Spatial-CoT structured
    captions; the historical regex path remains byte-for-byte unchanged for
    existing DiT checkpoints.
    """

    by_source: dict[str, tuple[int, int]] = {}
    for index, region in enumerate(source_regions):
        source_id = region.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"source_regions[{index}] lacks source_id")
        if source_id in by_source:
            raise ValueError(f"duplicate source-region id {source_id!r}")
        by_source[source_id] = _explicit_character_region(
            text, region, label=f"source_regions[{index}]"
        )

    cue_spans: list[tuple[int, int, int]] = []
    transcript_spans: list[tuple[int, int, int]] = []
    seen_transcripts: set[str] = set()
    for index, region in enumerate(transcript_regions):
        source_id = region.get("source_id")
        if not isinstance(source_id, str) or source_id not in by_source:
            raise ValueError(
                f"transcript_regions[{index}] names unknown source {source_id!r}"
            )
        if source_id in seen_transcripts:
            raise ValueError(f"duplicate transcript region for {source_id!r}")
        start, end = _explicit_character_region(
            text, region, label=f"transcript_regions[{index}]"
        )
        source_start, source_end = by_source[source_id]
        if start < source_start or end > source_end:
            raise ValueError(
                f"transcript region for {source_id!r} is outside its source span"
            )
        seen_transcripts.add(source_id)
        cue_spans.append((source_start, source_end, 1))
        transcript_spans.append((start, end, 2))

    region_ids = _assign_character_regions(
        text,
        offsets,
        attention_mask,
        cue_spans,
        require_all_regions=bool(cue_spans),
    )
    if transcript_spans:
        transcript_ids = _assign_character_regions(
            text,
            offsets,
            attention_mask,
            transcript_spans,
            require_all_regions=True,
        )
        transcript_mask = transcript_ids.ne(0)
        region_ids[transcript_mask] = transcript_ids[transcript_mask]
    return region_ids


def find_speech_quote_regions(text: str) -> List[CaptionRegion]:
    """Return conservative character spans for speech-aware caption regions.

    Region ids are 0=ordinary/spatial/padding, 1=speech/event cue and 2=quoted
    speech.  Unknown caption templates intentionally return no spans.
    """

    if not isinstance(text, str) or not text:
        return []

    for match in re.finditer(r'["“]', text):
        first_quote = match.start()
        prefix = text[:first_quote].rstrip()
        if not prefix:
            continue

        if re.search(r"\bsays\s*$", prefix, flags=re.IGNORECASE):
            event_start = 0
            while event_start < first_quote and text[event_start].isspace():
                event_start += 1
            event_end = first_quote
            while event_end > event_start and text[event_end - 1].isspace():
                event_end -= 1
        elif re.search(r"\bSpoken words:\s*$", prefix, flags=re.IGNORECASE):
            cue = re.search(r"\bSpoken words:\s*$", prefix, flags=re.IGNORECASE)
            assert cue is not None
            event_start = cue.start()
            event_end = cue.end()
        else:
            continue

        close_quote = next(
            (
                index
                for index in range(first_quote + 1, len(text))
                if text[index] in {'"', "”"}
            ),
            None,
        )
        if close_quote is None or close_quote <= first_quote + 1:
            continue
        return [
            (event_start, event_end, 1),
            (first_quote + 1, close_quote, 2),
        ]

    return []


def build_caption_region_ids(
    text: str,
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    strategy: str = "speech_quote_v1",
) -> torch.Tensor:
    """Build one token-aligned region-id vector from tokenizer offsets."""

    region_ids = torch.zeros(offsets.shape[0], dtype=torch.long)
    if strategy != "speech_quote_v1":
        return region_ids
    spans = find_speech_quote_regions(text)
    if not spans:
        return region_ids

    offsets_cpu = offsets.cpu()
    mask_cpu = attention_mask.cpu().bool()
    for token_index, (token_start, token_end) in enumerate(offsets_cpu.tolist()):
        if not mask_cpu[token_index] or token_end <= token_start:
            continue
        for span_start, span_end, region_id in spans:
            if token_start < span_end and token_end > span_start:
                region_ids[token_index] = region_id
                break
    return region_ids


def build_batch_caption_region_ids(
    texts: Sequence[str],
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    strategy: str = "speech_quote_v1",
) -> torch.Tensor:
    """Batch form of :func:`build_caption_region_ids`."""

    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError(f"offsets must be [B,T,2], got {tuple(offsets.shape)}")
    if attention_mask.shape != offsets.shape[:2]:
        raise ValueError(
            "attention_mask must match offsets [B,T], got "
            f"{tuple(attention_mask.shape)} vs {tuple(offsets.shape[:2])}"
        )
    if len(texts) != offsets.shape[0]:
        raise ValueError("text batch length does not match offsets")

    return torch.stack(
        [
            build_caption_region_ids(
                text,
                offsets[index],
                attention_mask[index],
                strategy=strategy,
            )
            for index, text in enumerate(texts)
        ],
        dim=0,
    )


def _conditioner_tokenizer_spec(conditioner: Any) -> Optional[TokenizerSpec]:
    if not hasattr(conditioner, "tokenizer") or not hasattr(conditioner, "max_length"):
        return None
    region_config = None
    if getattr(conditioner, "caption_region_embedding", False):
        region_config = {
            "enabled": True,
            "strategy": getattr(
                conditioner, "caption_region_strategy", "speech_quote_v1"
            ),
        }
    return conditioner.tokenizer, int(conditioner.max_length), region_config


def collect_conditioner_tokenizers(
    model: Any,
    model_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, TokenizerSpec]:
    """Collect worker-side tokenizer specs from DiT or Transfusion T2A models.

    Existing ``MultiConditioner`` models expose a mapping of metadata keys to
    conditioners.  The Transfusion wrapper owns a single ``qwen_conditioner``;
    its metadata key is config-selectable and defaults to ``prompt``.
    """

    tokenizers: Dict[str, TokenizerSpec] = {}
    multi_conditioner = getattr(model, "conditioner", None)
    for key, conditioner in getattr(multi_conditioner, "conditioners", {}).items():
        spec = _conditioner_tokenizer_spec(conditioner)
        if spec is not None:
            tokenizers[key] = spec

    qwen_conditioner = getattr(model, "qwen_conditioner", None)
    spec = _conditioner_tokenizer_spec(qwen_conditioner)
    if spec is not None:
        metadata_key = "prompt"
        if model_config is not None:
            model_section = model_config.get("model", {})
            if isinstance(model_section, Mapping):
                text_section = model_section.get("text", {})
                if isinstance(text_section, Mapping):
                    metadata_key = str(text_section.get("metadata_key", metadata_key))
        tokenizers.setdefault(metadata_key, spec)
    tokenizer = getattr(model, "tokenizer", None)
    max_prompt_tokens = getattr(model, "max_prompt_tokens", None)
    if tokenizer is not None and max_prompt_tokens is not None:
        metadata_key = "prompt"
        if model_config is not None:
            model_section = model_config.get("model", {})
            if isinstance(model_section, Mapping):
                text_section = model_section.get("text", {})
                if isinstance(text_section, Mapping):
                    metadata_key = str(text_section.get("metadata_key", metadata_key))
        tokenizers.setdefault(
            metadata_key, (tokenizer, int(max_prompt_tokens), None)
        )
    return tokenizers


def tokenize_text_metadata(text: str, spec: Any) -> Dict[str, torch.Tensor]:
    """Tokenize one metadata string from a worker-safe conditioner spec.

    Both the historical ``(tokenizer, max_length)`` tuple and the current
    ``(tokenizer, max_length, region_config)`` tuple are accepted.  Keeping this
    normalization in one place prevents the dataset and model paths from
    silently drifting when an optional token-aligned feature is added.
    """

    if not isinstance(text, str):
        raise TypeError(f"text metadata must be a string, got {type(text).__name__}")
    if not isinstance(spec, (tuple, list)) or len(spec) not in (2, 3):
        raise ValueError(
            "tokenizer spec must be (tokenizer, max_length) or "
            "(tokenizer, max_length, region_config)"
        )

    tokenizer = spec[0]
    max_length = int(spec[1])
    if max_length <= 0:
        raise ValueError(f"tokenizer max_length must be positive, got {max_length}")
    region_config = spec[2] if len(spec) == 3 else None
    wants_region_ids = bool(
        isinstance(region_config, Mapping)
        and region_config.get("enabled", False)
    )

    tokenizer_kwargs = {
        "truncation": True,
        "max_length": max_length,
        "padding": "max_length",
        "return_tensors": "pt",
    }
    if wants_region_ids:
        tokenizer_kwargs["return_offsets_mapping"] = True

    try:
        encoded = tokenizer(text, **tokenizer_kwargs)
    except (NotImplementedError, TypeError, ValueError) as exc:
        if wants_region_ids:
            raise RuntimeError(
                "caption-region conditioning requires a fast tokenizer with "
                "return_offsets_mapping support"
            ) from exc
        raise

    tokenized = {
        "input_ids": torch.as_tensor(encoded["input_ids"]).squeeze(0),
        "attention_mask": torch.as_tensor(encoded["attention_mask"]).squeeze(0),
    }
    if wants_region_ids:
        offsets = torch.as_tensor(encoded["offset_mapping"]).squeeze(0)
        tokenized["region_ids"] = build_caption_region_ids(
            text,
            offsets,
            tokenized["attention_mask"],
            strategy=str(region_config.get("strategy", "speech_quote_v1")),
        )
    return tokenized
