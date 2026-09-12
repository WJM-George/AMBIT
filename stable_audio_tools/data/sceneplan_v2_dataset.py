"""Fail-closed ScenePlan latent dataset for semantic text plus 4+4 controls."""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

from .model_sceneplan import (
    compile_model_44_controls,
    compile_model_renderer_caption,
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
    tokenize_model_semantic_caption,
)


VALID_P9_MARKER_STATES = frozenset(
    {
        "P9_complete_frozen_waiting_for_user_acceptance",
        "P9_complete_frozen_ready_for_P10_preflight",
    }
)


class ScenePlanV2Dataset(torch.utils.data.Dataset):
    """Read immutable latents and compile clean semantic plus local controls.

    The frozen SQLite database keeps random DDP sampling cheap without
    exploding 1.1M examples into individual JSON/NPY files. Every returned
    latent is padded (never cropped) to the configured 432- or 648-frame
    envelope and carries an exact loss padding mask. Qwen receives semantic text only; four categorical event
    tracks and four geometric trajectory tracks are returned independently.
    """

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def __init__(
        self,
        index_path: str | Path,
        *,
        tokenizer_spec: Any,
        expected_num_samples: int,
        index_num_samples: int | None = None,
        sample_ordinals: Sequence[int] | None = None,
        ordinal_start: int | None = None,
        ordinal_stop: int | None = None,
        latent_crop_length: int = 432,
        caption_max_tokens: int = 512,
        random_crop: bool = False,
        require_frozen: bool = True,
        speech_timing_index_path: str | Path | None = None,
        speech_timing_index_sha256: str | None = None,
        expected_speech_timing_rows: int | None = None,
        require_speech_timing: bool = False,
        semantic_caption_mode: str = "v2",
        semantic_caption_v2_probability: float = 1.0,
        semantic_caption_seed: int = 20260830,
    ) -> None:
        super().__init__()
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        if self.index_path.suffix != ".sqlite":
            raise ValueError("ScenePlan-v2 training index must be a frozen SQLite file")
        if random_crop:
            raise ValueError("ScenePlan-v2 forbids random latent crops")
        if int(latent_crop_length) not in (432, 648):
            raise ValueError(
                "ScenePlan-v2 batch padding ceiling must be 432 or 648 frames"
            )
        if int(caption_max_tokens) != 512:
            raise ValueError("ScenePlan semantic captions require the 512-token contract")
        if not isinstance(tokenizer_spec, (tuple, list)) or len(tokenizer_spec) not in (2, 3):
            raise ValueError("ScenePlan-v2 requires the Qwen prompt tokenizer spec")
        if require_frozen:
            dataset_root = self.index_path.parent.parent
            marker_path = dataset_root / "FROZEN_P9.json"
            if not marker_path.is_file():
                raise RuntimeError(
                    "ScenePlan-v2 formal loading is blocked until P9 freeze exists: "
                    f"{marker_path}"
                )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("schema") != "stable_audio_tools.sceneplan_v2_p9_marker"
                or marker.get("state") not in VALID_P9_MARKER_STATES
                or marker.get("p10_training_started") is not False
                or marker.get("p11_training_started", False) is not False
            ):
                raise RuntimeError(f"invalid ScenePlan-v2 P9 marker: {marker_path}")
            freeze_path = Path(marker["freeze_manifest"]).resolve(strict=True)
            if self._sha256_file(freeze_path) != marker["freeze_manifest_sha256"]:
                raise RuntimeError("ScenePlan-v2 freeze-manifest SHA256 mismatch")
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            training_summary_path = Path(freeze["training_index_summary"]).resolve(
                strict=True
            )
            if (
                self._sha256_file(training_summary_path)
                != freeze["training_index_summary_sha256"]
            ):
                raise RuntimeError("ScenePlan-v2 training-index summary SHA256 mismatch")
            training_summary = json.loads(
                training_summary_path.read_text(encoding="utf-8")
            )
            matching = [
                row
                for row in training_summary.get("splits", [])
                if Path(row["path"]).resolve(strict=True) == self.index_path
            ]
            if len(matching) != 1:
                raise RuntimeError("frozen training-index summary does not name this split")
            if self._sha256_file(self.index_path) != matching[0]["sha256"]:
                raise RuntimeError(f"frozen ScenePlan-v2 SQLite SHA256 mismatch: {self.index_path}")
        self.tokenizer = tokenizer_spec[0]
        model_max = int(tokenizer_spec[1])
        if model_max != int(caption_max_tokens):
            raise ValueError(
                f"Qwen/ScenePlan caption limits differ: {model_max} != {caption_max_tokens}"
            )
        self.latent_crop_length = int(latent_crop_length)
        self.caption_max_tokens = int(caption_max_tokens)
        self._connection: sqlite3.Connection | None = None
        self._timing_connection: sqlite3.Connection | None = None
        self.require_speech_timing = bool(require_speech_timing)
        self.speech_timing_index_path: Path | None = None
        self.semantic_caption_mode = str(semantic_caption_mode)
        self.semantic_caption_v2_probability = float(
            semantic_caption_v2_probability
        )
        self.semantic_caption_seed = int(semantic_caption_seed)
        if self.semantic_caption_mode not in {
            "v1",
            "v2",
            "deterministic_v1_v2",
            "paired_v1_v2",
        }:
            raise ValueError(
                "semantic_caption_mode must be v1, v2, "
                "deterministic_v1_v2, or paired_v1_v2"
            )
        if not 0.0 <= self.semantic_caption_v2_probability <= 1.0:
            raise ValueError("semantic_caption_v2_probability must be in [0,1]")
        expected_probability = {
            "v1": 0.0,
            "v2": 1.0,
        }.get(self.semantic_caption_mode)
        if (
            expected_probability is not None
            and self.semantic_caption_v2_probability != expected_probability
        ):
            raise ValueError(
                f"{self.semantic_caption_mode} mode requires "
                f"semantic_caption_v2_probability={expected_probability}"
            )
        if (
            self.semantic_caption_mode == "deterministic_v1_v2"
            and not 0.0 < self.semantic_caption_v2_probability < 1.0
        ):
            raise ValueError(
                "deterministic_v1_v2 requires a probability strictly between 0 and 1"
            )
        if (
            self.semantic_caption_mode == "paired_v1_v2"
            and self.semantic_caption_v2_probability != 0.5
        ):
            raise ValueError("paired_v1_v2 requires probability=0.5")
        self.semantic_caption_requires_epoch_key = (
            self.semantic_caption_mode == "paired_v1_v2"
        )

        if speech_timing_index_path is not None:
            timing_path = Path(speech_timing_index_path).expanduser().resolve(
                strict=True
            )
            if timing_path.suffix != ".sqlite":
                raise ValueError("speech timing sidecar must be an immutable SQLite file")
            if speech_timing_index_sha256 is not None:
                expected_digest = str(speech_timing_index_sha256)
                if len(expected_digest) != 64 or self._sha256_file(timing_path) != expected_digest:
                    raise RuntimeError("speech timing sidecar SHA256 mismatch")
            timing_connection = sqlite3.connect(
                f"file:{timing_path}?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )
            try:
                timing_connection.execute("PRAGMA query_only=ON")
                timing_metadata = dict(
                    timing_connection.execute("SELECT key,value FROM metadata")
                )
                required_timing_metadata = {
                    "schema": "stable_audio_tools.sceneplan_speech_timing_index",
                    "schema_version": "1",
                    "caption_compiler": "sceneplan_semantic_caption_v1",
                    "caption_max_tokens": "512",
                    "token_roles": "event_and_speech_-1_to_4",
                    "teacher": "relative_lexical_duration_partition_v1",
                }
                for key, expected in required_timing_metadata.items():
                    if timing_metadata.get(key) != expected:
                        raise RuntimeError(
                            f"speech timing metadata {key!r} changed: "
                            f"{timing_metadata.get(key)!r} != {expected!r}"
                        )
                timing_rows = int(
                    timing_connection.execute(
                        "SELECT COUNT(*) FROM speech_timing"
                    ).fetchone()[0]
                )
                if int(timing_metadata.get("rows", -1)) != timing_rows:
                    raise RuntimeError("speech timing sidecar row count is inconsistent")
                if (
                    expected_speech_timing_rows is not None
                    and timing_rows != int(expected_speech_timing_rows)
                ):
                    raise RuntimeError(
                        "speech timing sidecar has "
                        f"{timing_rows} rows, expected {expected_speech_timing_rows}"
                    )
            finally:
                timing_connection.close()
            self.speech_timing_index_path = timing_path
        elif self.require_speech_timing:
            raise ValueError(
                "require_speech_timing=true requires a speech timing sidecar"
            )
        if (
            self.speech_timing_index_path is not None
            and self.semantic_caption_mode != "v1"
        ):
            raise ValueError(
                "semantic caption v2 is incompatible with a v1 text-digest timing sidecar"
            )

        connection = self._open_connection()
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        contract_revision = str(metadata.get("contract_revision") or "")
        if contract_revision not in {"5", "6"}:
            raise RuntimeError(
                "the semantic 4+4 loader requires frozen ScenePlan contract revision 5 or 6, "
                f"got {contract_revision!r}"
            )
        self.contract_revision = int(contract_revision)
        self.structured_feature_dim = 5
        index_max_frames = 432 if contract_revision == "5" else 648
        sceneplan_schema_version = "1" if contract_revision == "5" else "2"
        if index_max_frames > self.latent_crop_length:
            raise RuntimeError(
                "ScenePlan-v2 index envelope exceeds the configured batch ceiling"
            )
        if contract_revision == "6" and self.speech_timing_index_path is not None:
            raise RuntimeError(
                "ScenePlan revision 6 forbids a speech timing sidecar"
            )
        required = {
            "schema": "stable_audio_tools.sceneplan_v2_training_index",
            "schema_version": "3",
            "contract_revision": contract_revision,
            "frozen": "true",
            "latent_channels": "64",
            "max_latent_frames": str(index_max_frames),
            "caption_max_tokens": str(self.caption_max_tokens),
            "random_crop": "false",
            "conditioning_contract_revision": "2",
            "model_sceneplan_schema_version": sceneplan_schema_version,
            "caption_compiler_version": "5",
            # This is the immutable P9 artifact dimension.  The new on-the-fly
            # model conditioner intentionally recompiles a 5-D trajectory.
            "structured_feature_dim": "9",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                if key == "frozen" and not require_frozen:
                    continue
                raise RuntimeError(
                    f"ScenePlan-v2 index metadata {key!r}={metadata.get(key)!r}, expected {expected!r}"
                )
        count = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        expected_index_count = int(
            index_num_samples
            if index_num_samples is not None
            else expected_num_samples
        )
        if count != expected_index_count:
            raise RuntimeError(
                f"ScenePlan-v2 index is incomplete: {count} != {expected_index_count}"
            )
        if int(metadata.get("rows", -1)) != count:
            raise RuntimeError("ScenePlan-v2 index metadata row count mismatch")
        has_ordinal_range = ordinal_start is not None or ordinal_stop is not None
        if has_ordinal_range and (
            ordinal_start is None or ordinal_stop is None
        ):
            raise ValueError(
                "ScenePlan ordinal range requires both ordinal_start and ordinal_stop"
            )
        if has_ordinal_range and sample_ordinals is not None:
            raise ValueError(
                "ScenePlan ordinal range and sample_ordinals are mutually exclusive"
            )
        self._ordinal_start = 0
        if sample_ordinals is None and not has_ordinal_range:
            if int(expected_num_samples) != count:
                raise RuntimeError(
                    "expected_num_samples differs from the full index without a "
                    "sample_ordinals or ordinal-range selection"
                )
            self._ordinals = None
            self._length = count
        elif has_ordinal_range:
            start = int(ordinal_start)
            stop = int(ordinal_stop)
            if not 0 <= start < stop <= count:
                raise RuntimeError(
                    "ScenePlan ordinal range is outside the frozen index: "
                    f"[{start}, {stop}) vs {count} rows"
                )
            if stop - start != int(expected_num_samples):
                raise RuntimeError(
                    "ScenePlan ordinal range length differs from "
                    f"expected_num_samples: {stop - start} != "
                    f"{expected_num_samples}"
                )
            self._ordinals = None
            self._ordinal_start = start
            self._length = stop - start
        else:
            ordinals = tuple(int(value) for value in sample_ordinals)
            if len(ordinals) != int(expected_num_samples):
                raise RuntimeError(
                    "sample_ordinals count differs from expected_num_samples"
                )
            if len(set(ordinals)) != len(ordinals):
                raise RuntimeError("sample_ordinals must be unique")
            if any(value < 0 or value >= count for value in ordinals):
                raise RuntimeError("sample_ordinals contains an out-of-range index")
            self._ordinals = ordinals
            self._length = len(ordinals)
        self.sample_weights: list[float] = []
        self._length_bucket_indices_cache: dict[int, tuple[int, ...]] | None = None
        connection.close()
        self._connection = None

    def _open_connection(self) -> sqlite3.Connection:
        uri = f"file:{self.index_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._open_connection()
        return self._connection

    def _timing_db(self) -> sqlite3.Connection:
        if self.speech_timing_index_path is None:
            raise RuntimeError("no speech timing sidecar is configured")
        if self._timing_connection is None:
            uri = f"file:{self.speech_timing_index_path}?mode=ro&immutable=1"
            self._timing_connection = sqlite3.connect(
                uri, uri=True, check_same_thread=False
            )
            self._timing_connection.execute("PRAGMA query_only=ON")
        return self._timing_connection

    def _speech_timing(self, sample_id: str) -> dict[str, Any] | None:
        if self.speech_timing_index_path is None:
            return None
        row = self._timing_db().execute(
            "SELECT payload_zlib FROM speech_timing WHERE sample_id=?",
            (str(sample_id),),
        ).fetchone()
        if row is None:
            return None
        return json.loads(zlib.decompress(row[0]))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_timing_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
        timing_connection = getattr(self, "_timing_connection", None)
        if timing_connection is not None:
            timing_connection.close()

    def __len__(self) -> int:
        return self._length

    def length_bucket_indices(self) -> dict[int, tuple[int, ...]]:
        """Return dataset-local indices for the immutable 432/648 buckets.

        This is intentionally derived from the frozen SQLite index rather than
        trusted from a second manifest.  The rank-aware batch sampler calls it
        once in the parent process, so workers never repeat the scan.
        """

        if self._length_bucket_indices_cache is not None:
            return self._length_bucket_indices_cache
        buckets: dict[int, list[int]] = {432: [], 648: []}
        connection = self._open_connection()
        try:
            if self._ordinals is None:
                start = self._ordinal_start
                stop = start + self._length
                rows = connection.execute(
                    """
                    SELECT ordinal, latent_frames_valid
                    FROM samples
                    WHERE ordinal >= ? AND ordinal < ?
                    ORDER BY ordinal
                    """,
                    (start, stop),
                )
                observed = 0
                for ordinal, valid_frames in rows:
                    bucket = 432 if int(valid_frames) <= 432 else 648
                    if int(valid_frames) > self.latent_crop_length:
                        raise RuntimeError(
                            f"ordinal {ordinal}: latent length exceeds dataset ceiling"
                        )
                    buckets[bucket].append(int(ordinal) - start)
                    observed += 1
                if observed != self._length:
                    raise RuntimeError(
                        "ScenePlan length-bucket scan did not cover the dataset"
                    )
            else:
                # Explicit ordinal subsets are used only by small pilots.  A
                # single immutable scan is faster and safer than issuing up to
                # one SQL query per selected row.
                wanted = {value: index for index, value in enumerate(self._ordinals)}
                observed: set[int] = set()
                for ordinal, valid_frames in connection.execute(
                    "SELECT ordinal, latent_frames_valid FROM samples ORDER BY ordinal"
                ):
                    local = wanted.get(int(ordinal))
                    if local is None:
                        continue
                    bucket = 432 if int(valid_frames) <= 432 else 648
                    if int(valid_frames) > self.latent_crop_length:
                        raise RuntimeError(
                            f"ordinal {ordinal}: latent length exceeds dataset ceiling"
                        )
                    buckets[bucket].append(local)
                    observed.add(int(ordinal))
                if len(observed) != self._length:
                    raise RuntimeError(
                        "ScenePlan subset length-bucket scan is incomplete"
                    )
        finally:
            connection.close()
        self._length_bucket_indices_cache = {
            key: tuple(values) for key, values in buckets.items() if values
        }
        return self._length_bucket_indices_cache

    def _tokenize(self, caption: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        tokenized = tokenize_model_semantic_caption(
            caption,
            self.tokenizer,
            max_length=self.caption_max_tokens,
        )
        return {
            "input_ids": torch.as_tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.as_tensor(
                tokenized["attention_mask"], dtype=torch.bool
            ),
            "event_source_ids": torch.as_tensor(
                tokenized["event_source_ids"], dtype=torch.int8
            ),
            "speech_source_ids": torch.as_tensor(
                tokenized["speech_source_ids"], dtype=torch.int8
            ),
            "speech_lexical_mask": torch.as_tensor(
                tokenized["speech_lexical_mask"], dtype=torch.bool
            ),
        }

    def _semantic_caption(
        self,
        scene_plan: Mapping[str, Any],
        *,
        sample_id: str,
        semantic_epoch: int | None = None,
    ) -> dict[str, Any]:
        """Select a resume-stable prompt template for one immutable sample."""

        use_v2 = self.semantic_caption_mode == "v2"
        if self.semantic_caption_mode == "deterministic_v1_v2":
            digest = hashlib.blake2b(
                f"{self.semantic_caption_seed}:{sample_id}".encode("utf-8"),
                digest_size=8,
                person=b"sp-prompt-v2",
            ).digest()
            fraction = int.from_bytes(digest, "big") / float(1 << 64)
            use_v2 = fraction < self.semantic_caption_v2_probability
        elif self.semantic_caption_mode == "paired_v1_v2":
            if semantic_epoch is None or int(semantic_epoch) < 0:
                raise RuntimeError(
                    "paired_v1_v2 requires the sampler's non-negative epoch key"
                )
            # Every sample flips surface syntax in adjacent epochs. This
            # teaches invariance on the same semantics instead of permanently
            # confounding compiler version with a random content subset.
            digest = hashlib.blake2b(
                f"{self.semantic_caption_seed}:{sample_id}".encode("utf-8"),
                digest_size=8,
                person=b"sp-prompt-v2",
            ).digest()
            use_v2 = (int.from_bytes(digest, "big") + int(semantic_epoch)) % 2 == 0
        compiler = (
            compile_model_semantic_caption_v2
            if use_v2
            else compile_model_semantic_caption
        )
        return compiler(scene_plan)

    def __getitem__(
        self, index: int | tuple[int, int]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        semantic_epoch: int | None = None
        if isinstance(index, (tuple, list)):
            if len(index) != 2:
                raise IndexError(f"invalid ScenePlan sample key: {index!r}")
            index, semantic_epoch = int(index[0]), int(index[1])
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        ordinal = (
            self._ordinal_start + int(index)
            if self._ordinals is None
            else int(self._ordinals[int(index)])
        )
        row = self._db().execute(
            """
            SELECT s.sample_id, s.model_num_samples, s.latent_frames_valid,
                   s.renderer_caption_zlib, s.scene_plan_zlib, l.path,
                   s.latent_key, s.latent_tensor_sha256
            FROM samples AS s
            JOIN latent_shards AS l ON l.id = s.latent_shard_id
            WHERE s.ordinal = ?
            """,
            (ordinal,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"frozen ScenePlan-v2 index has no ordinal {ordinal}")
        (
            sample_id,
            model_num_samples,
            valid_frames,
            caption_zlib,
            scene_plan_zlib,
            latent_path,
            latent_key,
            latent_tensor_sha256,
        ) = row
        if sample_id != latent_key:
            raise RuntimeError(f"{sample_id}: latent key/sample id mismatch")
        caption = json.loads(zlib.decompress(caption_zlib))
        scene_plan = json.loads(zlib.decompress(scene_plan_zlib))
        if caption != compile_model_renderer_caption(scene_plan):
            raise RuntimeError(
                f"{sample_id}: frozen model caption/ScenePlan compiler mismatch"
            )
        if not math.isclose(
            float(scene_plan["duration_sec"]),
            int(model_num_samples) / 44_100.0,
            rel_tol=0.0,
            abs_tol=1.1e-6,
        ):
            raise RuntimeError(f"{sample_id}: frozen model sample count mismatch")
        if math.ceil(int(model_num_samples) / 1024) != int(valid_frames):
            raise RuntimeError(f"{sample_id}: frozen valid latent frame count mismatch")
        if not 0 < int(valid_frames) <= self.latent_crop_length:
            raise RuntimeError(f"{sample_id}: invalid variable latent frame count")
        with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
            if latent_key not in handle.keys():
                raise RuntimeError(f"{sample_id}: latent key absent from shard")
            latent = handle.get_tensor(latent_key).clone()
        if latent.dtype != torch.float16 or tuple(latent.shape) != (64, int(valid_frames)):
            raise RuntimeError(f"{sample_id}: latent shape/dtype changed")
        if not torch.isfinite(latent).all().item():
            raise RuntimeError(f"{sample_id}: latent contains non-finite values")
        # The tensor checksum was exhaustively verified and frozen at P9. Keep
        # it in metadata for provenance without hashing every tensor on every epoch.
        if not isinstance(latent_tensor_sha256, str) or len(latent_tensor_sha256) != 64:
            raise RuntimeError(f"{sample_id}: invalid frozen latent checksum")
        padded = torch.zeros((64, self.latent_crop_length), dtype=torch.float16)
        padded[:, : int(valid_frames)] = latent
        padding_mask = torch.zeros(self.latent_crop_length, dtype=torch.bool)
        padding_mask[: int(valid_frames)] = True

        semantic_caption = self._semantic_caption(
            scene_plan,
            sample_id=str(sample_id),
            semantic_epoch=semantic_epoch,
        )
        prompt = self._tokenize(semantic_caption)
        speech_sources = [
            source
            for source in scene_plan["sources"]
            if source.get("kind") == "speech"
        ]
        if self.speech_timing_index_path is not None:
            timing = self._speech_timing(str(sample_id))
            target_fraction = torch.zeros(
                self.caption_max_tokens, dtype=torch.float32
            )
            target_mask = torch.zeros(
                self.caption_max_tokens, dtype=torch.bool
            )
            if speech_sources:
                if len(speech_sources) != 1:
                    raise RuntimeError(
                        f"{sample_id}: timing sidecar requires one formal speech source"
                    )
                if timing is None and self.require_speech_timing:
                    raise RuntimeError(
                        f"{sample_id}: formal speech row has no timing teacher"
                    )
                if timing is not None:
                    caption_digest = hashlib.sha256(
                        semantic_caption["text"].encode("utf-8")
                    ).hexdigest()
                    if timing.get("caption_text_sha256") != caption_digest:
                        raise RuntimeError(
                            f"{sample_id}: timing teacher caption changed"
                        )
                    valid_tokens = int(prompt["attention_mask"].sum().item())
                    if int(timing.get("caption_valid_tokens", -1)) != valid_tokens:
                        raise RuntimeError(
                            f"{sample_id}: timing teacher token count changed"
                        )
                    source_label = int(timing["source_label"])
                    positive_speech = prompt["speech_source_ids"] > 0
                    if (
                        not bool(positive_speech.any())
                        or not bool(
                            prompt["speech_source_ids"][positive_speech]
                            .eq(source_label)
                            .all()
                        )
                    ):
                        raise RuntimeError(
                            f"{sample_id}: timing teacher source label changed"
                        )
                    for item in timing.get("duration_targets") or ():
                        token_index = int(item["token_index"])
                        if not 0 <= token_index < self.caption_max_tokens:
                            raise RuntimeError(
                                f"{sample_id}: timing teacher token is out of range"
                            )
                        if int(prompt["input_ids"][token_index]) != int(
                            item["token_id"]
                        ):
                            raise RuntimeError(
                                f"{sample_id}: timing teacher token ID changed"
                            )
                        if not bool(prompt["speech_lexical_mask"][token_index]):
                            raise RuntimeError(
                                f"{sample_id}: timing teacher targets punctuation"
                            )
                        fraction = float(item["fraction"])
                        if not math.isfinite(fraction) or fraction <= 0.0:
                            raise RuntimeError(
                                f"{sample_id}: timing teacher fraction is invalid"
                            )
                        if bool(target_mask[token_index]):
                            raise RuntimeError(
                                f"{sample_id}: duplicate timing teacher token"
                            )
                        target_fraction[token_index] = fraction
                        target_mask[token_index] = True
                    if not torch.equal(
                        target_mask, prompt["speech_lexical_mask"]
                    ):
                        raise RuntimeError(
                            f"{sample_id}: timing teacher lexical support changed"
                        )
                    if not math.isclose(
                        float(target_fraction.sum().item()),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1.0e-5,
                    ):
                        raise RuntimeError(
                            f"{sample_id}: timing teacher fractions do not sum to one"
                        )
            elif timing is not None:
                raise RuntimeError(
                    f"{sample_id}: non-speech row unexpectedly has timing teacher"
                )
            prompt["speech_duration_target_fraction"] = target_fraction
            prompt["speech_duration_target_mask"] = target_mask
        elif self.require_speech_timing and speech_sources:
            raise RuntimeError(f"{sample_id}: required timing sidecar is unavailable")
        structured = compile_model_44_controls(
            scene_plan,
            model_num_samples=int(model_num_samples),
            latent_frames_valid=int(valid_frames),
        )
        if structured["source_trajectory_features"].shape != (
            4,
            int(valid_frames),
            self.structured_feature_dim,
        ):
            raise RuntimeError(f"{sample_id}: structured position/control shape changed")
        trajectory = np.zeros(
            (4, self.latent_crop_length, self.structured_feature_dim), dtype=np.float32
        )
        trajectory[:, : int(valid_frames)] = structured["source_trajectory_features"]
        event_ids = np.zeros((4, self.latent_crop_length), dtype=np.int8)
        event_ids[:, : int(valid_frames)] = structured["source_event_frame_ids"]
        speech_active = np.zeros(self.latent_crop_length, dtype=np.uint8)
        speech_active[: int(valid_frames)] = structured["speech_active_frame_mask"]
        controls = {
            "source_event_frame_ids": torch.as_tensor(event_ids, dtype=torch.int8),
            "source_trajectory_features": torch.as_tensor(trajectory),
            "frame_valid_mask": padding_mask.clone(),
            # Ground-truth-only supervision.  The local conditioner ignores it.
            "speech_active_frame_mask": torch.as_tensor(
                speech_active, dtype=torch.bool
            ),
        }
        metadata = {
            "sample_id": sample_id,
            # P11 is a task view over this exact P10 target state.  Keeping the
            # validated compact plan in worker metadata avoids a second SQLite
            # read and guarantees that Planner labels, semantic caption, and
            # 4+4 controls are compiled from one object.
            "model_sceneplan": scene_plan,
            "model_num_samples": int(model_num_samples),
            "prompt": prompt,
            "prompt_text": semantic_caption["text"],
            "semantic_caption_compiler_version": int(
                semantic_caption["compiler_version"]
            ),
            "semantic_caption_epoch": semantic_epoch,
            "sceneplan_44": controls,
            "padding_mask": [padding_mask],
            "seconds_start": 0.0,
            "seconds_total": float(model_num_samples) / 44_100.0,
            "latent_stored_length": int(valid_frames),
            "latent_crop_length": self.latent_crop_length,
            "latent_bucket_frames": 432 if int(valid_frames) <= 432 else 648,
            "latent_crop_start": 0,
            "latent_tensor_sha256": latent_tensor_sha256,
            "audio": padded,
        }
        return padded, metadata


__all__ = ["ScenePlanV2Dataset"]
