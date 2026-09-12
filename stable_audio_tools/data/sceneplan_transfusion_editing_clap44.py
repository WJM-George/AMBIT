"""Training-only source/target supervision for the native Editing CLAP.

Old plans are labels in this pretraining reader. Neither plans, captions,
counterfactual descriptions nor asset identities are inference inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
import zlib

import torch
from safetensors import safe_open

from .model_sceneplan import validate_model_sceneplan
from .sceneplan_transfusion_editing import sha256_json

CLAP44_DATA_CONTRACT = "editing_clap44_paired_content_binding_supervision_v1"


def _text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _content(source: Mapping[str, Any]) -> dict[str, str]:
    if source["kind"] == "speech":
        return {"kind": "speech", "voice": _text(source["speaker_description"]), "words": _text(source["transcript"])}
    return {"kind": str(source["kind"]), "description": _text(source["description"])}


def _content_text(source: Mapping[str, Any]) -> str:
    value = _content(source)
    if value["kind"] == "speech":
        return f'Speech, {value["voice"]}, saying {json.dumps(value["words"], ensure_ascii=False)}'
    return f'{value["kind"]}: {value["description"]}'


def _structure(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [{"content": _content(s), "activity": s["activity"], "trajectory": s["trajectory"], "gain_db": s.get("gain_db", 0.0)} for s in plan["sources"]]
    # Sort whole bound events. Sorting content and positions independently
    # would incorrectly make a dog-left / bell-right swap a positive.
    return sorted(entries, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))


def _scene_text(plan: Mapping[str, Any]) -> str:
    sentences = []
    for s in sorted(plan["sources"], key=lambda x: (x["activity"]["onset_sec"], json.dumps(_content(x), sort_keys=True))):
        activity = s["activity"]
        trajectory = json.dumps(s["trajectory"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        sentences.append(f'{_content_text(s)}; active {activity["onset_sec"]:.4f} to {activity["offset_sec"]:.4f} seconds; trajectory {trajectory}; gain {float(s.get("gain_db", 0)):.2f} dB.')
    return f'{len(sentences)} audible sources; {plan["room"]["type"]} acoustic environment. ' + " ".join(sentences)


def _scene_key(plan: Mapping[str, Any]) -> str:
    return sha256_json({"room": plan["room"], "events": _structure(plan)})


def _asset_ids(members: Sequence[Mapping[str, Any]]) -> list[str]:
    values = set()
    for member in members:
        for record in (member, member.get("asset_ref", {})):
            for key in ("asset_id", "parent_asset_id", "identity_hash"):
                if record.get(key):
                    values.add(f'{key}:{record[key]}')
    if not values:
        raise ValueError("CLAP44 requires constituent identities to exclude false negatives")
    return sorted(values)


def make_clap44_label(plan: Mapping[str, Any], members: Sequence[Mapping[str, Any]], *, pair_id: str, role: str, operation: str) -> dict[str, Any]:
    validate_model_sceneplan(plan)
    if role not in {"source", "target"} or not pair_id:
        raise ValueError("invalid CLAP44 pair role/identity")
    content = sorted((_content(s) for s in plan["sources"]), key=lambda x: json.dumps(x, sort_keys=True))
    # Lists preserve multiplicity: two dogs are not equivalent to one dog.
    semantic_key = sha256_json(content)
    scene_key = _scene_key(plan)
    return {
        "pair_id": pair_id, "role": role, "operation": operation,
        "semantic_key": semantic_key, "scene_key": scene_key,
        "content_ids": sorted({sha256_json(x) for x in content}),
        "asset_ids": _asset_ids(members),
        "semantic_text": f'{len(content)} audible sources. ' + " ".join(sorted(_content_text(s) + "." for s in plan["sources"])),
        "scene_text": _scene_text(plan),
    }


def binding_counterfactuals(plan: Mapping[str, Any], *, maximum: int = 4) -> list[dict[str, str]]:
    """Anchor-local text negatives with identical content/count distributions.

    Swap complete trajectories or activity intervals between distinguishable
    events. Identical descriptions and equivalent resulting scenes are
    rejected. These captions must never enter other anchors' negative pools.
    """
    validate_model_sceneplan(plan)
    if maximum < 0:
        raise ValueError("counterfactual maximum must be nonnegative")
    if maximum == 0:
        return []
    original = _scene_key(plan); seen = {original}; results = []
    sources = plan["sources"]
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if _content(sources[i]) == _content(sources[j]):
                continue
            for field in ("trajectory", "activity"):
                if sources[i][field] == sources[j][field]:
                    continue
                if field == "activity" and abs(sources[i][field]["onset_sec"] - sources[j][field]["onset_sec"]) < 0.25:
                    continue
                changed = copy.deepcopy(plan)
                changed["sources"][i][field], changed["sources"][j][field] = copy.deepcopy(sources[j][field]), copy.deepcopy(sources[i][field])
                try:
                    validate_model_sceneplan(changed)
                except ValueError:
                    # A moving trajectory may cease to fit the swapped
                    # activity interval. Such a caption is not valid evidence.
                    continue
                key = _scene_key(changed)
                if key in seen:
                    continue
                seen.add(key)
                results.append({"kind": f"swapped_{field}", "scene_key": key, "scene_text": _scene_text(changed)})
                if len(results) >= maximum:
                    return results
    return results


class EditingCLAP44Dataset(torch.utils.data.Dataset):
    """Read both audio roles from immutable Editing train/validation pairs."""

    def __init__(self, index_path: str | Path, *, expected_rows: int, row_ordinals: Sequence[int] | None = None, verify_tensor_hashes: bool = True):
        self.path = Path(index_path).resolve(strict=True)
        self.verify_tensor_hashes = bool(verify_tensor_hashes)
        marker = json.loads(self.path.with_suffix(self.path.suffix + ".frozen.json").read_text())
        self.index_sha256 = str(marker["index_sha256"])
        if marker.get("state") != "materialized_complete_frozen" or marker.get("schema") != "sceneplan_transfusion_editing_training_index":
            raise ValueError("CLAP44 requires a frozen materialized Editing index")
        db = self._connect()
        try:
            metadata = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('schema','schema_version','state','split','rows')"))
        finally:
            db.close()
        if metadata.get("split") not in {"train", "validation"}:
            raise ValueError("CLAP44 development never opens the independent test split")
        if metadata.get("schema") != marker["schema"] or metadata.get("state") != marker["state"] or int(metadata["rows"]) != expected_rows:
            raise ValueError("CLAP44 index metadata/row-count mismatch")
        if marker.get("split") != metadata["split"] or int(marker.get("rows", -1)) != expected_rows or Path(marker.get("index_path", "")).resolve() != self.path:
            raise ValueError("CLAP44 frozen marker names a different split/index")
        self.split = metadata["split"]
        self.expected_rows = expected_rows
        self.ordinals = None if row_ordinals is None else tuple(int(x) for x in row_ordinals)
        if self.ordinals is not None and (not self.ordinals or len(set(self.ordinals)) != len(self.ordinals) or min(self.ordinals) < 0 or max(self.ordinals) >= expected_rows):
            raise ValueError("invalid CLAP44 row ordinals")
        self._connection = None

    def _connect(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        return db

    def __getstate__(self):
        value = dict(self.__dict__); value["_connection"] = None; return value

    def __del__(self):
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def __len__(self):
        return self.expected_rows if self.ordinals is None else len(self.ordinals)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ordinal = index if self.ordinals is None else self.ordinals[index]
        if self._connection is None:
            self._connection = self._connect()
        columns = "pair_id,operation,latent_frames_valid,model_num_samples,old_sceneplan_zlib,new_sceneplan_zlib,old_sceneplan_sha256,new_sceneplan_sha256,source_members_zlib,target_members_zlib,source_members_sha256,target_members_sha256,source_latent_path,source_latent_key,source_latent_tensor_sha256,target_latent_path,target_latent_key,target_latent_tensor_sha256"
        row = self._connection.execute(f"SELECT {columns} FROM pairs WHERE pair_ordinal=?", (ordinal,)).fetchone()
        if row is None:
            raise RuntimeError("CLAP44 pair ordinal is missing")
        frames = int(row["latent_frames_valid"])
        if not 1 <= frames <= 648 or (int(row["model_num_samples"]) + 1023) // 1024 != frames:
            raise ValueError("CLAP44 source/target geometry differs from the native VAE")
        views = []
        for role, plan_role in (("source", "old"), ("target", "new")):
            plan = json.loads(zlib.decompress(row[f"{plan_role}_sceneplan_zlib"]))
            members = json.loads(zlib.decompress(row[f"{role}_members_zlib"]))
            if sha256_json(plan) != row[f"{plan_role}_sceneplan_sha256"] or sha256_json(members) != row[f"{role}_members_sha256"]:
                raise RuntimeError("CLAP44 supervision provenance hash mismatch")
            with safe_open(row[f"{role}_latent_path"], framework="pt", device="cpu") as f:
                latent = f.get_tensor(row[f"{role}_latent_key"]).clone()
            if latent.dtype != torch.float16 or latent.shape != (64, frames) or not bool(torch.isfinite(latent).all()):
                raise ValueError("CLAP44 latent shape/dtype/finite check failed")
            if self.verify_tensor_hashes and hashlib.sha256(latent.contiguous().numpy().tobytes()).hexdigest() != row[f"{role}_latent_tensor_sha256"]:
                raise RuntimeError("CLAP44 latent tensor hash mismatch")
            views.append({"latent": latent, "label": make_clap44_label(plan, members, pair_id=row["pair_id"], role=role, operation=row["operation"]), "counterfactuals": binding_counterfactuals(plan)})
        return views


def collate_clap44(pairs: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("CLAP44 batches preserve both roles of every editing pair")
    views = [v for pair in pairs for v in pair]
    # Fixed maximum padding keeps all DDP audio tower shapes identical. The
    # model excludes padded frames before downsampling and attention/pooling.
    latent = torch.zeros(len(views), 64, 648, dtype=torch.float16)
    mask = torch.zeros(len(views), 648, dtype=torch.bool)
    negative_texts, negative_owners, negative_kinds = [], [], []
    for i, view in enumerate(views):
        frames = view["latent"].shape[-1]
        latent[i, :, :frames] = view["latent"]; mask[i, :frames] = True
        for negative in view["counterfactuals"]:
            negative_texts.append(negative["scene_text"]); negative_owners.append(i); negative_kinds.append(negative["kind"])
    return {"latent": latent, "mask": mask, "labels": [v["label"] for v in views], "negative_scene_texts": negative_texts, "negative_owners": torch.tensor(negative_owners, dtype=torch.long), "negative_kinds": negative_kinds}
