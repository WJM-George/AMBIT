from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm
from urllib3.exceptions import ProtocolError

from audio_dataset_download.catalog import DATASETS, DatasetSpec


ROOT = Path(os.environ.get("AUDIO_DATASET_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
SECONDARY_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
CACHE_ROOT = Path(os.environ.get("AUDIO_DATASET_CACHE_ROOT", os.environ.get("AMBIT_CACHE_ROOT", "cache")))
TMP_ROOT = Path(os.environ.get("AUDIO_DATASET_TMP", os.environ.get("AMBIT_CACHE_ROOT", "cache/tmp")))
CODE_ROOT = Path(__file__).resolve().parent
SLS_URI = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1"
SECONDARY_DATASET_KEYS = set(
    item.strip()
    for item in os.environ.get(
        "AUDIO_DATASET_SECONDARY_KEYS",
        "bewo_1m,sphere360,audio_flan,spatial_librispeech,audioset",
    ).split(",")
    if item.strip()
)


def configure_environment() -> None:
    os.environ.setdefault("AUDIO_DATASET_ROOT", str(ROOT))
    os.environ.setdefault("AUDIO_DATASET_SECONDARY_ROOT", str(SECONDARY_ROOT))
    os.environ.setdefault("AUDIO_DATASET_CACHE_ROOT", str(CACHE_ROOT))
    os.environ.setdefault("AUDIO_DATASET_TMP", str(TMP_ROOT))
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
    default_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if default_token_path.exists():
        os.environ.setdefault("HF_TOKEN_PATH", str(default_token_path))
    os.environ.setdefault("TMPDIR", str(TMP_ROOT))
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def validate_storage_paths() -> None:
    roots = [ROOT.resolve(), SECONDARY_ROOT.resolve()]
    code_root = CODE_ROOT.resolve()
    for root in roots:
        if root == code_root or code_root in root.parents:
            raise RuntimeError(
                f"Dataset storage root must not point inside the downloader code package: {code_root}. "
                "Use ${AMBIT_DATA_ROOT}, ${AMBIT_DATA_ROOT}, or another external storage directory."
            )


def ensure_layout() -> None:
    validate_storage_paths()
    for path in [
        ROOT,
        ROOT / "datasets",
        ROOT / "logs",
        ROOT / "manifests",
        SECONDARY_ROOT,
        SECONDARY_ROOT / "datasets",
        CACHE_ROOT,
        Path(os.environ["HF_HOME"]),
        Path(os.environ["HF_HUB_CACHE"]),
        Path(os.environ["HF_DATASETS_CACHE"]),
        TMP_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(dataset_key: str) -> Path:
    log_path = ROOT / "logs" / f"{dataset_key}-{timestamp()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def token() -> str | bool | None:
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return value or None


def dataset_dir(spec: DatasetSpec) -> Path:
    root = SECONDARY_ROOT if spec.key in SECONDARY_DATASET_KEYS else ROOT
    return root / "datasets" / spec.key


def write_manifest(spec: DatasetSpec, payload: dict[str, Any]) -> Path:
    manifest = {
        "dataset": asdict(spec),
        "created_at_utc": timestamp(),
        "root": str(ROOT),
        "secondary_root": str(SECONDARY_ROOT),
        "dataset_path": str(dataset_dir(spec)),
        "cache_root": str(CACHE_ROOT),
        "tmp_root": str(TMP_ROOT),
        **payload,
    }
    path = ROOT / "manifests" / f"{spec.key}-{timestamp()}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote manifest: %s", path)
    return path


def discover_splits(repo_id: str) -> list[dict[str, Any]]:
    url = "https://datasets-server.huggingface.co/splits"
    response = requests.get(url, params={"dataset": repo_id}, headers=auth_headers(), timeout=60)
    response.raise_for_status()
    return response.json().get("splits", [])


def list_parquet_urls(repo_id: str, config: str, split: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{config}/{split}"
    response = requests.get(url, headers=auth_headers(), timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return [item["url"] if isinstance(item, dict) and "url" in item else str(item) for item in data]
    if isinstance(data, dict):
        files = data.get("parquet_files") or data.get("files") or data.get("urls") or []
        return [item["url"] if isinstance(item, dict) and "url" in item else str(item) for item in files]
    raise ValueError(f"Unexpected parquet API response for {repo_id}/{config}/{split}: {type(data)}")


def auth_headers() -> dict[str, str]:
    hf_token = token()
    if isinstance(hf_token, str):
        return {"Authorization": f"Bearer {hf_token}"}
    return {}


def filename_from_url(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    return name or fallback


# Transient network failures that should trigger a resume rather than abort the
# whole run. Large Zenodo/HF files routinely drop the connection mid-stream and
# raise one of these; we retry with an HTTP Range request from the bytes already
# on disk instead of restarting from zero.
RETRYABLE_DOWNLOAD_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ProtocolError,
    IncompleteRead,
)


def download_url(
    url: str,
    output_path: Path,
    desc: str,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 10,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")

    attempt = 0
    while True:
        existing = partial.stat().st_size if partial.exists() else 0
        headers = dict(auth_headers())
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                if existing and response.status_code == 200:
                    logging.info("Server ignored Range; restarting %s", output_path)
                    partial.unlink(missing_ok=True)
                    existing = 0
                response.raise_for_status()
                total_header = response.headers.get("content-length")
                total = int(total_header) + existing if total_header else None
                mode = "ab" if existing else "wb"
                with partial.open(mode) as handle, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bar.update(len(chunk))
            break
        except RETRYABLE_DOWNLOAD_ERRORS as exc:
            attempt += 1
            done = partial.stat().st_size if partial.exists() else 0
            if attempt > max_retries:
                raise RuntimeError(
                    f"Giving up on {url} after {max_retries} retries "
                    f"({done} bytes downloaded to {partial})."
                ) from exc
            wait = min(60, 2 ** attempt)
            logging.warning(
                "Download interrupted (%s) at %d bytes; resuming, retry %d/%d in %ds: %s",
                type(exc).__name__, done, attempt, max_retries, wait, url,
            )
            time.sleep(wait)

    partial.replace(output_path)


def download_hf_snapshot(spec: DatasetSpec, allow_patterns: list[str] | None, ignore_patterns: list[str] | None, max_workers: int) -> dict[str, Any]:
    if not spec.repo_id:
        raise ValueError(f"{spec.key} has no Hugging Face repo_id")
    dest = dataset_dir(spec) / "snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading Hugging Face snapshot %s into %s", spec.repo_id, dest)
    try:
        local_path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            token=token(),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
        )
    except HfHubHTTPError as exc:
        if spec.gated:
            raise RuntimeError(
                f"{spec.repo_id} looks gated or private. Accept access on Hugging Face and set HF_TOKEN, "
                "or run `hf auth login`, then rerun this script."
            ) from exc
        raise
    return {"mode": "hf_snapshot", "local_path": local_path}


def download_hf_parquet(spec: DatasetSpec, config_filter: str | None, split_filter: str | None) -> dict[str, Any]:
    if not spec.repo_id:
        raise ValueError(f"{spec.key} has no Hugging Face repo_id")
    split_rows = discover_splits(spec.repo_id)
    selected = [
        row
        for row in split_rows
        if (not config_filter or row.get("config") == config_filter)
        and (not split_filter or row.get("split") == split_filter)
    ]
    if not selected:
        raise RuntimeError(f"No splits matched dataset={spec.repo_id} config={config_filter} split={split_filter}")

    results: list[dict[str, Any]] = []
    for row in selected:
        config = row["config"]
        split = row["split"]
        urls = list_parquet_urls(spec.repo_id, config, split)
        split_dir = dataset_dir(spec) / "parquet" / config / split
        split_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading %d parquet files for %s/%s/%s", len(urls), spec.repo_id, config, split)
        for index, url in enumerate(urls):
            filename = filename_from_url(url, f"part-{index:05d}.parquet")
            output_path = split_dir / filename
            if output_path.exists():
                logging.info("Skipping existing parquet file: %s", output_path)
                continue
            download_url(url, output_path, desc=f"{spec.key}:{config}:{split}:{index}")
        results.append({"config": config, "split": split, "count": len(urls), "path": str(split_dir)})
    return {"mode": "hf_parquet", "splits": results}


def spatial_librispeech_sample_ids(metadata_path: Path, start: int, count: int | None, download_all: bool) -> list[int]:
    if download_all:
        import pyarrow.parquet as pq

        table = pq.read_table(metadata_path, columns=["sample_id"])
        return sorted(int(value) for value in table.column("sample_id").to_pylist())
    if count is None:
        return []
    return list(range(start, start + count))


def download_spatial_librispeech(
    include_noise: bool,
    start: int,
    count: int | None,
    download_all: bool,
    workers: int,
) -> dict[str, Any]:
    spec = DATASETS["spatial_librispeech"]
    dest = dataset_dir(spec)
    metadata_path = dest / "metadata" / "metadata.parquet"
    if not metadata_path.exists():
        download_url(f"{SLS_URI}/metadata.parquet", metadata_path, "spatial-librispeech:metadata")
    else:
        logging.info("Skipping existing metadata: %s", metadata_path)

    sample_ids = spatial_librispeech_sample_ids(metadata_path, start, count, download_all)
    downloaded: list[dict[str, Any]] = []
    if not sample_ids:
        logging.info("No sample count supplied; downloaded metadata only.")
        return {"mode": "apple_spatial_librispeech", "metadata": str(metadata_path), "samples": downloaded}

    def download_sample(sample_id: int) -> list[dict[str, Any]]:
        sample_name = f"{sample_id:06d}.flac"
        records: list[dict[str, Any]] = []
        speech_path = dest / "ambisonics" / sample_name
        if not speech_path.exists():
            download_url(f"{SLS_URI}/ambisonics/{sample_name}", speech_path, f"sls:speech:{sample_name}")
        records.append({"kind": "ambisonics", "sample_id": sample_id, "path": str(speech_path)})

        if include_noise:
            noise_path = dest / "noise_ambisonics" / sample_name
            if not noise_path.exists():
                download_url(f"{SLS_URI}/noise_ambisonics/{sample_name}", noise_path, f"sls:noise:{sample_name}")
            records.append({"kind": "noise_ambisonics", "sample_id": sample_id, "path": str(noise_path)})
        return records

    logging.info("Downloading %d Spatial LibriSpeech sample ids with %d workers", len(sample_ids), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_sample, sample_id) for sample_id in sample_ids]
        for future in as_completed(futures):
            downloaded.extend(future.result())
            if len(downloaded) % 1000 == 0:
                logging.info("Downloaded/verified %d files so far", len(downloaded))

    return {
        "mode": "apple_spatial_librispeech",
        "metadata": str(metadata_path),
        "sample_id_count": len(sample_ids),
        "file_count": len(downloaded),
        "include_noise": include_noise,
    }


def zenodo_file_list(record_id: str) -> list[dict[str, Any]]:
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json().get("files", [])


def download_zenodo(
    spec: DatasetSpec,
    *,
    extract: bool = False,
    file_filter: list[str] | None = None,
) -> dict[str, Any]:
    if not spec.zenodo_record_id:
        raise ValueError(f"{spec.key} has no zenodo_record_id")
    dest = dataset_dir(spec)
    archive_dir = dest / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    files = zenodo_file_list(spec.zenodo_record_id)
    if file_filter:
        needles = [item.lower() for item in file_filter]
        files = [row for row in files if any(n in row["key"].lower() for n in needles)]
    if not files:
        raise RuntimeError(f"No Zenodo files matched record={spec.zenodo_record_id} filter={file_filter}")

    downloaded: list[dict[str, Any]] = []
    for row in files:
        name = row["key"]
        output_path = archive_dir / name
        if output_path.exists() and output_path.stat().st_size == row.get("size", -1):
            logging.info("Skipping existing archive: %s", output_path)
            downloaded.append({"file": name, "path": str(output_path), "status": "skipped"})
            continue
        url = row["links"]["self"]
        logging.info("Downloading Zenodo file %s (%.2f GB)", name, row.get("size", 0) / 1e9)
        download_url(url, output_path, desc=f"{spec.key}:{name}")
        downloaded.append({"file": name, "path": str(output_path), "status": "ok", "bytes": row.get("size")})

    payload: dict[str, Any] = {
        "mode": "zenodo",
        "record_id": spec.zenodo_record_id,
        "archive_dir": str(archive_dir),
        "files": downloaded,
    }
    if extract:
        payload["extracted_dir"] = str(extract_zenodo_archives(spec, archive_dir))
    return payload


def extract_zenodo_archives(spec: DatasetSpec, archive_dir: Path) -> Path:
    """Unpack Zenodo zip archives. Split noisy train uses 7z when available."""
    extract_dir = dataset_dir(spec) / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    singles = [
        "FSDKaggle2019.meta.zip",
        "FSDKaggle2019.doc.zip",
        "FSDKaggle2019.audio_train_curated.zip",
        "FSDKaggle2019.audio_test.zip",
    ]
    for name in singles:
        archive = archive_dir / name
        if not archive.exists():
            logging.warning("Missing archive, skip extract: %s", archive)
            continue
        target = extract_dir / name.replace(".zip", "")
        if target.exists() and any(target.iterdir()):
            logging.info("Already extracted: %s", target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        logging.info("Extracting %s -> %s", archive, target)
        subprocess.run(["unzip", "-q", str(archive), "-d", str(target)], check=True)

    noisy_first = archive_dir / "FSDKaggle2019.audio_train_noisy.z01"
    noisy_parts = sorted(archive_dir.glob("FSDKaggle2019.audio_train_noisy.z*"))
    noisy_zip = archive_dir / "FSDKaggle2019.audio_train_noisy.zip"
    noisy_target = extract_dir / "FSDKaggle2019.audio_train_noisy"
    if noisy_target.exists() and any(noisy_target.iterdir()):
        logging.info("Already extracted: %s", noisy_target)
        return extract_dir

    if noisy_first.exists() or noisy_zip.exists():
        noisy_target.mkdir(parents=True, exist_ok=True)
        if shutil_which("7z"):
            entry = str(noisy_first if noisy_first.exists() else noisy_zip)
            logging.info("Extracting split noisy archive with 7z: %s", entry)
            subprocess.run(["7z", "x", "-y", entry, f"-o{noisy_target}"], check=True)
        elif shutil_which("zip"):
            merged = archive_dir / "FSDKaggle2019.audio_train_noisy_merged.zip"
            if not merged.exists():
                logging.info("Merging split zip parts with zip -s 0")
                subprocess.run(
                    ["zip", "-s", "0", str(noisy_zip), "--out", str(merged)],
                    cwd=str(archive_dir),
                    check=True,
                )
            logging.info("Extracting merged noisy archive -> %s", noisy_target)
            subprocess.run(["unzip", "-q", str(merged), "-d", str(noisy_target)], check=True)
        else:
            raise RuntimeError(
                "Install p7zip-full (7z) or zip to extract FSDKaggle2019.audio_train_noisy split archive."
            )
    return extract_dir


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def clone_github_repo(spec: DatasetSpec) -> dict[str, Any]:
    if not spec.repo_id:
        raise ValueError(f"{spec.key} has no repository URL")
    dest = dataset_dir(spec) / "repo"
    if (dest / ".git").exists():
        logging.info("Updating existing Git repository: %s", dest)
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Cloning %s into %s", spec.repo_id, dest)
        subprocess.run(["git", "clone", spec.repo_id, str(dest)], check=True)
    return {"mode": "github_repo", "local_path": str(dest)}


def run_dataset(
    dataset_key: str,
    mode: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    config: str | None = None,
    split: str | None = None,
    max_workers: int = 8,
    sls_start: int = 0,
    sls_count: int | None = None,
    sls_all: bool = False,
    sls_include_noise: bool = False,
    sls_workers: int = 8,
    zenodo_extract: bool = False,
    zenodo_files: list[str] | None = None,
) -> Path:
    configure_environment()
    ensure_layout()
    spec = DATASETS[dataset_key]
    log_path = setup_logging(dataset_key)
    logging.info("Dataset: %s", spec.name)
    logging.info("Description: %s", spec.description)
    if spec.note:
        logging.info("Note: %s", spec.note)
    logging.info("Log file: %s", log_path)

    if mode == "auto":
        mode = spec.kind
    if mode == "hf_snapshot":
        payload = download_hf_snapshot(spec, allow_patterns, ignore_patterns, max_workers)
    elif mode == "hf_parquet":
        payload = download_hf_parquet(spec, config or None, split or None)
    elif mode == "apple_spatial_librispeech":
        payload = download_spatial_librispeech(sls_include_noise, sls_start, sls_count, sls_all, sls_workers)
    elif mode == "github_repo":
        payload = clone_github_repo(spec)
    elif mode == "zenodo":
        payload = download_zenodo(spec, extract=zenodo_extract, file_filter=zenodo_files)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return write_manifest(spec, payload)


def comma_patterns(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    patterns: list[str] = []
    for value in values:
        patterns.extend(item.strip() for item in value.split(",") if item.strip())
    return patterns or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and organize audio/spatial datasets.")
    parser.add_argument("dataset", choices=sorted(DATASETS), help="Dataset key to download.")
    parser.add_argument(
        "--mode",
        choices=["auto", "hf_snapshot", "hf_parquet", "apple_spatial_librispeech", "github_repo", "zenodo"],
        default="auto",
        help="Download strategy. Use hf_parquet for split-organized parquet files.",
    )
    parser.add_argument("--config", help="HF config to download in parquet mode.")
    parser.add_argument("--split", help="HF split to download in parquet mode.")
    parser.add_argument("--allow-pattern", action="append", help="Snapshot allow pattern, repeatable or comma-separated.")
    parser.add_argument("--ignore-pattern", action="append", help="Snapshot ignore pattern, repeatable or comma-separated.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for Hugging Face snapshot downloads.")
    parser.add_argument("--sls-start", type=int, default=0, help="Spatial LibriSpeech first sample id.")
    parser.add_argument("--sls-count", type=int, help="Spatial LibriSpeech sample count. Omit to download metadata only.")
    parser.add_argument("--sls-all", action="store_true", help="Download every Spatial LibriSpeech sample listed in metadata.")
    parser.add_argument("--sls-include-noise", action="store_true", help="Also download Spatial LibriSpeech noise samples.")
    parser.add_argument("--sls-workers", type=int, default=8, help="Parallel workers for Spatial LibriSpeech file downloads.")
    parser.add_argument("--extract", action="store_true", help="Unpack Zenodo zip archives after download.")
    parser.add_argument(
        "--zenodo-file",
        action="append",
        help="Zenodo: download only archives whose names contain this substring (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dataset(
        dataset_key=args.dataset,
        mode=args.mode,
        allow_patterns=comma_patterns(args.allow_pattern),
        ignore_patterns=comma_patterns(args.ignore_pattern),
        config=args.config,
        split=args.split,
        max_workers=args.max_workers,
        sls_start=args.sls_start,
        sls_count=args.sls_count,
        sls_all=args.sls_all,
        sls_include_noise=args.sls_include_noise,
        sls_workers=args.sls_workers,
        zenodo_extract=args.extract,
        zenodo_files=args.zenodo_file,
    )


if __name__ == "__main__":
    main()
