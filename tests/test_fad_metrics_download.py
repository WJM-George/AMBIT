import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from stable_audio_tools.training.metrics import fad_metrics


class _FakeResponse:
    def __init__(self, status_code, chunks=(), headers=None):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.headers = dict(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def iter_content(self, chunk_size):
        self.chunk_size = chunk_size
        yield from self._chunks


class ResumableDownloadTest(unittest.TestCase):
    def _paths(self):
        temporary_directory = tempfile.TemporaryDirectory()
        destination = Path(temporary_directory.name) / "model.pt"
        self.addCleanup(temporary_directory.cleanup)
        return destination

    def test_resumes_partial_file_when_server_honors_range(self):
        destination = self._paths()
        Path(str(destination) + ".tmp").write_bytes(b"abc")
        response = _FakeResponse(206, [b"def"], {"content-length": "3"})
        with patch.object(fad_metrics.requests, "get", return_value=response) as get:
            fad_metrics._download_file_resumable(
                "https://example.test/model", str(destination), description="model"
            )
        self.assertEqual(destination.read_bytes(), b"abcdef")
        self.assertFalse(Path(str(destination) + ".tmp").exists())
        self.assertEqual(get.call_args.kwargs["headers"], {"Range": "bytes=3-"})

    def test_restarts_when_server_ignores_range(self):
        destination = self._paths()
        Path(str(destination) + ".tmp").write_bytes(b"stale")
        response = _FakeResponse(200, [b"fresh"], {"content-length": "5"})
        with patch.object(fad_metrics.requests, "get", return_value=response):
            fad_metrics._download_file_resumable(
                "https://example.test/model", str(destination), description="model"
            )
        self.assertEqual(destination.read_bytes(), b"fresh")

    def test_partial_response_is_not_published(self):
        destination = self._paths()
        response = _FakeResponse(200, [b"short"], {"content-length": "8"})
        with patch.object(fad_metrics.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "incomplete download"):
                fad_metrics._download_file_resumable(
                    "https://example.test/model",
                    str(destination),
                    description="model",
                )
        self.assertFalse(destination.exists())
        self.assertEqual(Path(str(destination) + ".tmp").read_bytes(), b"short")

    def test_complete_temporary_file_handles_range_416(self):
        destination = self._paths()
        Path(str(destination) + ".tmp").write_bytes(b"done")
        response = _FakeResponse(416, headers={"Content-Range": "bytes */4"})
        with patch.object(fad_metrics.requests, "get", return_value=response):
            fad_metrics._download_file_resumable(
                "https://example.test/model", str(destination), description="model"
            )
        self.assertEqual(destination.read_bytes(), b"done")


class ClapCheckpointResolutionTest(unittest.TestCase):
    def test_bare_filename_is_independent_of_process_cwd(self):
        with tempfile.TemporaryDirectory() as workdir, patch.dict(
            os.environ, {}, clear=False
        ), patch("os.getcwd", return_value=workdir):
            os.environ.pop("STABLE_AUDIO_TOOLS_CACHE_DIR", None)
            name, checkpoint = fad_metrics._resolve_clap_checkpoint(
                "630k-audioset-fusion-best.pt"
            )
        repo_root = Path(fad_metrics.__file__).resolve().parents[3]
        self.assertEqual(name, "630k-audioset-fusion-best.pt")
        self.assertEqual(
            checkpoint,
            repo_root / "load" / "clap_score" / name,
        )

    def test_cache_root_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
            os.environ,
            {"STABLE_AUDIO_TOOLS_CACHE_DIR": cache_dir},
        ):
            name, checkpoint = fad_metrics._resolve_clap_checkpoint(
                "630k-audioset-fusion-best.pt"
            )
        self.assertEqual(checkpoint, Path(cache_dir) / "clap_score" / name)

    def test_explicit_checkpoint_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "630k-audioset-fusion-best.pt"
            name, checkpoint = fad_metrics._resolve_clap_checkpoint(requested)
        self.assertEqual(name, requested.name)
        self.assertEqual(checkpoint, requested)


if __name__ == "__main__":
    unittest.main()
