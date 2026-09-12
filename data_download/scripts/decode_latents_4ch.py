"""Compatibility CLI; preserve the legacy default of also exporting WAV previews."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.vae.eval.decode_latents_4ch import *
from scripts.vae.eval.decode_latents_4ch import main as _decode_main


def main():
    return _decode_main(default_listen_wav=True)


if __name__ == "__main__":
    main()
