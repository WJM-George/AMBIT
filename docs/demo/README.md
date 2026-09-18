# AMBIT listening page

Open `index.html` locally (a static file server is enough).

Decoder: MIT KEMAR 8-virtual-speaker FOA binaural.

Samples are a listening-page shortlist:
3 generation + 3 editing examples after quality gates and an audio-preview
waveform/mel screen (no hard cutoff, GT-like envelope).
They are showcase clips, not a test-set average.

Rebuild:

```bash
python scripts/t2a/eval/build_ambit_demo_page.py
```
