# VAE 4ch Five-Band Energy Statistics

- Samples: 50,000
- Seed: `20260710`
- Preprocessing: full-clip resample to 44.1 kHz, deterministic 4 s crop, joint peak 0.9
- STFT: n_fft=2048, hop=512, Hann, center=False
- No persistent decoded-audio or STFT cache was created.

## Training mixture (`mixture_all`)

All-channel mean; energy share is conditioned on active crops.
Active crop fraction (W / XYZ / all): 97.382% / 97.832% / 97.847%

| Band | Power mean dB | Power p50 dB | Mean share | Share p10-p90 | Density mean dB/Hz |
|---|---:|---:|---:|---:|---:|
| 0-250 Hz | -29.720 | -25.868 | 35.783% | 1.885-81.844% | -54.045 |
| 250 Hz-2 kHz | -26.575 | -23.364 | 50.198% | 9.775-86.480% | -59.169 |
| 2-8 kHz | -35.378 | -32.860 | 11.915% | 0.690-31.250% | -73.207 |
| 8-14 kHz | -55.934 | -51.350 | 1.615% | 0.000-2.820% | -93.763 |
| 14-22.05 kHz | -77.455 | -81.095 | 0.489% | 0.000-0.116% | -116.534 |

## All non-speech (`non_speech`)

All-channel mean; energy share is conditioned on active crops.
Active crop fraction (W / XYZ / all): 99.614% / 99.690% / 99.690%

| Band | Power mean dB | Power p50 dB | Mean share | Share p10-p90 | Density mean dB/Hz |
|---|---:|---:|---:|---:|---:|
| 0-250 Hz | -26.546 | -23.141 | 39.902% | 1.175-87.617% | -50.575 |
| 250 Hz-2 kHz | -24.015 | -22.446 | 44.966% | 7.092-86.657% | -56.468 |
| 2-8 kHz | -32.079 | -30.544 | 13.249% | 0.549-36.369% | -69.868 |
| 8-14 kHz | -49.888 | -45.634 | 1.666% | 0.000-3.227% | -87.677 |
| 14-22.05 kHz | -69.418 | -65.725 | 0.217% | 0.000-0.178% | -108.479 |

## All speech (`speech_all`)

All-channel mean; energy share is conditioned on active crops.
Active crop fraction (W / XYZ / all): 94.185% / 95.170% / 95.208%

| Band | Power mean dB | Power p50 dB | Mean share | Share p10-p90 | Density mean dB/Hz |
|---|---:|---:|---:|---:|---:|
| 0-250 Hz | -34.265 | -28.492 | 29.607% | 3.576-63.257% | -59.013 |
| 250 Hz-2 kHz | -30.242 | -24.102 | 58.043% | 22.358-86.343% | -63.035 |
| 2-8 kHz | -40.102 | -35.512 | 9.915% | 0.867-28.377% | -77.990 |
| 8-14 kHz | -64.593 | -65.090 | 1.538% | 0.000-2.014% | -102.480 |
| 14-22.05 kHz | -88.965 | -93.954 | 0.898% | 0.000-0.029% | -128.069 |

## Combined TTS (`tts_combined`)

All-channel mean; energy share is conditioned on active crops.
Active crop fraction (W / XYZ / all): 87.819% / 89.882% / 89.961%

| Band | Power mean dB | Power p50 dB | Mean share | Share p10-p90 | Density mean dB/Hz |
|---|---:|---:|---:|---:|---:|
| 0-250 Hz | -41.487 | -30.893 | 26.943% | 2.024-60.552% | -67.076 |
| 250 Hz-2 kHz | -37.314 | -25.363 | 55.822% | 12.886-86.638% | -70.504 |
| 2-8 kHz | -45.245 | -35.873 | 11.842% | 1.222-29.689% | -83.249 |
| 8-14 kHz | -56.831 | -48.865 | 3.403% | 0.013-9.663% | -94.835 |
| 14-22.05 kHz | -83.159 | -92.579 | 1.990% | 0.000-0.200% | -122.312 |

## Per-Source Median Energy Share (All Channels)

| Source | Active all | 0-250 | 250-2k | 2-8k | 8-14k | 14-22.05k |
|---|---:|---:|---:|---:|---:|---:|
| Existing non-speech FOA | 99.920% | 25.639% | 48.663% | 6.683% | 0.138% | 0.000% |
| Expansion non-speech FOA | 99.620% | 38.460% | 41.150% | 5.879% | 0.210% | 0.003% |
| Spatial LibriSpeech | 100.000% | 27.466% | 62.973% | 3.629% | 0.001% | 0.000% |
| TTS SDB QC-clean subset | 89.980% | 21.818% | 59.996% | 5.910% | 0.380% | 0.000% |
| TTS SDC QC-clean subset | 89.930% | 20.855% | 60.026% | 6.201% | 0.418% | 0.000% |
