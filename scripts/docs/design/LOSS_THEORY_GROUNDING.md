# FOA VAE Loss — Signal-Processing Theory Grounding

> Companion to [`LOSS_DESIGN_SPATIAL_PHASE.md`](LOSS_DESIGN_SPATIAL_PHASE.md).
> That file records *what we decided to build and why it is novel*. This file
> records *the signal-processing venue/journal theory each loss term is derived
> from*, so every term in the objective is "from a paper, adapted to a problem
> we actually observed", not hand-tuned and back-justified.
>
> Structure per loss family: **Problem observed → Theory (venue/journal) →
> How our code follows it → Full-band check**. Every claim is cross-checked
> against the running config
> [`stable_audio_4ch_vae_ds1024_z64_phase_scm.json`](../../../stable_audio_tools/configs/model_configs/autoencoders/ablation_arms/stable_audio_4ch_vae_ds1024_z64_phase_scm.json)
> and the implementations in
> [`losses/auraloss.py`](../../../stable_audio_tools/training/losses/auraloss.py) and
> [`losses/semantic.py`](../../../stable_audio_tools/training/losses/semantic.py).

The three reconstruction targets and their theory anchors:

| Perceptual axis | Loss term (code) | Primary theory anchor (venue) |
|---|---|---|
| **Frequency / timbre (full band)** | `MultiResolutionSTFTLoss` (K-weighted, adaptive-log) | Schwär & Müller, *IEEE Signal Processing Letters* 2023 |
| **Phase / temporal fine structure** | `FrequencyGatedIFGDPhaseLoss` (= `if_gd_loss` + `normalized_complex_distance_loss`) | Ai & Ling, *ICASSP* 2023; Takamichi et al., *IWAENC* 2018 / *Signal Processing* 2020 |
| **Space (FOA, storage-compressible)** | `FOASpatialCovarianceLoss` | Vilkamo et al., *JAES* 2013; Duong et al., *IEEE TASLP* 2010 |

All frequency-dependent weights are one shared curve family `g(f)` (§4), whose
corner frequencies come from psychoacoustics journals (§4, §5).

---

## 1. Frequency (full-band) loss — the anchor term

### 1.1 Problem observed
The reconstruction anchor must constrain the **entire** spectrum, not just the
region a mel filterbank emphasises. Two failure modes we care about: (a) a plain
multi-scale magnitude loss can give *uninformative gradients* for exactly the
tonal/frequency content it is supposed to fix, and (b) naive `log(x+ε)`
compression pushes near-silent bins to large negative values and destabilises
the loss landscape. We need a full-band term whose gradient is trustworthy at
every frequency.

### 1.2 Theory (venue/journal)
- **Schwär & Müller, "Multi-Scale Spectral Loss Revisited," *IEEE Signal
  Processing Letters*, vol. 30, pp. 1712–1716, 2023** (DOI 10.1109/LSP.2023.3333205).
  This is the signal-processing **journal** reference for how to configure a
  multi-scale spectral (MSS) loss so that it *provides informative gradients*.
  Key results we use:
  1. Aggregating spectrograms at **multiple window sizes** trades time vs.
     frequency resolution so signals must match at all resolutions
     simultaneously — this is what makes the loss *full-band* by construction.
  2. **Magnitude compression matters**: they show plain `log(x+ε)` amplifies
     spectral-leakage fluctuations (large negative values at window zeros) and
     recommend a **non-negative** compression `log(x/·+1)` (their "C2", offset 1
     instead of ε) for a smoother landscape.
  3. Window type/size and the matrix distance (L1 vs L2) measurably change the
     Gradient-Sign Ranking Accuracy.
- Multi-resolution STFT as a reconstruction objective: **Yamamoto, Song & Kim,
  "Parallel WaveGAN," ICASSP 2020**; packaged in **Steinmetz & Reiss,
  "auraloss," DMRN 2020** (our code is a fork of auraloss).
- **K-weighting** perceptual pre-filter: **ITU-R BS.1770** loudness weighting,
  applied as an FIR prefilter so the full-band magnitude error is weighted by
  perceived loudness rather than raw energy.

### 1.3 How our code follows it
Running config (`spectral.mrstft`):
```
fft_sizes  = [2048, 1024, 512, 256, 128, 64, 32]   # 7 resolutions
hop_sizes  = [512, 256, 128, 64, 32, 16, 8]
win_lengths= [2048, 1024, 512, 256, 128, 64, 32]
perceptual_weighting = true                        # K-weighting FIR
weight mrstft = 1.0                                 # the anchor
```
- Seven resolutions from 2048 down to **32 samples** → the small windows give the
  time/high-frequency resolution, the large windows the low-frequency resolution;
  together they cover DC→Nyquist. This is exactly the MSS "match at all
  resolutions" argument of Schwär & Müller §II.
- The log-magnitude term uses the **adaptive non-negative compression** from
  `STFTMagnitudeLoss.forward` in [`auraloss.py`](../../../stable_audio_tools/training/losses/auraloss.py):
  `log(x_mag / σ + 1)` with `σ = sqrt(std(x)² + std(y)²)`. The `+1` offset is
  the Schwär & Müller C2 recommendation (output ≥ 0), not `log(x+ε)`.
- `perceptual_weighting=True` builds `FIRFilter(filter_type="kw")` (ITU-R BS.1770
  K-weighting) and applies it to both signals before the magnitude comparison.
- 4-channel handling: the four FOA channels are folded into the batch dimension,
  so the magnitude loss is computed per physical channel (W, Y, Z, X) — no
  virtual down-mix.

### 1.4 Full-band check
This term is the **only** full-band, equally-resolved reconstruction anchor
(`weight = 1.0`). The other spectral terms are perceptual *shaping on top of it*,
never replacements:
- `band_mel` (in the base config) and the HF-overshoot term reweight specific
  regions; they are switched relative to this anchor, not summed as the primary
  fidelity signal.
- `high_frequency_overshoot` (asymmetric, `fmin=8000`, cosine-decayed 0.2→0.05)
  is a one-sided penalty that only fires when recon **exceeds** the target above
  8 kHz — it fixes HF hallucination without competing with the full-band anchor.

**Statement for the paper:** the full-band frequency fidelity is carried by the
K-weighted 7-resolution MSS loss configured per Schwär & Müller (SPL 2023);
every other spectral term is an additive perceptual correction.

---

## 2. Phase loss — temporal fine structure (`FrequencyGatedIFGDPhaseLoss`)

### 2.1 Problem observed
Magnitude-only losses ignore phase, but phase carries transient sharpness and
(for multichannel) inter-channel timing. Directly regressing phase fails because
of **phase wrapping**: the raw error `|φ̂−φ|` is not the true circular error, so
L1/L2 on phase suffer *error expansion*. We need a phase objective that is
wrapping-safe and that concentrates supervision where phase is perceptually
encoded.

### 2.2 Theory (venue/journal)
- **Ai & Ling, "Neural Speech Phase Prediction Based on Parallel Estimation
  Architecture and Anti-Wrapping Losses," *ICASSP* 2023** (arXiv:2211.15974,
  DOI 10.1109/ICASSP49357.2023.10096553); extended in **arXiv:2403.17378**
  ("Low-Latency…", extended journal version). They prove an **anti-wrapping
  function** must be **even, 2π-periodic and monotonic** on `(-π, π]`, and define
  three losses on the *wrapped* phase:
  - **Instantaneous Phase (IP)** loss — direct wrapped-phase error;
  - **Group Delay (GD)** loss — anti-wrapped **difference along frequency**;
  - **Instantaneous Angular Frequency (IAF)** loss — anti-wrapped **difference
    along time**.
  Ablations show GD and IAF (the derivative terms) are what remove the
  "spectral horizontal stripe" noise and dull HF — i.e. the derivative-of-phase
  terms matter most.
- **Takamichi, Saito, Takamune, Kitamura & Saruwatari, "Phase reconstruction
  from amplitude spectrograms based on von-Mises-distribution DNN," *IWAENC*
  2018** (arXiv:1807.03474, DOI 10.1109/IWAENC.2018.8521313), extended as a
  **journal** paper in ***Signal Processing*, vol. 169, 107368, 2020**
  ("…directional-statistics deep neural networks"). Because phase is a **circular
  variable**, its natural likelihood is the **von Mises distribution**; the
  maximum-likelihood loss is the **negative cosine of the phase error**,
  `1 − cos(φ̂−φ)`. They add a **group-delay loss** for the same reason we do:
  group delay is more perceptually/structurally reliable than raw phase.

### 2.3 How our code follows it
Our loss uses the **unit-phasor cosine** form (implemented in `if_gd_loss` and
`FrequencyGatedIFGDPhaseLoss._gated_phasor_term`):
- Form adjacent-element products of complex STFT values and normalise to unit
  phasors `U = S_hi · conj(S_lo) / (|S_hi||S_lo|)`, then
  `cosine_distance = 1 − Re[U_pred · conj(U_ref)]`.
- **This is simultaneously (a) the von Mises MLE loss** `1 − cos(Δφ)` of Takamichi
  et al. and **(b) inherently anti-wrapping** in the Ai & Ling sense: cosine of a
  phase difference is even, 2π-periodic and monotonic on `(-π, π]`, so working on
  phasors never sees a wrap boundary and cannot suffer error expansion. We get
  the Ai & Ling anti-wrapping guarantee *for free* from the phasor formulation,
  instead of applying their explicit `f_AW(x)=x−2π·round(x/2π)`.
- **Axis mapping is exactly theirs:**
  - `dim=-1` (adjacent **time** frames) = Ai & Ling **IAF** / instantaneous
    frequency term (`Ut` in `if_gd_loss`);
  - `dim=-2` (adjacent **frequency** bins) = Ai & Ling **GD** / Takamichi
    group-delay term (`Uf`).
- **Energy weighting:** each phasor distance is weighted by a *detached*
  geometric-mean magnitude `sqrt(|S_hi^p||S_lo^p||S_hi^r||S_lo^r|)`, so silent
  bins (whose phase is meaningless) get ~0 weight — the standard practice of
  weighting phase error by amplitude reliability.
- **Complex-distance regulariser** `normalized_complex_distance_loss`
  (`log(|Δ|²/σ+1)`, the SAME "L_cd" term, `cd_weight=1`) couples magnitude and
  phase; it reuses the same non-negative log compression as §1.
- No K-weighting prefilter here **by design**: any linear filter applied to both
  signals cancels inside the phasor products (it would only rescale the magnitude
  weights, which the gates already do).

### 2.4 Full-band handling + storage-compression angle
Phase is supervised at **all four resolutions** `n_ffts=[2048,1024,512,256]`, but
with **frequency-gated weights** rather than flat:
- `λ_IF(f) = g(f; fc=2000, β=2, floor=0.1)` and
  `λ_GD(f) = g(f; fc=8000, β=2, floor=0.25)` (config values).
- **Why gate instead of flat (this is the "phase compresses storage" argument):**
  auditory-nerve **phase-locking** to the temporal fine structure degrades above
  ~1.5–4 kHz (**Palmer & Russell, *Hearing Research* 24:1–15, 1986**), and
  fine-structure **ITD/IPD** sensitivity vanishes just above ~1.4 kHz
  (**Brughera, Dunai & Hartmann, *JASA* 133(5):2839–2855, 2013**). So
  high-frequency *fine-structure phase* is largely not perceived and does not
  need to be preserved bit-for-bit → the decoder can spend capacity elsewhere.
  Group-delay audibility extends higher (transient alignment; **Blauert & Laws,
  *JASA* 63(5):1478–1483, 1978**), which is why `λ_GD` has a higher corner
  (8 kHz) and larger floor (0.25) than `λ_IF`.
- The gates multiply the (detached) magnitude weights and the loss is a
  **weighted mean**, so turning gates on does **not** change the loss scale
  (safe to ramp in during resume).

**Difference vs. the references (state in paper):** Ai & Ling and Takamichi apply
their phase/GD losses **flat across frequency**; ours is the same estimator with
a **psychoacoustic frequency gate** (continuous curve, not hard bands).

---

## 3. Spatial loss — FOA spatial covariance (`FOASpatialCovarianceLoss`)

### 3.1 Problem observed
For 4-channel FOA `s(f,t)=[W,Y,Z,X]ᵀ`, what must survive the VAE is not the raw
per-channel waveform but the **spatial image**: direction, width/diffuseness,
inter-channel level and inter-channel **phase (IPD)**. A per-channel magnitude or
even the intensity-vector direction alone is incomplete (it misses reactive
intensity, the XYZ cross-terms and diffuseness). We need a single, complete
second-order spatial target.

### 3.2 Theory (venue/journal)
- **Vilkamo, Bäckström & Kuntz, "Optimized Covariance Domain Framework for
  Time–Frequency Processing of Spatial Audio," *Journal of the Audio Engineering
  Society (JAES)*, 61(6):403–411, 2013.** Central result we rely on: *the
  perceptually relevant spatial information in a frequency band is contained in
  the **covariance matrix** of the multichannel signal* (channel energies +
  inter-channel dependencies), because with the acoustic transfer path it forms
  the binaural cues the auditory system decodes. Any linear spatial rendering
  operates in the covariance domain (`C_y = A C Aᴴ`). **⇒ Matching the SCM on
  perceptual TF tiles matches every downstream linear renderer** (binaural,
  loudspeaker, rotation).
- **Duong, Vincent & Gribonval, "Under-Determined Reverberant Audio Source
  Separation Using a Full-Rank Spatial Covariance Model," *IEEE TASLP*,
  18(7):1830–1840, 2010** (DOI 10.1109/TASL.2010.2050716). Establishes the
  **full-rank spatial covariance matrix** as the complete second-order TF model
  of a spatial sound field (superior to rank-1 narrowband models). Justifies
  using the **full complex SCM**, not just the intensity vector.
- **Duplex theory** (Lord Rayleigh) + **Brughera et al., *JASA* 2013**: interaural
  **phase** cues are usable only below ~1.4–1.5 kHz; above that, spatial hearing
  relies on level/coherence. Grounds the IPD frequency gate.

### 3.3 How our code follows it
For each resolution `n_ffts=[2048,512]`, form the time-smoothed (`smooth_frames=5`)
`4×4` complex SCM `C(f,t)`, **trace-normalise** it (`C̃ = C/tr C`), and split the
error by a **polar decomposition** into three perceptual parts (config weights all
1.0 internally, ramped by outer `scm=0.015`):
- **level** — L1 between trace-normalised **diagonal** energy fractions
  (generalises the directional/omni energy-ratio = image width);
- **coherence** — L1 between off-diagonal **coherence magnitudes**
  `γ_ij = |C_ij|/sqrt(C_ii C_jj) ∈ [0,1]` (diffuseness / envelopment; the
  Duong full-rank statistic);
- **ipd** — cosine distance between off-diagonal **unit phasors**
  `U_ij = C_ij/|C_ij|` (inter-channel phase), **softly masked by the reference
  coherence** `γ_ref` (incoherent pairs have meaningless phase) and gated by
  `λ_IPD(f)=g(f; fc=1500, β=2, floor=0.1)` (duplex theory / Brughera 2013).
- All terms weighted by reference **trace energy** (per-item normalised) and
  reduced as weighted means ⇒ **bounded** (trace-1 PSD Frobenius distance ≤ 2)
  and **scale-invariant**, which is friendly to the GAN training balance.

**Relation to the intensity-vector loss:** the active-intensity direction loss
(`FOASpatialConsistencyLoss`, after FOA Tokenizer arXiv:2510.22241) compares
`Re{conj(W)·[X,Y,Z]}` = the **real part of the first SCM row** — a strict special
case. The SCM loss additionally constrains reactive intensity (imag part), the
XYZ block, and diffuseness (Duong full-rank argument).

### 3.4 Full-band check
SCM is matched across the whole band at both resolutions; only the **phase (IPD)**
sub-term is frequency-gated (down-weighted > ~1.5 kHz per duplex theory), while
**level and coherence remain full-band** — matching the psychoacoustic fact that
high-frequency spatial perception persists through level/coherence even when IPD
does not.

---

## 4. The shared frequency-gate family `g(f)` (one curve)

All frequency-dependent weights above are the **same** monotone log-frequency
logistic (`_logistic_frequency_gate` in `semantic.py`):

```
g(f; fc, β, floor) = floor + (1 − floor) / (1 + (f/fc)^β)        ∈ [floor, 1]
```

Smooth, bounded, `=1` at DC, decays to `floor` above corner `fc` with steepness
`β`. Only `(fc, β, floor)` change per use; each has a physical meaning and a
citation:

| Use | fc | β | floor | Psychoacoustic basis (journal) |
|---|---|---|---|---|
| `λ_IF` (phase, fine structure) | 2000 | 2 | 0.10 | Phase-locking roll-off, Palmer & Russell, *Hear. Res.* 1986 |
| `λ_GD` (phase, group delay) | 8000 | 2 | 0.25 | GD audibility to ~8 kHz, Blauert & Laws, *JASA* 1978 |
| `λ_IPD` (spatial, inter-channel phase) | 1500 | 2 | 0.10 | ITD/IPD upper limit ~1.4 kHz, Brughera et al., *JASA* 2013 |

Using one closed-form curve (vs. hard bands) is the continuous analogue of
Qwen-Music's band-mode idea, but done in the **loss** and grounded in these
three psychoacoustics papers.

---

## 5. Candidate additional losses with a journal theory base (proposals, not yet enabled)

Per the request to "find more journal articles and develop more suitable losses",
these are theory-grounded options we can ablate later. Listed with the exact
journal anchor and the specific problem each would address; none are wired in yet.

1. **PMSQE — perceptual speech-quality frequency loss.**
   *J. M. Martín-Doñas, A. M. Gomez, J. A. Gonzalez, A. M. Peinado, "A Deep
   Learning Loss Function Based on the Perceptual Evaluation of the Speech
   Quality," IEEE Signal Processing Letters, 25(11):1680–1684, 2018*
   (DOI 10.1109/LSP.2018.2871419). A **differentiable PESQ-like** frequency-domain
   loss (Bark loudness, symmetric+asymmetric disturbance). Would directly optimise
   the PESQ/DNSMOS axis we now only *measure* — a natural add for the speech arm.

2. **ERB/Bark perceptual reweighting of the band-mel curve.**
   *B. C. J. Moore & B. R. Glasberg, "Suggested formulae for calculating
   auditory-filter bandwidths and excitation patterns," JASA 74(3):750–753, 1983*
   (ERB) — replaces the empirically fitted band-mel weights of
   `BandWeightedMelSpectrogramLoss` with an ERB-derived excitation weighting, so
   the "presence-region bump" is a formula, not five hand-set numbers.

3. **Interaural-coherence (IC) preservation term.**
   *C. Faller & J. Merimaa, "Source localization in complex listening situations:
   Selection of binaural cues based on interaural coherence," JASA 116(5):3075–3089,
   2004.* Justifies an explicit IC-matching term (envelopment/robust-cue
   selection). Our SCM `coherence` sub-term already approximates this; the paper
   is the journal basis if we want to weight it up or report IC error.

4. **Reﬁne the MSS window set per Schwär & Müller (SPL 2023).**
   Same journal as §1: their analysis suggests **prime-sized** windows and a
   flat-top/Hann choice reduce spectral-leakage artefacts in the loss landscape.
   A drop-in config change (no new code) to test whether it improves convergence.

---

## 6. References (with identifiers)

Frequency / spectral:
- Schwär, Müller. "Multi-Scale Spectral Loss Revisited." *IEEE Signal Processing
  Letters* 30:1712–1716, 2023. DOI 10.1109/LSP.2023.3333205.
- Yamamoto, Song, Kim. "Parallel WaveGAN." *ICASSP* 2020.
- Steinmetz, Reiss. "auraloss: Audio-focused loss functions in PyTorch." *DMRN* 2020.
- ITU-R BS.1770 — K-weighting loudness.
- Défossez, Copet, Synnaeve, Adi. "High Fidelity Neural Audio Compression"
  (EnCodec). arXiv:2210.13438, 2022. (multi-scale mel precedent)

Phase:
- Ai, Ling. "Neural Speech Phase Prediction … Anti-Wrapping Losses." *ICASSP* 2023.
  arXiv:2211.15974, DOI 10.1109/ICASSP49357.2023.10096553. Extended: arXiv:2403.17378.
- Takamichi, Saito, Takamune, Kitamura, Saruwatari. "Phase reconstruction from
  amplitude spectrograms based on von-Mises-distribution DNN." *IWAENC* 2018.
  arXiv:1807.03474, DOI 10.1109/IWAENC.2018.8521313. Journal extension:
  *Signal Processing* 169:107368, 2020.
- Palmer, Russell. "Phase-locking in the cochlear nerve of the guinea-pig…"
  *Hearing Research* 24:1–15, 1986.
- Blauert, Laws. "Group delay distortions in electroacoustical systems."
  *JASA* 63(5):1478–1483, 1978.

Spatial:
- Vilkamo, Bäckström, Kuntz. "Optimized Covariance Domain Framework for
  Time–Frequency Processing of Spatial Audio." *JAES* 61(6):403–411, 2013.
- Duong, Vincent, Gribonval. "Under-Determined Reverberant Audio Source
  Separation Using a Full-Rank Spatial Covariance Model." *IEEE TASLP*
  18(7):1830–1840, 2010. DOI 10.1109/TASL.2010.2050716.
- Brughera, Dunai, Hartmann. "Human interaural time difference thresholds for
  sine tones: The high-frequency limit." *JASA* 133(5):2839–2855, 2013.
  DOI 10.1121/1.4795778.
- FOA Tokenizer. arXiv:2510.22241. (intensity-vector special case)
- Hirvonen, Namazi. "Compression of HOA with Multichannel RVQGAN."
  arXiv:2411.12008, *ICASSP* 2025. (broadband time-domain covariance L1 — the
  prior art our TF-complex SCM generalises)

Proposals (§5):
- Martín-Doñas, Gomez, Gonzalez, Peinado. "A Deep Learning Loss Function Based on
  the Perceptual Evaluation of the Speech Quality." *IEEE SPL* 25(11):1680–1684,
  2018. DOI 10.1109/LSP.2018.2871419.
- Moore, Glasberg. "Suggested formulae for calculating auditory-filter
  bandwidths and excitation patterns." *JASA* 74(3):750–753, 1983.
- Faller, Merimaa. "Source localization in complex listening situations…"
  *JASA* 116(5):3075–3089, 2004.

---

## 7. Symbol → paper map (quick reference)

| Code symbol (file) | Math | Paper it comes from |
|---|---|---|
| `MultiResolutionSTFTLoss` (auraloss.py) | Σ_r d(\|STFT_r(x)\|,\|STFT_r(ŷ)\|) | Schwär & Müller SPL'23; Yamamoto ICASSP'20 |
| `STFTMagnitudeLoss` `log(x/σ+1)` | non-neg log compression (C2) | Schwär & Müller SPL'23 §IV-C |
| `FIRFilter(filter_type="kw")` | K-weighting prefilter | ITU-R BS.1770 |
| `if_gd_loss` `Ut` term (dim=-1) | `1−cos(Δφ)` along time | Ai & Ling IAF; von Mises MLE (Takamichi) |
| `if_gd_loss` `Uf` term (dim=-2) | `1−cos(Δφ)` along freq | Ai & Ling GD; Takamichi group delay |
| `_logistic_frequency_gate` `λ_IF/λ_GD` | `g(f;fc,β,floor)` | Palmer&Russell'86; Blauert&Laws'78 |
| `FOASpatialCovarianceLoss` `C̃=C/tr C` | trace-norm SCM match | Vilkamo JAES'13; Duong TASLP'10 |
| … `level` (diagonal) | energy fractions | Vilkamo JAES'13 |
| … `coherence` `γ_ij` | \|C_ij\|/√(C_ii C_jj) | Duong TASLP'10; Faller&Merimaa'04 |
| … `ipd` `U_ij` + `λ_IPD` | off-diag unit phasor, gated | Duong TASLP'10; Brughera JASA'13 |
