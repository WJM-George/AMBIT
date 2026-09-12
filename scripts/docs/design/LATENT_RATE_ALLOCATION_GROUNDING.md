# FOA VAE Latent Rate-Allocation (W-downmix + spatial residual) — Theory Grounding

> Companion to [`LOSS_THEORY_GROUNDING.md`](LOSS_THEORY_GROUNDING.md) and
> [`LOSS_DESIGN_SPATIAL_PHASE.md`](LOSS_DESIGN_SPATIAL_PHASE.md).
> Those files ground the **loss** terms. This file grounds the **architecture /
> latent-organization** change introduced in the `04_rate_alloc` stage
> (`vae_abl_wdmix_scm`): giving the omni **W** a large "transport" sub-block of
> the VAE latent and forcing the directional channels (Y,Z,X) to be a **compact
> spatial residual**. The point of this doc is that this is **not an engineering
> hack** — it is the neural-VAE realization of the *downmix + parametric
> spatial side-information* paradigm that has been the standard in spatial-audio
> coding for 20 years (DirAC, MPEG Surround/SAOC, MPEG-H 3D Audio, 3GPP-IVAS
> SPAR), combined with the *rate-distortion* view of the VAE objective.
>
> Structure per mechanism: **Problem observed → Theory (venue/journal) → How our
> code follows it → Novelty check**. Cross-checked against the running config
> [`stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json`](../../../stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json),
> the bottleneck [`models/bottleneck.py`](../../../stable_audio_tools/models/bottleneck.py)
> (`GroupedVAEBottleneck`), and the training wrapper
> [`training/autoencoders.py`](../../../stable_audio_tools/training/autoencoders.py)
> (`w_downmix` aux decode + grouped KL).

## Empirical trigger (why we changed the architecture, not the loss)

The `codec_compare_90_diag` experiment (40 speech / 25 music / 25 sound, mean±std)
ran a **capacity diagnostic**: feed the VAE `[W,W,W,W]` (all four input channels
set to the omni W) so the latent only has to carry *one* signal, and still score
the reconstructed W against the true W. Result: the **same** VAE reached
W SI-SDR **≥ DAC on all three categories** (speech 10.8 / music 7.5 / sound 2.2 dB
vs DAC 9.6 / 6.4 / 2.2) and PESQ 2.72 → 3.20 — with **no loss change**. Feeding
`[W,0,0,0]` (zeroing directional inputs *without* changing the latent structure)
did **not** help. Conclusion: **the W-vs-DAC gap is a latent rate-allocation
problem** — one 64-ch @ ds1024 latent is shared across 4 FOA channels, so W gets
≈¼ of the budget, whereas DAC spends its whole budget on one channel. The fix is
to *reallocate* latent bits toward W, which is exactly the downmix+residual idea
below.

The three mechanisms and their theory anchors:

| Mechanism (code) | What it does | Primary theory anchor (venue) |
|---|---|---|
| **W-transport sub-block** (`w_downmix` masked-latent aux decode) | first `n_w` latent channels must reconstruct W alone | DirAC omni transport — Pulkki, *JAES* 2007; SPAR downmix — 3GPP-IVAS, *ICASSP/EUSIPCO* 2025 |
| **Compact spatial residual** (remaining channels + `foa_scm`) | Y,Z,X carried as a low-rate covariance-matched residual | MPEG Surround/SAOC & MPEG-H side-info — Herre et al., *JAES* 2008 / *IEEE JSTSP* 2015; SPAR residuals |
| **Grouped-β rate allocation** (`GroupedVAEBottleneck`, `kl_w` / `kl_spatial`) | lower KL on W-group (more bits), higher KL on spatial-group (fewer bits) | Rate-distortion VAE — Alemi et al., *ICML* 2018; β-VAE — Higgins et al., *ICLR* 2017 |

---

## 1. W as a self-decodable "transport" sub-block

### 1.1 Problem observed
The diagnostic above shows W can reach DAC-level fidelity *if it is not forced to
share latent capacity with the directional channels*. We need, inside a single
model, a **sub-block of the latent that is a sufficient statistic for W** — i.e.
a "downmix/transport" channel living in latent space — so the decoder reads a
clean W code instead of a W smeared across all 64 dims.

### 1.2 Theory (venue/journal)
- **Directional Audio Coding (DirAC)** — V. Pulkki, "Spatial Sound Reproduction
  with Directional Audio Coding," *J. Audio Eng. Soc.*, vol. 55, no. 6,
  pp. 503–516, 2007. DirAC transmits **one omni transport channel (the B-format
  W, scaled 1/√2)** plus per-time-frequency **direction + diffuseness** metadata;
  the multichannel field is *synthesized* at the decoder from that single
  transport signal. The transport channel *is* the omni; spatial cues are a small
  parametric side stream.
- **3GPP-IVAS SPAR (Spatial Reconstruction)** — "Ambisonics Coding in IVAS:
  A Hybrid SPAR and DirAC System," *ICASSP* 2025 (DOI
  10.1109/ICASSP49660.2025.10888651); "Coding Higher Order Ambisonics in 3GPP
  IVAS," *EUSIPCO* 2025. SPAR is the closest classical prior art: it "creates a
  **4-channel downmix consisting of W and residuals Y′, Z′, X′**", where the
  residuals are made **low-energy / highly compressible** by exploiting the
  bandwise inter-channel **covariance**, and reconstructs the field from W +
  residuals + prediction metadata. This is precisely "keep W, make Y/Z/X a
  compact residual."

### 1.3 How our code follows it
`training/autoencoders.py`, generator step (no new parameters — the **shared
decoder** is reused):
```
n_w = self.w_downmix_n_w                 # 40 of 64 latent channels
latents_w = latents.clone()
latents_w[:, n_w:, :] = 0.0              # mask the spatial-residual group
decoded_w = self.autoencoder.decode(latents_w)   # decode from W-group ALONE
loss_info["decoded_w"] = decoded_w[:, 0:1, :]    # its W output
loss_info["reals_w"]   = reals[:, 0:1, :]        # scored against the true W
```
The `w_downmix` loss = the same K-weighted multi-resolution STFT (`self.sdstft`,
with the `w_cep` cepstral term) + a small SI-SDR, applied to `decoded_w` vs the
true W (config `w_downmix.weights = {stft: 0.3, sisdr: 0.003}`, cosine-ramped
1.00M→1.02M). Because the first `n_w` channels must reconstruct W **on their own**
(spatial channels zeroed), gradient pressure turns them into a DirAC/SPAR-style
**omni transport sub-block**; the remaining channels are freed for the residual.

### 1.4 Novelty check
DirAC/SPAR build the transport by **hand-designed linear analysis** and transmit
metadata explicitly. We instead **learn** the transport *inside a generative VAE
latent* via a masked-latent self-decoding constraint, with **zero added
parameters** and an unchanged latent shape (so the Stage-2 DiT pretransform
interface is untouched). To our knowledge this "learned in-latent downmix
sub-block" for FOA has no direct precedent.

---

## 2. Directional channels as a compact, covariance-matched residual

### 2.1 Problem observed
Once W owns a transport sub-block, Y/Z/X must be represented **cheaply** — but the
spatial *image* (direction, width, diffuseness) must survive. We need the residual
group to be low-rate yet perceptually complete for spatial reproduction.

### 2.2 Theory (venue/journal)
- **MPEG Surround** — J. Herre et al., "MPEG Surround — The ISO/MPEG Standard for
  Efficient and Compatible Multichannel Audio Coding," *J. Audio Eng. Soc.*,
  vol. 56, no. 11, pp. 932–955, 2008. Encodes a **downmix + parametric side
  info** = inter-channel **level differences** and **coherence/correlation** per
  frequency band; the multichannel signal is regenerated by TF matrixing +
  decorrelation.
- **MPEG-H 3D Audio / SAOC-3D** — J. Herre, J. Hilpert, A. Kuntz, J. Plogsties,
  "MPEG-H 3D Audio — The New Standard for Coding of Immersive Spatial Audio,"
  *IEEE J. Sel. Topics Signal Process.*, vol. 9, no. 5, pp. 770–779, 2015
  (DOI 10.1109/JSTSP.2015.2411578). Generalizes the above: map inputs to a
  **downmix + compact side information** (object level differences + inter-object
  correlations per TF tile); decode by a single-step TF matrixing. The perceptually
  sufficient spatial descriptor is a **second-order (covariance) quantity per band**.
- This is exactly the descriptor our **`foa_scm`** loss already matches (full
  complex spatial covariance, polar-decomposed into level / coherence / IPD),
  grounded in **Vilkamo et al., *JAES* 2013** and **Duong et al., *IEEE TASLP*
  2010** (see `LOSS_THEORY_GROUNDING.md` §3). SPAR's "residuals from bandwise
  covariance" is the same principle.

### 2.3 How our code follows it
No change to the spatial objective is needed — it was already the right one. The
spatial-residual group (`latents[:, n_w:]`) is shaped by (a) the main 4-ch
reconstruction, (b) `foa_scm` (0.015) matching the full covariance = MPEG-H's
level+coherence side info generalized to complex IPD, and (c) `phase_ifgd` (0.08).
The `04` change simply gives this residual its **own** latent channels (freed by
§1) and compresses them (§3), so it behaves like MPEG-H side information.

### 2.4 Novelty check
Classical codecs transmit these covariance cues as **quantized metadata** next to
a waveform downmix. We keep them as a **learned continuous residual latent**
regularized to the *same* covariance target, so a single VAE latent is
simultaneously (i) a DiT-friendly continuous code and (ii) a downmix+side-info
decomposition.

---

## 3. The transport-vs-residual bit split = grouped-β rate allocation

### 3.1 Problem observed
"Give W more capacity, compress the residual" must be made precise and be
controllable, without changing latent dimensionality (the DiT interface) and
without collapsing latent smoothness.

### 3.2 Theory (venue/journal)
- **Rate-distortion view of the VAE** — A. A. Alemi, B. Poole, I. Fischer,
  J. V. Dillon, R. A. Saurous, K. Murphy, "Fixing a Broken ELBO," *ICML* 2018,
  PMLR 80:159–168. The ELBO decomposes as **distortion + β·rate**, where the KL
  term *is* the **rate** (bits) the latent carries, and β sweeps the
  rate-distortion curve. Lower β ⇒ higher rate (more bits, higher fidelity);
  higher β ⇒ lower rate (more compression).
- **β-VAE** — I. Higgins et al., "β-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework," *ICLR* 2017. Establishes β on the KL as the
  knob that trades reconstruction for a more compressed/structured code.

### 3.3 How our code follows it
`GroupedVAEBottleneck` (`models/bottleneck.py`) is **parameter-free** (identical
sampling to the plain VAE, so warm-start is loss-free) and returns per-group KL:
```
kl_w       = KL over channels [0 : n_w)         # the W transport group
kl_spatial = KL over channels [n_w : 64)        # the spatial residual group
```
`create_loss_modules_from_bottleneck` applies **separate β** to each
(config `bottleneck.weights`):
```
kl_w       = 5e-5     # lower β  → more rate/bits for W  (higher omni fidelity)
kl_spatial = 2e-4     # higher β → compress the directional residual
```
The per-channel average β stays ≈ the original `1e-4` (`(40·5e-5 + 24·2e-4)/64 ≈
1.06e-4`), so the **total rate budget is preserved** — we only *reallocate* it,
which keeps the DiT-facing latent statistics close to the parent. This is exactly
a per-group point on the Alemi rate-distortion curve.

### 3.4 Novelty check
β-VAE/RD-VAE use a **single global** β. We use a **structured, per-latent-group**
β chosen from the spatial-audio-coding prior (W transport deserves more rate than
the directional residual). Grouped-β rate allocation driven by an acoustic
downmix argument is, to our knowledge, new for neural spatial-audio latents.

---

## 4. Positioning vs. the closest neural prior art

- **Hirvonen & Namazi, "Compression of Higher Order Ambisonics with a Multichannel
  RVQGAN," arXiv:2411.12008 (ICASSP 2025)** — neural HOA codec with a broadband
  covariance loss, but **discrete RVQ tokens, no downmix/transport structure, no
  rate-allocated continuous latent**. We are a continuous VAE (DiT pretransform)
  with an explicit learned transport+residual split.
- **"Residual Learning for Neural Ambisonics Encoders," arXiv:2601.18322** — adds a
  learned residual on top of a *linear* encoder for **microphone→ambisonics**
  capture, not codec-latent rate allocation. Shares the "linear base + learned
  residual" spirit; different problem.
- **SPAR/DirAC/MPEG-H** — signal-domain, hand-designed transforms + transmitted
  metadata. We realize the same *paradigm* inside a learned generative latent.

**One-sentence positioning.** `04_rate_alloc` is the *learned-latent* counterpart
of DirAC/SPAR/MPEG-H downmix+side-info coding: a single FOA VAE whose latent is
organized (by a masked-latent self-downmix constraint) into a **W transport
sub-block** and a **covariance-matched spatial residual**, with the bit split set
by **grouped-β rate allocation** (RD-VAE) — closing the W gap to DAC while keeping
the spatial moat and the DiT-friendly continuous latent.

---

## References

1. V. Pulkki. "Spatial Sound Reproduction with Directional Audio Coding."
   *J. Audio Eng. Soc.*, 55(6):503–516, 2007.
2. J. Herre, K. Kjörling, J. Breebaart, et al. "MPEG Surround — The ISO/MPEG
   Standard for Efficient and Compatible Multichannel Audio Coding."
   *J. Audio Eng. Soc.*, 56(11):932–955, 2008.
3. J. Herre, J. Hilpert, A. Kuntz, J. Plogsties. "MPEG-H 3D Audio — The New
   Standard for Coding of Immersive Spatial Audio." *IEEE J. Sel. Topics Signal
   Process.*, 9(5):770–779, 2015. DOI 10.1109/JSTSP.2015.2411578.
4. 3GPP-IVAS Ambisonics coding (SPAR + DirAC): "Ambisonics Coding in IVAS: A
   Hybrid SPAR and DirAC System," *ICASSP* 2025, DOI
   10.1109/ICASSP49660.2025.10888651; "Coding Higher Order Ambisonics in 3GPP
   IVAS — Scaling Parametric Audio Coding to Higher Bitrates," *EUSIPCO* 2025.
5. A. A. Alemi, B. Poole, I. Fischer, J. V. Dillon, R. A. Saurous, K. Murphy.
   "Fixing a Broken ELBO." *ICML* 2018, PMLR 80:159–168 (arXiv:1711.00464).
6. I. Higgins, L. Matthey, A. Pal, et al. "β-VAE: Learning Basic Visual Concepts
   with a Constrained Variational Framework." *ICLR* 2017.
7. J. Vilkamo, T. Bäckström, A. Kuntz. "Optimized Covariance Domain Framework for
   Time–Frequency Processing of Spatial Audio." *J. Audio Eng. Soc.*, 2013.
8. N. Q. K. Duong, E. Vincent, R. Gribonval. "Under-Determined Reverberant Audio
   Source Separation Using a Full-Rank Spatial Covariance Model." *IEEE TASLP*,
   18(7):1830–1840, 2010.
9. T. Hirvonen, A. Namazi. "Compression of Higher Order Ambisonics with a
   Multichannel RVQGAN." arXiv:2411.12008 (ICASSP 2025).

_Code cross-refs:_ `GroupedVAEBottleneck` and `vae_sample_grouped`
(`models/bottleneck.py`); `w_downmix` aux decode + `_set_w_downmix_weight_for_step`
+ grouped-KL wiring in `create_loss_modules_from_bottleneck`
(`training/autoencoders.py`); config
`stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json`.
