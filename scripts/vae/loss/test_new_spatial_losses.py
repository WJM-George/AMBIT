"""Sanity checks for FrequencyGatedIFGDPhaseLoss and FOASpatialCovarianceLoss.

Run:  uv run python scripts/vae/loss/test_new_spatial_losses.py
"""

import math

import torch

from stable_audio_tools.training.losses.semantic import (
    FOASpatialCovarianceLoss,
    FrequencyGatedIFGDPhaseLoss,
    _logistic_frequency_gate,
)

SR = 44100
T = 176400 // 4  # 1 s, keep the test fast


def make_foa_batch(batch=2, seed=0):
    """Synthetic FOA batch: a few point sources encoded to WYZX + weak noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T) / SR
    out = torch.zeros(batch, 4, T)
    for b in range(batch):
        for _ in range(3):
            az = torch.rand(1, generator=g).item() * 2 * math.pi
            el = (torch.rand(1, generator=g).item() - 0.5) * math.pi
            f0 = 100 + torch.rand(1, generator=g).item() * 4000
            src = torch.sin(2 * math.pi * f0 * t) * torch.rand(1, generator=g)
            # ACN/SN3D FOA encoding, channel order W Y Z X
            out[b, 0] += src                                  # W
            out[b, 1] += src * math.sin(az) * math.cos(el)    # Y
            out[b, 2] += src * math.sin(el)                   # Z
            out[b, 3] += src * math.cos(az) * math.cos(el)    # X
    out += 0.01 * torch.randn(batch, 4, T, generator=g)
    return 0.1 * out


def rotate_foa_z(x, angle):
    """Rotate the sound field around Z: mixes Y and X, leaves W and Z."""
    y, xx = x[:, 1], x[:, 3]
    out = x.clone()
    out[:, 1] = math.cos(angle) * y + math.sin(angle) * xx
    out[:, 3] = -math.sin(angle) * y + math.cos(angle) * xx
    return out


def main():
    torch.manual_seed(0)

    # -- gate curve basics
    f = torch.tensor([0.0, 1500.0, 22050.0])
    gate = _logistic_frequency_gate(f, fc=1500.0, beta=2.0, floor=0.1)
    assert abs(gate[0].item() - 1.0) < 1e-6
    assert abs(gate[1].item() - (0.1 + 0.9 / 2)) < 1e-6
    assert gate[2].item() < 0.11
    print(f"gate(0, fc, nyq) = {[round(v, 4) for v in gate.tolist()]}")

    ref = make_foa_batch()

    phase_loss = FrequencyGatedIFGDPhaseLoss(sample_rate=SR)
    scm_loss = FOASpatialCovarianceLoss(sample_rate=SR)

    # -- identity ~= 0, and gradients flow / loss finite on perturbed input.
    # The phase loss has a tiny identity floor from denominator-clamped
    # noise-floor bins (same property as SAME's IFGD); what matters is that it
    # is negligible next to a real error.
    pred = (ref + 0.05 * torch.randn_like(ref)).requires_grad_(True)
    for name, loss in [("phase", phase_loss), ("scm", scm_loss)]:
        v_id = loss(ref.clone(), ref)
        v = loss(pred, ref)
        v.backward()
        assert torch.isfinite(v), f"{name} loss not finite"
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        print(
            f"{name}(x, x) = {v_id.item():.3e}   {name}(x+n, x) = {v.item():.4f}"
            f"   |grad| = {pred.grad.abs().mean():.3e}"
        )
        assert v_id.item() < 0.01 * v.item(), f"{name} identity floor not negligible"
        pred.grad = None

    # -- SCM loss must respond strongly to a pure spatial rotation
    # (same per-channel-ish energy statistics, wrong direction)
    rotated = rotate_foa_z(ref, math.pi / 2)
    v_rot = scm_loss(rotated, ref)
    v_noise = scm_loss(ref + 0.05 * torch.randn_like(ref), ref)
    print(f"scm(rot90, x) = {v_rot.item():.4f}   scm(x+n, x) = {v_noise.item():.4f}")
    assert v_rot.item() > 2 * v_noise.item(), "SCM loss not direction sensitive"

    # -- sub-term telemetry populated
    _ = scm_loss(rotated, ref)
    print(
        "scm terms: level={:.4f} coh={:.4f} ipd={:.4f} mean_ipd_cos={:.4f}".format(
            scm_loss.last_level_loss.item(),
            scm_loss.last_coherence_loss.item(),
            scm_loss.last_ipd_loss.item(),
            scm_loss.last_mean_ipd_cos.item(),
        )
    )
    _ = phase_loss(ref + 0.05 * torch.randn_like(ref), ref)
    print(
        "phase terms: if={:.4f} gd={:.4f} cd={:.4f}".format(
            phase_loss.last_if_loss.item(),
            phase_loss.last_gd_loss.item(),
            phase_loss.last_cd_loss.item(),
        )
    )

    # -- division of labor: a 3-sample delay on one channel is (by design)
    # nearly invisible to per-channel IF/GD (phase *derivatives* are invariant
    # to constant delays), but it corrupts inter-channel phase, which the SCM
    # IPD term must catch.
    delayed = ref.clone()
    delayed[:, 1] = torch.roll(delayed[:, 1], shifts=3, dims=-1)
    v_delay_phase = phase_loss(delayed, ref)
    v_delay_scm = scm_loss(delayed, ref)
    print(
        f"delay Y by 3 samples: phase = {v_delay_phase.item():.4f} (expected ~0), "
        f"scm = {v_delay_scm.item():.4f}, scm_ipd = {scm_loss.last_ipd_loss.item():.4f}"
    )
    assert v_delay_phase.item() < 0.05, "IF/GD should be nearly delay-invariant"
    assert scm_loss.last_ipd_loss.item() > 0.02, "SCM IPD term must catch inter-channel delay"

    # -- phase loss must respond to phase-only distortion (random smooth
    # phase rotation of the STFT, magnitudes preserved)
    n_fft, hop = 1024, 256
    window = torch.hann_window(n_fft)
    flat = ref.reshape(-1, T)
    spec = torch.stft(flat, n_fft, hop, window=window, return_complex=True)
    jitter = torch.randn(spec.shape[0], spec.shape[1], 1) * 0.8
    spec_rot = spec * torch.exp(1j * jitter)  # per-bin constant-in-time phase offset
    dirty = torch.istft(spec_rot, n_fft, hop, window=window, length=T).reshape(ref.shape)
    v_phase_dirty = phase_loss(dirty, ref)
    print(
        f"phase(random per-bin phase rotation) = {v_phase_dirty.item():.4f} "
        f"(if={phase_loss.last_if_loss.item():.4f} gd={phase_loss.last_gd_loss.item():.4f} "
        f"cd={phase_loss.last_cd_loss.item():.4f})"
    )
    # istft consistency-projection heals much of the rotation; require the
    # response to stand far above the identity floor rather than a fixed value.
    assert v_phase_dirty.item() > 50 * 2.5e-4

    # -- fp16-ish input dtype safety (loss casts to fp32 internally)
    v = scm_loss(ref.to(torch.bfloat16).float(), ref)
    assert torch.isfinite(v)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
