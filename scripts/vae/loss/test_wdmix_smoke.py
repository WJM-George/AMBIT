#!/usr/bin/env python3
"""Smoke test for the W-downmix + grouped-KL FOA VAE arm (CPU, tiny tensors).

Verifies, without launching a trainer:
  1. the config builds a model with the GroupedVAEBottleneck and encode() returns
     kl_w / kl_spatial in info;
  2. the shared-decoder masked W-downmix produces a [B,1,T] omni recon;
  3. the training wrapper wires every loss declared by the production config;
  4. the complete generator objective computes and backprops.
"""
import json
import sys
from pathlib import Path

import torch

_SAT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SAT))

from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.bottleneck import GroupedVAEBottleneck  # noqa: E402
from stable_audio_tools.training.factory import create_training_wrapper_from_config  # noqa: E402

CFG = (_SAT / "stable_audio_tools/configs/model_configs/autoencoders/"
       "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json")


def main():
    cfg = json.loads(CFG.read_text())
    torch.manual_seed(0)

    # 1) model + bottleneck
    model = create_model_from_config(cfg)
    assert isinstance(model.bottleneck, GroupedVAEBottleneck), type(model.bottleneck)
    assert model.bottleneck.n_w == 40
    model.eval()

    T = 1024 * 20  # 20 latent frames
    x = torch.randn(1, 4, T)
    with torch.no_grad():
        latents, info = model.encode(x, return_info=True)
    print(f"[1] latents {tuple(latents.shape)}  info keys={sorted(info)}")
    assert "kl_w" in info and "kl_spatial" in info and "kl" in info
    assert latents.shape[1] == 64

    # 2) masked W-downmix decode through the shared decoder
    n_w = model.bottleneck.n_w
    with torch.no_grad():
        dec_full = model.decode(latents)
        z_masked = latents.clone(); z_masked[:, n_w:, :] = 0.0
        dec_w = model.decode(z_masked)
    print(f"[2] dec_full {tuple(dec_full.shape)}  dec_w {tuple(dec_w.shape)}  W={tuple(dec_w[:,0:1].shape)}")
    assert dec_full.shape[1] == 4 and dec_w.shape[1] == 4

    # 3) training wrapper: build + one generator step (global_step even => gen)
    wrapper = create_training_wrapper_from_config(cfg, model)
    names = [m.name for m in wrapper.losses_gen.losses]
    print(f"[3] gen loss modules: {names}")
    expected = {
        "loss_adv", "feature_matching_loss", "mrstft_loss",
        "high_frequency_overshoot_loss", "foa_scm_loss", "phase_ifgd_loss",
        "sisdr_loss", "w_downmix_mrstft", "w_downmix_sisdr",
        "kl_w_loss", "kl_spatial_loss",
    }
    assert set(names) == expected, (set(names) - expected, expected - set(names))
    assert wrapper.w_downmix_n_w == 40
    assert wrapper._w_downmix_loss_index is not None

    # Resume compatibility: only loss-module buffers are reconciled. Model
    # weights and unrelated state must remain bit-identical.
    current = wrapper.state_dict()
    fake_resume = {
        "state_dict": {
            key: value.clone()
            for key, value in current.items()
            if not key.startswith(("losses_gen.", "losses_disc."))
        }
    }
    fake_resume["state_dict"]["losses_gen.losses.999.weight"] = torch.tensor(9.0)
    wrapper.on_load_checkpoint(fake_resume)
    assert "losses_gen.losses.999.weight" not in fake_resume["state_dict"]
    assert all(key in fake_resume["state_dict"] for key in current)

    # 4) exercise the exact generator loss path training_step builds, incl. the
    #    masked W-downmix aux decode, MRSTFT-on-1ch, grouped KL, and backprop.
    wrapper.train()
    wd_mod = wrapper.losses_gen.losses[wrapper._w_downmix_loss_index]
    wrapper._set_w_downmix_weight_for_step(1000000)   # ramp start -> 0
    w_start = float(wd_mod.weight)
    wrapper._set_w_downmix_weight_for_step(1010000)   # ramp midpoint -> ~0.15
    w_mid = float(wd_mod.weight)
    wrapper._set_w_downmix_weight_for_step(1025000)   # past ramp end -> full 0.3
    w_end = float(wd_mod.weight)
    print(f"[4] w_downmix ramp: start={w_start:.4f} mid={w_mid:.4f} end={w_end:.4f} (expect 0 / ~0.15 / 0.3)")
    assert w_start < 1e-4 and abs(w_mid - 0.15) < 0.02 and abs(w_end - 0.3) < 1e-4

    reals = torch.randn(2, 4, T, requires_grad=False)
    latents, enc_info = model.encode(reals, return_info=True)
    decoded = model.decode(latents)
    z_masked = latents.clone(); z_masked[:, n_w:, :] = 0.0
    decoded_w = model.decode(z_masked)

    loss_info = {"reals": reals, "decoded": decoded}
    loss_info.update(enc_info)
    loss_info["decoded_w"] = decoded_w[:, 0:1, :T]
    loss_info["reals_w"] = reals[:, 0:1, :T]

    total, losses = wrapper.losses_gen(loss_info)
    fired = {k: float(v) for k, v in losses.items()}
    print(f"[4] fired losses: {fired}")
    for k in expected - {"loss_adv", "feature_matching_loss"}:
        assert k in fired, f"missing {k}"
        assert fired[k] == fired[k], f"{k} is NaN"
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    print(f"[4] backprop OK; params with finite grad = {n_grad}")
    assert n_grad > 0

    print("\nSMOKE OK: grouped_vae + w_downmix wired; masked W-downmix decode + "
          "MRSTFT-on-W + grouped KL compute and backprop.")


if __name__ == "__main__":
    main()
