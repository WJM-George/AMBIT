"""
4-channel Oobleck VAE: construction + warm-start from a pretrained 2-channel checkpoint.

This module adds NO new model architecture. The 4-channel autoencoder is a stock
``AudioAutoencoder`` built from a config whose Oobleck encoder/decoder are set to
``in_channels=4`` / ``out_channels=4`` (see configs/model_configs/autoencoders/stable_audio_4ch_vae.json).

The only thing the stock framework cannot do is initialize that 4-channel model from
a pretrained *stereo* (2-channel) checkpoint, because the encoder's first conv and the
decoder's last conv have a different channel dimension. The surgery here:

  * encoder first conv  (in_channels: 2 -> 4): copy pretrained weights into the first 2
    input channels, ZERO-init the 2 extra input channels. With zero-init the extra
    inputs are initially ignored, so on a stereo/binaural signal padded to [L,R,0,0] the
    model starts out numerically identical to the pretrained one.
  * decoder last conv   (out_channels: 2 -> 4): copy pretrained weights into the first 2
    output channels, and REPLICATE-init the 2 extra output channels (ch2<-ch0, ch3<-ch1)
    so the new channels start as plausible audio rather than silence.

``WNConv1d`` uses the legacy ``torch.nn.utils.weight_norm`` parametrization, so the
pretrained state dict stores ``weight_g`` / ``weight_v`` (not ``weight``). We reconstruct
the effective weight, do the surgery on the materialized weight, then re-apply weight_norm.
"""

import json
from typing import Dict, Any, Tuple

import torch
from torch.nn.utils import weight_norm, remove_weight_norm

from .factory import create_model_from_config


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def create_4ch_autoencoder_from_config_path(config_path: str):
    with open(config_path) as f:
        config = json.load(f)
    return create_4ch_autoencoder_from_config(config)


def create_4ch_autoencoder_from_config(config: Dict[str, Any]):
    model = create_model_from_config(config)
    # Sanity: this helper is only meaningful for a >2 channel autoencoder.
    in_ch = getattr(model, "in_channels", None)
    out_ch = getattr(model, "out_channels", None)
    assert (in_ch or 0) >= 2 and (out_ch or 0) >= 2, \
        f"Expected a multi-channel autoencoder, got in={in_ch} out={out_ch}"
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def load_raw_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from .safetensors or .ckpt/.pt, unwrapping a 'state_dict' key."""
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def extract_autoencoder_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Return a state dict whose keys start with 'encoder.'/'decoder.'/'bottleneck.', stripping
    common wrapper prefixes. Handles:
      * already-unwrapped autoencoder checkpoints (keys already 'encoder.*')
      * full Stable Audio models where the VAE lives under 'pretransform.model.*'
      * Lightning training wrappers ('autoencoder.*' / 'model.*' / 'ema_*')
    """
    candidate_prefixes = [
        "",
        "autoencoder.",
        "model.",
        "pretransform.model.",
        "pretransform.autoencoder.",
        "pretransform.",
        "ema_model.",
        "autoencoder_ema.",
    ]
    for prefix in candidate_prefixes:
        if any(k.startswith(prefix + "encoder.") for k in state_dict):
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith(prefix + "encoder.") or k.startswith(prefix + "decoder.") \
                        or k.startswith(prefix + "bottleneck."):
                    stripped[k[len(prefix):]] = v
            return stripped
    # Could not find encoder.* keys under any known prefix.
    raise KeyError(
        "Could not locate 'encoder.*' keys in the checkpoint under any known prefix. "
        f"Top-level keys (sample): {list(state_dict)[:8]}"
    )


# ---------------------------------------------------------------------------
# Weight surgery
# ---------------------------------------------------------------------------

def _reconstruct_wn_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Reconstruct effective conv weight from legacy weight_norm (dim=0) parameters."""
    # For dim=0, the norm is computed over every dimension except 0.
    dims = tuple(range(1, weight_v.dim()))
    norm = weight_v.norm(p=2, dim=dims, keepdim=True)
    return weight_g * weight_v / (norm + 1e-12)


def _conv_module_and_prefix_encoder_first(model) -> Tuple[torch.nn.Module, str]:
    # OobleckEncoder.layers is an nn.Sequential whose layer 0 is the first WNConv1d.
    return model.encoder.layers[0], "encoder.layers.0"


def _conv_module_and_prefix_decoder_last(model) -> Tuple[torch.nn.Module, str]:
    # OobleckDecoder.layers ends with [..., activation, WNConv1d(out), Tanh()/Identity()].
    layers = model.decoder.layers
    idx = len(layers) - 2  # last conv is second-to-last (last is Tanh/Identity)
    return layers[idx], f"decoder.layers.{idx}"


def warm_start_2ch_to_4ch(model, ae_state_dict_2ch: Dict[str, torch.Tensor],
                          replicate_output: bool = True, verbose: bool = True):
    """
    Load a pretrained 2-channel autoencoder state dict into a freshly built 4-channel
    `model`, performing channel surgery on the I/O convolutions.

    Returns (missing_keys, unexpected_keys) from the partial load (excluding the two
    surgically-handled convs).
    """
    sd = dict(ae_state_dict_2ch)  # shallow copy we can pop from

    enc_conv, enc_prefix = _conv_module_and_prefix_encoder_first(model)
    dec_conv, dec_prefix = _conv_module_and_prefix_decoder_last(model)

    # --- reconstruct pretrained effective weights for the two surgical convs ---
    enc_w_old = _reconstruct_wn_weight(sd[f"{enc_prefix}.weight_g"], sd[f"{enc_prefix}.weight_v"])
    enc_b_old = sd.get(f"{enc_prefix}.bias", None)
    dec_w_old = _reconstruct_wn_weight(sd[f"{dec_prefix}.weight_g"], sd[f"{dec_prefix}.weight_v"])
    dec_b_old = sd.get(f"{dec_prefix}.bias", None)

    # --- pop the two surgically-handled convs ---
    for k in list(sd.keys()):
        if k.startswith(enc_prefix + ".") or k.startswith(dec_prefix + "."):
            sd.pop(k)

    # Alias-free activations wrap SnakeBeta under ``act``. Preserve the pretrained
    # Snake parameters when comparing an antialiased decoder to the stock decoder.
    model_sd = model.state_dict()
    remapped_activation = []
    for k in list(sd.keys()):
        if k in model_sd or not (k.endswith(".alpha") or k.endswith(".beta")):
            continue
        prefix, parameter = k.rsplit(".", 1)
        candidate = f"{prefix}.act.{parameter}"
        if candidate in model_sd and tuple(model_sd[candidate].shape) == tuple(sd[k].shape):
            sd[candidate] = sd.pop(k)
            remapped_activation.append((k, candidate))

    if verbose and remapped_activation:
        print(
            f"[warm_start_2ch_to_4ch] remapped {len(remapped_activation)} "
            "Snake parameters into alias-free activation wrappers"
        )

    # --- drop any remaining SHAPE-mismatched tensors (strict=False only ignores
    # missing/unexpected keys, NOT shape mismatches). Architecture variants that
    # change strides (kernel=2*stride) or latent width have differently-shaped
    # conv/projection weights; those keep their fresh init and the rest copies. ---
    skipped_shape = []
    for k in list(sd.keys()):
        if k in model_sd and tuple(model_sd[k].shape) != tuple(sd[k].shape):
            sd.pop(k)
            skipped_shape.append(k)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose and skipped_shape:
        print(f"[warm_start_2ch_to_4ch] skipped {len(skipped_shape)} shape-mismatched "
              f"tensors (kept fresh init): {skipped_shape[:4]}{' ...' if len(skipped_shape) > 4 else ''}")

    # --- encoder first conv: in_channels 2 -> 4 ---
    remove_weight_norm(enc_conv)
    with torch.no_grad():
        old_in = enc_w_old.shape[1]
        assert enc_conv.weight.shape[0] == enc_w_old.shape[0], "encoder out-dim mismatch"
        assert enc_conv.weight.shape[1] >= old_in, "target encoder must have >= input channels"
        enc_conv.weight.zero_()
        enc_conv.weight[:, :old_in, :].copy_(enc_w_old.to(enc_conv.weight))
        if enc_b_old is not None and enc_conv.bias is not None:
            enc_conv.bias.copy_(enc_b_old.to(enc_conv.bias))
    weight_norm(enc_conv)

    # --- decoder last conv: out_channels 2 -> 4 ---
    remove_weight_norm(dec_conv)
    with torch.no_grad():
        old_out = dec_w_old.shape[0]
        assert dec_conv.weight.shape[1] == dec_w_old.shape[1], "decoder in-dim mismatch"
        assert dec_conv.weight.shape[0] >= old_out, "target decoder must have >= output channels"
        dec_conv.weight.zero_()
        dec_conv.weight[:old_out, :, :].copy_(dec_w_old.to(dec_conv.weight))
        if replicate_output:
            for c in range(old_out, dec_conv.weight.shape[0]):
                dec_conv.weight[c].copy_(dec_w_old[c % old_out].to(dec_conv.weight))
        if dec_b_old is not None and dec_conv.bias is not None:
            dec_conv.bias.zero_()
            dec_conv.bias[:old_out].copy_(dec_b_old.to(dec_conv.bias))
            if replicate_output:
                for c in range(old_out, dec_conv.bias.shape[0]):
                    dec_conv.bias[c].copy_(dec_b_old[c % old_out].to(dec_conv.bias))
    weight_norm(dec_conv)

    if verbose:
        real_missing = [k for k in missing
                        if not (k.startswith(enc_prefix) or k.startswith(dec_prefix))]
        print(f"[warm_start_2ch_to_4ch] partial load done. "
              f"missing(non-surgical)={len(real_missing)} unexpected={len(unexpected)}")
        print(f"[warm_start_2ch_to_4ch] encoder first conv  '{enc_prefix}': "
              f"in 2->{enc_conv.in_channels} (extra inputs zero-init)")
        print(f"[warm_start_2ch_to_4ch] decoder last conv   '{dec_prefix}': "
              f"out 2->{dec_conv.out_channels} "
              f"(extra outputs {'replicated' if replicate_output else 'zero'}-init)")
        if real_missing:
            print(f"[warm_start_2ch_to_4ch] sample missing keys: {real_missing[:5]}")
        if unexpected:
            print(f"[warm_start_2ch_to_4ch] sample unexpected keys: {list(unexpected)[:5]}")

    return missing, unexpected


def build_4ch_vae_with_warm_start(config_path: str, pretrained_ckpt_path: str = None,
                                  replicate_output: bool = True, verbose: bool = True):
    """Convenience: build the 4ch VAE and (optionally) warm-start it from a 2ch checkpoint."""
    model = create_4ch_autoencoder_from_config_path(config_path)
    if pretrained_ckpt_path is not None:
        raw = load_raw_state_dict(pretrained_ckpt_path)
        ae_sd = extract_autoencoder_state_dict(raw)
        warm_start_2ch_to_4ch(model, ae_sd, replicate_output=replicate_output, verbose=verbose)
    return model
