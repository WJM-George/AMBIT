from .autoencoders import AudioAutoencoder, OobleckEncoder, OobleckDecoder
from .autoencoders import VAEBottleneck, GroupedVAEBottleneck
from .diffusion import ConditionedDiffusionModelWrapper, DiTWrapper
from .conditioners import QwenTextConditioner, ScenePlan44LocalConditioner, MultiConditioner


def create_model_from_config(config):
    model = config["model"]
    if config["model_type"] == "autoencoder":
        if model["encoder"]["type"] != "oobleck" or model["decoder"]["type"] != "oobleck":
            raise ValueError("AMBIT requires the Oobleck FOA codec")
        spec = model["bottleneck"]
        classes = {"vae": VAEBottleneck, "grouped_vae": GroupedVAEBottleneck}
        bottleneck = classes[spec["type"]](**spec.get("config", {}))
        return AudioAutoencoder(
            OobleckEncoder(**model["encoder"]["config"]),
            OobleckDecoder(**model["decoder"]["config"]),
            io_channels=model["io_channels"],
            latent_dim=model["latent_dim"],
            downsampling_ratio=model["downsampling_ratio"],
            sample_rate=config["sample_rate"],
            bottleneck=bottleneck,
            soft_clip=model["decoder"].get("soft_clip", False),
        )
    if config["model_type"] != "diffusion_cond" or model["diffusion"]["type"] != "dit":
        raise ValueError("AMBIT requires a conditioned DiT renderer")
    spec = model["diffusion"]
    if spec.get("diffusion_objective") != "rectified_flow":
        raise ValueError("AMBIT uses rectified flow")
    classes = {"qwen_text": QwenTextConditioner, "sceneplan_44": ScenePlan44LocalConditioner}
    cond = model["conditioning"]
    conditioners = {}
    for item in cond["configs"]:
        kwargs = dict(item["config"])
        kwargs.setdefault("output_dim", cond["cond_dim"])
        conditioners[item["id"]] = classes[item["type"]](**kwargs)
    network = DiTWrapper(diffusion_objective="rectified_flow", **spec["config"])
    return ConditionedDiffusionModelWrapper(
        network,
        MultiConditioner(
            conditioners,
            default_keys=cond.get("default_keys"),
            pre_encoded_keys=cond.get("pre_encoded_keys"),
        ),
        io_channels=model["io_channels"],
        sample_rate=config["sample_rate"],
        min_input_length=network.model.patch_size,
        pretransform=None,
        diffusion_objective="rectified_flow",
        cross_attn_cond_ids=spec.get("cross_attention_cond_ids", []),
        global_cond_ids=spec.get("global_cond_ids", []),
        input_concat_ids=spec.get("input_concat_ids", []),
        mask_padding_attention=spec.get("mask_padding_attention", False),
    )
