#!/usr/bin/env python3
"""Edit native 44.1kHz FOA using a checksum-pinned, independently tested release."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

import numpy as np
import soundfile as sf
import torch

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_audio_io import check_audio, validate_release
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import CLAP44_PIPELINE_CONTRACT, load_clap44_validated_release
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_release import _immutable, gt, audio

def read_source(path):
    path = Path(path).resolve(strict=True)
    artifact = gt._artifact(path)
    wave, rate = sf.read(path,dtype="float32",always_2d=True)
    if (rate != 44100 or wave.shape[1] != 4 or not 0 < len(wave) <= 648*1024 or not np.isfinite(wave).all()):
        raise ValueError("source must be finite native 44.1kHz four-channel FOA within the 648-frame model limit")
    if gt.sha256_file(path) != artifact["sha256"]:
        raise RuntimeError("source FOA changed while reading")
    return torch.from_numpy(wave.T.copy()).unsqueeze(0),artifact


@contextmanager
def editing_device(device):
    if device == "cpu":
        yield torch.device("cpu")
        return
    if device != "cuda":
        raise ValueError("device must be cuda or cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    yield torch.device("cuda", 0)


def run(args):
    if not args.instruction.strip():
        raise ValueError("provide a nonempty raw edit instruction")
    if not 0 <= args.seed < 2**63:
        raise ValueError("inference seed must fit a nonnegative signed 64-bit integer")
    wave, source = read_source(args.source)
    directory = args.output_dir.resolve()
    release = Path(args.release).resolve(strict=True)
    if gt.sha256_file(release) != args.release_sha256:
        raise RuntimeError("inference needs the pinned native release SHA256")
    contract = {
        "schema":"editing_clap44_audio_request_v1","pipeline_contract":CLAP44_PIPELINE_CONTRACT,
        "release":{"path":str(release),"sha256":args.release_sha256},"source_foa":source,
        "raw_edit_instruction":args.instruction,"model_num_samples":wave.shape[-1],
        "sample_rate":44100,"channels":4,"seed":args.seed,"steps":20,"cfg_scale":1.,
        "max_plan_tokens":512,"batch_size":1,"device":args.device,
    }
    contract_path = directory/"INFERENCE_CONTRACT.json"
    result_path = directory/"RESULT.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("inference output belongs to another source/instruction/settings")
    elif directory.exists() and any(directory.iterdir()):
        raise RuntimeError("inference output has unidentified artifacts; preserve it")
    if result_path.exists():
        validate_release(release,expected_sha256=args.release_sha256)
        result = json.loads(result_path.read_text())
        if result["contract"] != gt._artifact(contract_path) or result["release"] != contract["release"]:
            raise RuntimeError("completed inference identity changed")
        check_audio(result["edited_foa"]["path"],result["edited_foa"]["sha256"],samples=wave.shape[-1])
        gt._verify_artifact(result["new_sceneplan"])
        return result
    with editing_device(args.device) as device:
        pipeline, report = load_clap44_validated_release(release,expected_sha256=args.release_sha256,device=device)
        if report.get("quality_gate_passed") is not True or report["release_sha256"] != args.release_sha256:
            raise RuntimeError("formal inference did not load a validated native release")
        _immutable(contract_path,contract)
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(args.seed)
        if device.type=="cuda": torch.cuda.manual_seed_all(args.seed)
        output = pipeline.edit_audio(wave,[args.instruction],model_num_samples=[wave.shape[-1]],sample_rate=44100,
            vae_seeds=[args.seed],noise_seed=args.seed,max_plan_tokens=512,steps=20,cfg_scale=1.)
        samples = wave.shape[-1]
        edited = output["edited_foa"]
        mask = output["sample_attention_mask"]
        if (edited.ndim!=3 or edited.shape[:2]!=(1,4) or edited.shape[-1]<samples or
                not torch.isfinite(edited).all() or mask.shape!=(1,edited.shape[-1]) or
                not mask[0,:samples].all() or mask[0,samples:].any() or edited[:,:,samples:].count_nonzero()):
            raise RuntimeError("native output lost exact FOA length/padding or finite samples")
        if len(output["new_sceneplans"])!=1 or len(output["new_sceneplan_token_ids"])!=1:
            raise RuntimeError("native inference omitted its complete generated plan")
        plan = _immutable(directory/"NEW_SCENEPLAN.json",{
            "sceneplan":output["new_sceneplans"][0],
            "token_ids":output["new_sceneplan_token_ids"][0].detach().cpu().tolist(),
            "origin":"free_ar","model_inputs":["source_foa_audio","raw_edit_instruction"],
        })
        audio_path = directory/"edited.wav"
        # An interrupted attempt can leave an uncommitted WAV. Keep it before
        # retrying; RESULT.json is the publication boundary.
        if audio_path.exists():
            preserved = directory/f"edited.unpublished.{gt.sha256_file(audio_path)}.wav"
            if preserved.exists():
                if preserved.read_bytes()!=audio_path.read_bytes():
                    raise RuntimeError("unpublished audio recovery artifact changed")
                audio_path.unlink()
            else:
                os.replace(audio_path,preserved)
        digest = audio._atomic_wav(audio_path,edited[0,:,:samples])
        check_audio(audio_path,digest,samples=samples)
        result = {
            "schema":"editing_clap44_audio_inference_v1","status":"COMPLETE",
            "contract":gt._artifact(contract_path),"release":contract["release"],
            "edited_foa":{"path":str(audio_path),"sha256":digest},"new_sceneplan":plan,
            "model_num_samples":samples,"sample_rate":44100,"channels":4,"runtime":report,
            "individual_edit_quality":"not_inferred_from_model_release_status",
        }
        _immutable(result_path,result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--release-sha256",required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--instruction",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"event":"clap44_editing_complete","edited_foa":result["edited_foa"],
                      "new_sceneplan":result["new_sceneplan"]},ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
