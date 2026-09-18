#!/usr/bin/env python3
"""One real-model validation batch: output parity and wall-time benchmark."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

SNAPSHOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/source_snapshots/snapshot")
REPO = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2')
    sys.path.insert(0, str(SNAPSHOT))
    import torch
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = REPO/'stable_audio_tools/inference/sceneplan_generation_ar_vectorized.py'
    spec = importlib.util.spec_from_file_location('vectorized_ar', path)
    vector = importlib.util.module_from_spec(spec); spec.loader.exec_module(vector)
    torch.set_num_threads(4); torch.manual_seed(42); device = torch.device('cuda:0')
    codec = ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    model, p10 = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_trainable_state_dict(state['ar_adapter']); del state
    model.p10_dit.to(device=device, dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.eval()
    db = sqlite3.connect('file:${AMBIT_CKPT_ROOT}/generation_ar/diagnosis/panel.sqlite?mode=ro&immutable=1', uri=True)
    rows = [r for n in range(1,5) for r in db.execute('SELECT ordinal,raw_user_request FROM rows WHERE source_count=? ORDER BY ordinal LIMIT 8',(n,))]
    db.close()
    measurements = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        # Warm up both paths before timing either one.
        warm = model.generate_constrained([rows[0][1]], codec, device=device, max_plan_tokens=512)
        assert vector.generate_constrained_vectorized(model, [rows[0][1]], codec, device=device, max_plan_tokens=512) == warm
        for batch_size in (4,32):
            selected = rows[::8] if batch_size==4 else rows
            requests = [r[1] for r in selected]
            torch.cuda.synchronize(); start = time.perf_counter()
            original = model.generate_constrained(requests, codec, device=device, max_plan_tokens=512)
            torch.cuda.synchronize(); original_s = time.perf_counter()-start
            start = time.perf_counter()
            fast = vector.generate_constrained_vectorized(model, requests, codec, device=device, max_plan_tokens=512)
            torch.cuda.synchronize(); fast_s = time.perf_counter()-start
            assert fast == original, 'vectorized greedy changed generated tokens'
            measurements.append({'batch_size':batch_size,'ordinals':[r[0] for r in selected],'exact_tokens':True,
                                 'original_s':original_s,'vectorized_s':fast_s,'speedup':original_s/fast_s,
                                 'generated_tokens':sum(len(ids) for ids in original)})
    report = {'status':'PASS','validation_only':True,'checkpoint':str(args.checkpoint),
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'generator_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'measurements':measurements,'limit':'One repeat per size; limited parity panel, not a generation quality improvement.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
