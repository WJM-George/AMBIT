"""Observe native Qwen kernel choices across every supported batch/length key."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from scripts.t2a.experiments.ar_structured_v1 import runtime as rt


def run(args):
    cfg = rt.read(args.config)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    rank, local, world, device, topology = rt.allocated_runtime.distributed(timeout_seconds=900)
    out = args.output/f'rank{rank}'; out.mkdir(parents=True, exist_ok=False)
    finish = rt.install_autotune_observer({'name': f'rank{rank}', 'mode': 'capture'}, args.output)
    try:
        encoder = cfg['base_AR_configuration']['clap_dependency']['checkpoint']['path']
        text = rt.read(Path(encoder).parent/'TRAIN_CONTRACT.json')['native_config']['text']
        teacher = rt.FrozenCLAP44TextFeatures(text['model_path'], max_tokens=text['max_tokens'],
            batch_size=text['batch_size']).eval()
        tokens = teacher.conditioner.tokenizer('A bird is chirping.', add_special_tokens=True)['input_ids']
        shapes = [(batch, 65) for batch in range(1,65)] + [(32,length) for length in range(128,1025,128)]
        # Eight NB buckets cover B32x1024 teacher and B64x512 AR/RF.
        for batch, length in shapes:
            ids = torch.tensor((tokens*((length+len(tokens)-1)//len(tokens)))[:length])
            row = {'input_ids': ids, 'attention_mask': torch.ones(length,dtype=torch.bool)}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                values, mask = teacher.conditioner([row]*batch, device)
            if not bool(torch.isfinite(values).all()) or not bool(mask.all()):
                raise RuntimeError('Invalid frozen-Qwen kernel coverage probe')
            if rank == 0 and (batch % 8 == 0 or length != 65):
                print(json.dumps({'kernel_shape': [batch,length]}), flush=True)
            del values, mask
        finish()
        catalog = rt.read(out/'AUTOTUNE_CATALOG.json')
        records = list(catalog['records'].values())
        seen_batches = {r['native_autotune_key'][0] for r in records if r['function'].endswith('chunk_local_cumsum_scalar_kernel')}
        seen_NB = {r['native_autotune_key'][1] for r in records if r['function'].endswith('l2norm_fwd_kernel')}
        if not set(range(1,65)) <= seen_batches or not set(range(1,9)) <= seen_NB:
            raise RuntimeError('Native Qwen autotune key coverage is incomplete')
        rt.write(out/'COMPLETE.json', {'at': rt.now(), 'optimizer_updates': 0,
            'synthetic_token_sequences_only_for_kernel_coverage': True,
            'batch_sizes': sorted(seen_batches), 'normalization_NB': sorted(seen_NB),
            'frozen_Qwen_sha256': rt.state_hash(teacher.conditioner.model),
            'records': len(records), 'quality_gate_passed': False})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
