"""The sampler repair must preserve batch membership and exact row coverage."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = load('generation_ar_dataset_for_test', ROOT / 'stable_audio_tools/data/sceneplan_transfusion_generation_ar_dataset.py')
recipe = load('generation_ar_recipe_for_test', ROOT / 'scripts/t2a/train/generation_ar_sampling.py')


def orders(lengths, epoch, repaired):
    result = []
    for rank in range(3):
        sampler = base_module.LengthBucketDistributedSampler(lengths, num_replicas=3, rank=rank, batch_size=4, bucket_batches=64)
        if repaired:
            sampler = recipe.ShuffledGlobalBatchSampler(sampler)
        sampler.set_epoch(epoch)
        result.append(np.asarray(list(sampler)))
    return result


def test_preserves_uneven_tail_and_every_global_batch():
    lengths = np.arange(154) + 1
    original, changed = orders(lengths, 11, False), orders(lengths, 11, True)
    assert np.array_equal(np.sort(np.concatenate(changed)), np.arange(154))
    full_local = (154 // 12) * 4
    def batches(ranks):
        return sorted(tuple(x) for x in np.concatenate([r[:full_local].reshape(-1,4) for r in ranks], axis=1))
    assert batches(original) == batches(changed)
    for before, after in zip(original, changed):
        assert np.array_equal(before[full_local:], after[full_local:])
    assert all(np.array_equal(a,b) for a,b in zip(changed,orders(lengths,11,True)))
    assert not all(np.array_equal(a,b) for a,b in zip(changed,orders(lengths,12,True)))


def test_breaks_long_runs_of_one_length_class_without_extra_padding():
    labels = np.repeat(np.arange(4), 64 * 12)
    lengths = (labels + 1) * 100
    legacy, repaired = orders(lengths, 10, False), orders(lengths, 10, True)
    old_classes = labels[legacy[0].reshape(-1,4)[:,0]]
    new_classes = labels[repaired[0].reshape(-1,4)[:,0]]
    assert len(set(old_classes[:64])) == 1
    assert len(set(new_classes[:64])) == 4
    assert np.array_equal(np.sort(old_classes), np.sort(new_classes))
    for rank in repaired:
        # All local batches retain the original equal-length members.
        assert (lengths[rank.reshape(-1,4)].ptp(axis=1) == 0).all()
