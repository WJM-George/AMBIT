import pytest
import torch
from torch.utils.data import DistributedSampler

from stable_audio_tools.data.resumable_dataloader import ResumableDataLoader


def _values(loader):
    return torch.cat([batch for batch in loader]).tolist()


def test_resumes_at_next_unconsumed_batch():
    loader = ResumableDataLoader(
        torch.arange(12), batch_size=2, shuffle=False, in_order=True
    )
    iterator = iter(loader)
    assert next(iterator).tolist() == [0, 1]
    assert next(iterator).tolist() == [2, 3]

    state = loader.state_dict()
    assert state["batches_yielded"] == 2

    restored = ResumableDataLoader(
        torch.arange(12), batch_size=2, shuffle=False, in_order=True
    )
    restored.load_state_dict(state)
    assert _values(restored) == list(range(4, 12))


def test_completed_epoch_resumes_with_zero_cursor():
    loader = ResumableDataLoader(
        torch.arange(6), batch_size=2, shuffle=False, in_order=True
    )
    assert _values(loader) == list(range(6))
    assert loader.state_dict()["batches_yielded"] == 0


def test_rejects_unordered_delivery():
    with pytest.raises(ValueError, match="in_order=True"):
        ResumableDataLoader(
            torch.arange(6), batch_size=2, shuffle=False, in_order=False
        )


def test_rejects_changed_epoch_shape():
    source = ResumableDataLoader(
        torch.arange(6), batch_size=2, shuffle=False, in_order=True
    )
    next(iter(source))

    changed = ResumableDataLoader(
        torch.arange(8), batch_size=2, shuffle=False, in_order=True
    )
    with pytest.raises(ValueError, match="epoch length changed"):
        changed.load_state_dict(source.state_dict())


def test_mid_epoch_guard_rejects_legacy_checkpoint_without_cursor():
    loader = ResumableDataLoader(
        torch.arange(6), batch_size=2, shuffle=False, in_order=True
    )
    with pytest.raises(RuntimeError, match="has no resumable DataLoader cursor"):
        loader.assert_resume_state_loaded()

    loader.load_state_dict(loader.state_dict())
    loader.assert_resume_state_loaded()


def test_lightning_sampler_injection_preserves_resumable_loader():
    from pytorch_lightning.trainer.states import RunningStage
    from pytorch_lightning.utilities.data import _update_dataloader

    dataset = torch.arange(12)
    loader = ResumableDataLoader(
        dataset, batch_size=2, shuffle=True, in_order=True
    )
    sampler = DistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=42
    )

    updated = _update_dataloader(loader, sampler, RunningStage.TRAINING)

    assert isinstance(updated, ResumableDataLoader)
    assert updated.sampler is sampler
    assert len(updated) == 3
