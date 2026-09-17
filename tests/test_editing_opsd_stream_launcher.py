"""CPU-only launch checks. No external processes or GPU calls are allowed here."""
import hashlib
import json

import pytest

from scripts.t2a.rl import launch_editing_opsd_comparison as launcher


def configuration(directory):
    hashes = {}
    for world in (2, 4):
        for arm in ('off', 'on'):
            q = dict(arm=arm, output=str(directory/arm), connected_credit=arm == 'on',
                     physical_gpus=list(range(world)) if arm == 'off' else list(range(world, world*2)),
                     request_rows_per_rank=4//world, paired_rows_per_rank=32//world,
                     global_request_batch=4, global_paired_batch=32,
                     maximum_updates=1000, save_every=250)
            if world == 4:
                q['resize_parent_config'] = str(directory/f'{arm}.json')
            path = directory/f'{arm}{"4" if world == 4 else ""}.json'
            path.write_text(json.dumps(q))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/'PROTOCOL.json').write_text(json.dumps(dict(sources={}, configurations=hashes)))


def forbid_processes(*args, **kwargs):
    raise AssertionError('CPU-only verification must not query a GPU or launch a process.')


def test_user_hold_blocks_even_before_gpu_observation(tmp_path, monkeypatch):
    (tmp_path/'USER_HOLD.json').write_text('{}')
    monkeypatch.setattr(launcher.subprocess, 'run', forbid_processes)
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbid_processes)
    # No protocol is needed: the hold must be checked before any other work.
    with pytest.raises(RuntimeError, match='explicit go-ahead'):
        launcher.main(['--run-dir', str(tmp_path), '--expand', '--resume', '--resize'])
    assert not (tmp_path/'LAUNCH_STATUS.json').exists()


def test_dry_run_and_expansion_preserve_schedule_and_batch(tmp_path, monkeypatch, capsys):
    configuration(tmp_path)
    (tmp_path/'USER_HOLD.json').write_text('{}')
    monkeypatch.setattr(launcher.subprocess, 'run', forbid_processes)
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbid_processes)
    launcher.main(['--run-dir', str(tmp_path), '--dry-run', '--expand', '--resume', '--resize'])
    result = json.loads(capsys.readouterr().out)
    assert result['user_hold'] and result['gpu_processes_started'] == 0
    for command in result['commands'].values():
        assert '--nproc_per_node=4' in command and '--resize' in command
    for expanded in (False, True):
        _, jobs = launcher.prepare(tmp_path, expand=expanded)
        assert [jobs[arm]['config']['connected_credit'] for arm in ('off', 'on')] == [False, True]
        for job in jobs.values():
            q = job['config']
            world = len(q['physical_gpus'])
            assert q['request_rows_per_rank']*world == 4
            assert q['paired_rows_per_rank']*world == 32
            assert q['maximum_updates'] == 1000 and q['save_every'] == 250


def test_changed_config_and_implicit_resize_are_rejected(tmp_path):
    configuration(tmp_path)
    with pytest.raises(ValueError, match='requires'):
        launcher.prepare(tmp_path, expand=True, resize=True)
    with pytest.raises(ValueError, match='positive'):
        launcher.prepare(tmp_path, limit_updates=0)
    path = tmp_path/'on.json'
    q = json.loads(path.read_text()); q['save_every'] = 10
    path.write_text(json.dumps(q))
    with pytest.raises(ValueError, match='changed'):
        launcher.prepare(tmp_path)


def test_forced_evaluation_is_identical_for_both_arms_without_gpu_calls(tmp_path, monkeypatch, capsys):
    configuration(tmp_path)
    monkeypatch.setattr(launcher.subprocess, 'run', forbid_processes)
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbid_processes)
    launcher.main(['--run-dir',str(tmp_path),'--dry-run','--expand','--resume',
                   '--evaluate-at-updates','90','95'])
    result=json.loads(capsys.readouterr().out)
    for command in result['commands'].values():
        assert command[-3:] == ['--evaluate-at-updates','90','95']
    with pytest.raises(ValueError, match='within the training schedule'):
        launcher.main(['--run-dir',str(tmp_path),'--dry-run','--evaluate-at-updates','1001'])
