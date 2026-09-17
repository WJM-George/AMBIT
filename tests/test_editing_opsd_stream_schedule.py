"""A schedule edit must preserve resumable training and all recipe controls."""
import json
import pytest
from scripts.t2a.rl.train_editing_opsd_stream import (
    sha, validate_resume_configuration, is_milestone, update_events)


def scheduled_config(tmp_path):
    parent = tmp_path/'old.json'
    old = dict(maximum_updates=20000, save_every=5000, recovery_every_seconds=900,
               learning_rates={'AR': 2.5e-5}, seed=42, connected_credit=True)
    parent.write_text(json.dumps(old))
    current = tmp_path/'new.json'
    new = dict(old, maximum_updates=1000, save_every=250, recovery_every_seconds=300,
               resume_config_parent=str(parent))
    current.write_text(json.dumps(new))
    return parent, current, new


def test_schedule_migration_accepts_old_and_new_checkpoints(tmp_path):
    parent, current, _ = scheduled_config(tmp_path)
    validate_resume_configuration(current, sha(parent))
    validate_resume_configuration(current, sha(current))


@pytest.mark.parametrize('field,value', [('learning_rates', {'AR': 1e-3}),
                                        ('seed', 43), ('connected_credit', False)])
def test_schedule_migration_rejects_recipe_changes(tmp_path, field, value):
    parent, current, q = scheduled_config(tmp_path)
    q[field] = value
    current.write_text(json.dumps(q))
    with pytest.raises(ValueError, match='Schedule-only'):
        validate_resume_configuration(current, sha(parent))


def test_schedule_parent_must_match_checkpoint_hash(tmp_path):
    parent, current, _ = scheduled_config(tmp_path)
    with pytest.raises(ValueError, match='verified schedule parent'):
        validate_resume_configuration(current, '0'*64)


def test_four_milestones_are_cumulative_not_added_after_resume():
    q = dict(maximum_updates=1000, save_every=250)
    assert [step for step in range(22,1001) if is_milestone(step,q)] == [250,500,750,1000]


def test_periodic_evaluation_cannot_run_past_the_saved_optimizer_boundary():
    q = dict(maximum_updates=1000, save_every=250, recovery_every_seconds=300,
             evaluate_every_seconds=1800)
    # Reproduces the old step89 transition: newest checkpoint was six updates
    # behind, but the independent recovery timer was not due yet.
    assert update_events(89,q,since_save=160,since_evaluation=1801) == (True,True)
    assert update_events(90,q,since_save=10,since_evaluation=10,extra_evaluations=[90]) == (True,True)
    assert update_events(250,q,since_save=10,since_evaluation=10) == (True,True)
    assert update_events(1000,q,since_save=10,since_evaluation=10) == (True,True)
    assert update_events(91,q,since_save=301,since_evaluation=10) == (True,False)
    assert update_events(91,q,since_save=1,since_evaluation=10) == (False,False)
