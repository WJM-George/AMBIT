import pytest
import torch

from stable_audio_tools.inference.sampling import sample_discrete_euler
from stable_audio_tools.training.transfusion_opsd.event_objectives import native_resume_from_clean, native_continue_from_query
from stable_audio_tools.training.transfusion_opsd.objectives import EulerTrace, clean_prediction


@pytest.mark.parametrize('query', [0, 75, 99])
def test_zero_clean_intervention_exactly_replays_native_nonlinear_trajectory(query):
    generator = torch.Generator().manual_seed(719)
    initial = torch.randn(1, 4, 32, generator=generator)
    times = torch.linspace(1., 0., 101).sqrt()
    mask = torch.ones(1, 32, dtype=torch.bool)
    velocity = lambda z, t: torch.tanh(.73 * z) + .19 * t[:, None, None]
    states = []
    final = sample_discrete_euler(velocity, initial, times, disable_tqdm=True,
        callback=lambda values: states.append(values['x'].clone()))
    trace = EulerTrace(tuple([*states, final]), tuple(times.tolist()), mask)
    anchor = clean_prediction(states[query], times[query:query + 1],
        velocity(states[query], times[query:query + 1]))
    # Check the real native endpoint, not an independently rounded replica of
    # the intervention formula. Both explicit and reconstructed anchors work.
    assert torch.equal(native_resume_from_clean(trace, query, anchor, velocity, anchor=anchor), final)
    assert torch.equal(native_resume_from_clean(trace, query, anchor, velocity), final)
    changed = native_resume_from_clean(trace, query, anchor + .03, velocity, anchor=anchor)
    assert torch.isfinite(changed).all() and not torch.equal(changed, final)
    assert torch.equal(native_continue_from_query(trace, query, velocity), final)
    seen = []
    def revised(z, t):
        seen.append(z.detach().clone())
        return velocity(z, t) + .17
    continued = native_continue_from_query(trace, query, revised)
    assert torch.equal(seen[0], states[query]) and len(seen) == 100 - query
    assert not torch.equal(continued, final)
    assert torch.equal(trace.states[query], states[query])
    branched = native_continue_from_query(trace, query, revised, return_trace=True)
    assert len(branched.states) == len(trace.states) and branched.times == trace.times
    assert all(torch.equal(a, b) for a, b in zip(branched.states[:query + 1], trace.states[:query + 1]))
    assert torch.equal(branched.states[-1], continued)
