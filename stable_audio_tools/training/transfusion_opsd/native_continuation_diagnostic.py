"""Continue one declared native state without restarting its flow schedule."""
from __future__ import annotations

import torch


@torch.no_grad()
def continue_native_query(velocity, query, full_schedule, *, first_velocity=None):
    """Optionally replace exactly the first velocity, then use one executor.

    The caller separately records model calls used to construct the injected
    velocity and validates outputs. This function supplies no quality label.
    """
    from ...inference.sampling import sample_discrete_euler
    z, time = query['z'], query['time']
    index = query['query_index']
    schedule = torch.as_tensor(full_schedule, device=z.device, dtype=torch.float32)
    if (type(index) is not int or not 0 <= index < len(schedule)-1
            or schedule.ndim != 1 or len(schedule) != 101
            or not torch.isfinite(schedule).all() or not (schedule[:-1] > schedule[1:]).all()
            or schedule[-1] != 0 or time.shape != (z.shape[0],)
            or not torch.isfinite(z).all() or not torch.isfinite(time).all()
            or not torch.equal(time.to(z.device), schedule[index].expand_as(time))):
        raise ValueError('State and time must match the declared 100-step native schedule.')
    if first_velocity is not None:
        if (first_velocity.shape != z.shape or first_velocity.device != z.device
                or not torch.isfinite(first_velocity).all()):
            raise ValueError('Injected velocity must align with the actual native state.')
        fixed = first_velocity.detach()
        first = True
        def dispatch(state, t):
            nonlocal first
            if first:
                first = False
                return fixed
            return velocity(state, t)
    else:
        dispatch = velocity
    return sample_discrete_euler(dispatch, z.detach().clone(), schedule[index:], disable_tqdm=True)
