"""CPU-only verification of the custom reducer and resumed Adam updates."""
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.t2a.rl.train_editing_opsd_stream import synchronize_gradients


def make_model():
    torch.manual_seed(42)
    return torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2))


def objective(model, rank, step):
    x = torch.tensor([[float(rank+1), float(step+1)]])
    y = torch.tensor([[.2*rank, .4*step]])
    return (model(x)-y).square().mean()


def worker(rank, rendezvous, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.9,.95))
        objective(model, rank, 0).backward()
        synchronize_gradients(list(model.parameters()), 2)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        saved = Path(result_dir)/f'resume{rank}.pt'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()}, saved)
        restored = make_model()
        opt2 = torch.optim.AdamW(restored.parameters(), lr=.003, betas=(.9,.95))
        state = torch.load(saved, weights_only=False)
        restored.load_state_dict(state['model']);opt2.load_state_dict(state['optimizer'])
        for net,opt in [(model,optimizer),(restored,opt2)]:
            objective(net, rank, 1).backward()
            synchronize_gradients(list(net.parameters()), 2)
            opt.step()
        for a,b in zip(model.parameters(), restored.parameters()):
            assert torch.equal(a,b)
        torch.save(model.state_dict(), Path(result_dir)/f'final{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_distributed_updates_match_global_reference_and_resume(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'rendezvous'),str(tmp_path)), nprocs=2, join=True)
    reference = make_model()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, betas=(.9,.95))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        ((objective(reference,0,step)+objective(reference,1,step))/2).backward()
        optimizer.step()
    states = [torch.load(tmp_path/f'final{rank}.pt',weights_only=True) for rank in range(2)]
    for name,value in reference.state_dict().items():
        assert torch.equal(states[0][name],states[1][name])
        assert torch.allclose(value,states[0][name],rtol=1e-6,atol=1e-7)
