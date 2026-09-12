"""Reuse the established RF loss/CPU batch compiler with the expanded index."""
import sys
from pathlib import Path
from common import MAIN,OLD_RUN,read,sha
sys.path[:0]=[str(MAIN),str(OLD_RUN)]
import numpy as np
from balanced_inputs_v1 import BalancedInputs
from balanced_stream_v1 import BalancedStream,weighted_rf_loss
from throughput_cpu_loader_v1 import OrderedCPUStream
from mixture_dataset import MixtureDataset

class MixedStream(BalancedInputs):
    def __init__(self,run,diffusion,pipeline,rank,device,start,stop,chains,numa_node):
        plan=read(run/'PLAN.json');self.plan=plan;self.pipeline=pipeline;self.rank=rank;self.device=device
        for ref in plan['input_arrays'].values():assert sha(ref['path'])==ref['sha256']
        self.order=np.load(run/'ORDER.npy',mmap_mode='r',allow_pickle=False)
        self.times=np.load(run/'TIMES.npy',mmap_mode='r',allow_pickle=False)
        self.group_order=np.load(run/'GROUP_ORDER.npy',mmap_mode='r',allow_pickle=False)
        self.group_counts=np.load(run/'GROUP_COUNTS.npy',mmap_mode='r',allow_pickle=False)
        assert self.order.shape[1:]==(5,4) and self.times.shape==(500,5,4)
        assert self.group_order.shape==(30000,3,18)
        self.schedule={'noise_seed_offset':plan['noise_seed_offset']}
        self.dataset=MixtureDataset(plan['dataset'],(diffusion.conditioner.conditioners['prompt'].tokenizer,512))
        assert self.dataset.split=='train'
        self.chains=dict(chains or {'data':'0'*64,'noise':'0'*64})
        self.cursor=start;self.stop=stop
        selections=(self.group_order[step,rank,:int(self.group_counts[step])] for step in range(start,stop))
        self.stream=OrderedCPUStream(self,selections,workers=plan['CPU_worker_processes_per_rank'],numa_node=numa_node)

    def next(self,step):
        assert step==self.cursor and step<self.stop
        batch=self.stream.next();n=int(self.group_counts[step]);assert len(batch['rows'])==4*n
        batch['valid_groups']=np.ones(n,dtype=np.bool_);batch['global_valid_groups']=3*n
        batch['audit'].update(phase_update=step,valid_groups=[True]*n,global_valid_groups=3*n,optimizer_step=50000+step+1)
        self.cursor+=1;return batch

    commit=BalancedStream.commit

    def close(self):
        self.stream.close();self.dataset.close()
