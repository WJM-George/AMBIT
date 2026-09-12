"""Dispatch native frozen readers while preserving pair identities and controls."""
from common import *
from bisect import bisect_right
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset as OriginalDataset
from compound_dataset import ScenePlanTransfusionEditingDataset as CompoundDataset


class MixtureDataset:
    def __init__(self,binding,tokenizer_spec,*,sample_cases=None):
        self.binding=binding;self.tokenizer=tokenizer_spec[0];self.split=binding['split']
        self._connection=None;self.datasets=[];self.starts=[];self.sample_cases=sample_cases
        for c in binding['components']:
            cls=CompoundDataset if c['name']=='spatial_multi500k' else OriginalDataset
            dataset=cls(c['index_path'],tokenizer_spec=tokenizer_spec,
                expected_num_samples=c['rows'],index_num_samples=c['rows'],expected_index_sha256=c['index_sha256'],
                latent_crop_length=648,require_frozen=True,verify_tensor_hashes_on_access=True)
            assert dataset.split==self.split
            self.datasets.append(dataset);self.starts.append(c['offset'])
        assert self.starts[0]==0
        self._length=len(sample_cases) if sample_cases is not None else binding['rows']

    def __len__(self):return self._length

    def __getitem__(self,index):
        assert 0<=index<len(self)
        ordinal=self.sample_cases[index]['pair_ordinal'] if self.sample_cases is not None else index
        component=bisect_right(self.starts,ordinal)-1;local=ordinal-self.starts[component]
        target,metadata=self.datasets[component][local]
        assert metadata['pair_ordinal']==local
        metadata=dict(metadata,native_pair_ordinal=local,pair_ordinal=ordinal,
            dataset_component=self.binding['components'][component]['name'])
        if self.sample_cases is not None:
            spec=self.sample_cases[index]
            assert metadata['pair_id']==spec['pair_id'] and metadata['dataset_component']==spec['cohort']
        return target,metadata

    def close(self):
        for dataset in self.datasets:
            if dataset._connection is not None:dataset._connection.close();dataset._connection=None

    def __getstate__(self):
        state=dict(self.__dict__);state['_connection']=None
        return state
