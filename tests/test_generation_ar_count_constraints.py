from stable_audio_tools.inference.sceneplan_generation_ar_constraints import generate_with_declared_source_count


class Codec:
    def allowed_next_ids(self,prefix,*,min_sources=1,max_sources=4):return set(range(min_sources,max_sources+1))


class Model:
    def generate_constrained(self,requests,codec,*,device,max_plan_tokens):
        # The unconstrained model always prefers four; request-derived masks
        # restrict this without reference labels being supplied to the helper.
        return [[max(codec.allowed_next_ids([]))] for _ in requests]


def source(index):return f'Source {index}: sound; description=“dog barking”; active from 0 to 1000ms; static at (0, 0, 1000).'


def test_restores_request_order_and_exposes_constraint_metadata():
    result=generate_with_declared_source_count(Model(),[source(1)+' '+source(2),source(1),'A bird is singing.',source(1)+' '+source(2)],Codec(),device='cpu')
    assert result['token_ids']==[[2],[1],[4],[2]]
    assert result['declared_source_counts']==[2,1,None,2]
    assert result['count_constraint_applied']==[True,True,False,True]


def test_ambiguous_source_numbering_preserves_model_fallback():
    result=generate_with_declared_source_count(Model(),[source(1)+' '+source(1)],Codec(),device='cpu')
    assert result['token_ids']==[[4]]
    assert result['count_constraint_applied']==[False]
