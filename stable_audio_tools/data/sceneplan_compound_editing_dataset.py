"""Compound spatial editing contracts over the shared, strict latent reader."""

from .sceneplan_transfusion_editing_dataset import (
    ScenePlanTransfusionEditingDataset as BaseEditingDataset,
)

EDITING_PAIR_CONTRACT = "spatial_only_compound_edit_actions_v2"
EDITING_INSTRUCTION_CONTRACT = "unambiguous_per_source_raw_instructions_v2"


class ScenePlanTransfusionEditingDataset(BaseEditingDataset):
    """Accept the frozen v2 contract without widening the original v1 reader."""

    pair_contract = EDITING_PAIR_CONTRACT
    instruction_contract = EDITING_INSTRUCTION_CONTRACT
    operation_count_deltas = {
        **BaseEditingDataset.operation_count_deltas,
        "multi_source_relocation": 0,
        "source_position_swap": 0,
        "multi_source_static_to_linear": 0,
        "multi_source_linear_to_static": 0,
        "multi_source_motion_toggle": 0,
        "multi_source_position_cycle": 0,
    }
