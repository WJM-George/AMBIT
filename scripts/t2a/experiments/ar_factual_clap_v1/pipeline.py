"""Load an actual AR candidate with its explicitly bound frozen source encoder.

The native candidate constructor, decoder and FOA frontend are preserved. This
adapter changes only dependency loading; plan/audio promotion remains separate.
Existing native20k AR controls are supported without rewriting their contracts.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap

import torch

from . import integration, training_binding
from stable_audio_tools.models import sceneplan_transfusion_editing_clap44_pipeline as native

SCHEMA = 'editing_ar_factual_clap_candidate_pipeline_loading_v1'
sha = integration.file_sha256
read = training_binding.read


def checkpoint_fields(reference):
    return {key: reference[key] for key in ('path', 'sha256', 'step')}


def validate_input_roles(contract):
    if (contract['training_mode'] != 'ar_pretrain' or contract['variant'] != 'global_and_sequence'
            or contract['runtime_inputs'] != ['source_foa_latent', 'raw_edit_request']
            or any(contract[key] is not False for key in ('old_plan_input', 'source_caption_input',
                'target_audio_ar_input', 'independent_test_used'))):
        raise RuntimeError('This AR planning candidate must use real source audio and raw instructions')


class CandidateBinding:
    def __init__(self, reference, *, preflight_path, native_control_validation_report=None):
        self.reference = copy.deepcopy(reference)
        self.preflight_path = Path(preflight_path).resolve(strict=True)
        self.native_control_validation_report = native_control_validation_report
        self.contract = self.encoder_reference = self.encoder_identity = self.dependency = None
        self.effect_receipt = None

    def load_AR(self, checkpoint):
        if self.contract is not None:
            raise RuntimeError('An inference binding loads its identified AR checkpoint once')
        if Path(checkpoint).resolve(strict=True) != Path(self.reference['checkpoint']).resolve(strict=True):
            raise RuntimeError('AR candidate path differs from the requested checkpoint identity')
        payload, identity = integration.load_joint_checkpoint(checkpoint, verify_sources=True, require_latest=False)
        if identity != self.reference:
            raise RuntimeError('AR candidate differs from its complete checkpoint identity')
        contract = payload['run_contract']
        validate_input_roles(contract)
        preflight = read(self.preflight_path)
        for split in ('train', 'validation'):
            if contract['indices'][split]['sha256'] != preflight['indices'][split]['sha256']:
                raise RuntimeError('AR and inference preflight refer to different populations')
        binding = contract['config'].get('clap_dependency')
        if binding is None:
            actual = contract['clap_checkpoint']
            if actual['sha256'] != training_binding.NATIVE20K_SHA or actual['step'] != 20000:
                raise RuntimeError('The retained legacy control must use its original native20k CLAP')
            if self.native_control_validation_report is None:
                raise RuntimeError('Legacy native20k control requires its complete20k diagnostic report')
            self.encoder_reference = {'format': 'native_clap44', **actual,
                'contract_sha256': sha(Path(actual['path']).parent / 'TRAIN_CONTRACT.json')}
            control = {'clap_dependency': {'schema': training_binding.SCHEMA,
                'checkpoint': self.encoder_reference, 'preflight': str(self.preflight_path),
                'validation_report': str(Path(self.native_control_validation_report).resolve(strict=True))}}
            self.effect_receipt = training_binding.effect_startup_check(control)
        else:
            if Path(binding['preflight']).resolve(strict=True) != self.preflight_path:
                raise RuntimeError('Inference preflight differs from the actual AR training binding')
            self.encoder_reference = copy.deepcopy(binding['checkpoint'])
            if checkpoint_fields(self.encoder_reference) != contract['clap_checkpoint']:
                raise RuntimeError('AR checkpoint and its frozen encoder training binding differ')
            if contract['clap_effect_selection'] != binding.get('effect_selection'):
                raise RuntimeError('AR encoder effect-selection provenance changed')
            self.effect_receipt = training_binding.effect_startup_check(contract['config'])
        self.contract = contract
        return payload, identity

    def load_encoder(self, contract):
        if self.contract is None or contract != self.contract or self.encoder_identity is not None:
            raise RuntimeError('Load the actual AR binding before its frozen source encoder')
        encoder, identity, dependency = integration.load_source_encoder(self.encoder_reference,
            preflight_path=self.preflight_path, device='cpu')
        if checkpoint_fields(identity) != contract['clap_checkpoint']:
            raise RuntimeError('Actual inference encoder differs from the AR checkpoint')
        if 'clap_dependency' in contract and dependency != contract['clap_dependency']:
            raise RuntimeError('Inference frozen encoder provenance differs from AR training')
        self.encoder_identity, self.dependency = identity, dependency
        return encoder, identity

    def frontend(self, identity):
        if self.encoder_identity is None or identity != self.encoder_identity:
            raise RuntimeError('VAE frontend verification requires the actual loaded encoder')
        # The unchanged actual encoder contract stays in encoder_identity. The
        # dependency record supplies its verified frontend regardless of format.
        return self.dependency['frontend_files']


def constructor(binding):
    """Compile exactly three reversible changes to the preserved constructor."""
    original = ast.parse(textwrap.dedent(inspect.getsource(native.load_clap44_joint_candidate)))
    counts = {'identified_AR_loading': 0, 'bound_encoder_loading': 0, 'actual_frontend_provenance': 0}
    class Edit(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'load_joint_checkpoint':
                counts['identified_AR_loading'] += 1
                saved = copy.deepcopy(node); node.func.id = '_load_identified_AR'
                node._candidate_original = saved
            return self.generic_visit(node)
        def visit_Assign(self, node):
            text = ast.unparse(node.targets[0])
            expression = None
            if text == '(clap, clap_identity)':
                assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'load_clap44_checkpoint'
                counts['bound_encoder_loading'] += 1; expression = '_load_bound_encoder(contract)'
            elif text == 'frontend':
                assert ast.unparse(node.value) == "clap_identity['contract']['frontend_files']"
                counts['actual_frontend_provenance'] += 1; expression = '_bound_frontend(clap_identity)'
            if expression:
                saved = copy.deepcopy(node); node.value = ast.parse(expression, mode='eval').body
                node._candidate_original = saved
            return self.generic_visit(node)
    tree = Edit().visit(copy.deepcopy(original))
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            return node._candidate_original if hasattr(node, '_candidate_original') else super().generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(tree)), include_attributes=False) == ast.dump(original, include_attributes=False)
    assert all(value == 1 for value in counts.values()), counts
    namespace = dict(native.__dict__, _load_identified_AR=binding.load_AR,
        _load_bound_encoder=binding.load_encoder, _bound_frontend=binding.frontend)
    exec(compile(ast.fix_missing_locations(tree), __file__ + '::preserved_native_constructor', 'exec'), namespace)
    return namespace['load_clap44_joint_candidate'], counts


def load_candidate(reference, *, preflight_path, native_control_validation_report=None,
                   device='cpu', load_audio_autoencoder=True):
    """Load a real identified AR checkpoint for candidate diagnostics.

    A factual encoder is derived only from the actual AR training binding and
    needs that encoder's existing effect acceptance. This does not accept AR,
    Editing DiT, or edited audio and is not a production-release loader.
    """
    binding = CandidateBinding(reference, preflight_path=preflight_path,
        native_control_validation_report=native_control_validation_report)
    build, counts = constructor(binding)
    with torch.random.fork_rng(devices=[]):
        pipeline, report = build(reference['checkpoint'], device=device,
            load_audio_autoencoder=load_audio_autoencoder)
    pipeline.requires_grad_(False)
    if report['clap_checkpoint'] != binding.dependency['checkpoint']:
        raise RuntimeError('Candidate pipeline report names a different frozen encoder')
    report.update(loading_schema=SCHEMA,
        actual_AR_identity=copy.deepcopy(reference), encoder_dependency=copy.deepcopy(binding.dependency),
        encoder_effect_receipt=copy.deepcopy(binding.effect_receipt),
        actual_encoder_training_contract=copy.deepcopy(binding.encoder_identity['contract']),
        inference_parameters_frozen=not any(p.requires_grad for p in pipeline.parameters()),
        preserved_native_constructor_changes=counts,
        loader_sources={str(Path(path).resolve()): sha(path) for path in
            (__file__, integration.__file__, training_binding.__file__, native.__file__)},
        run_contract_returned_without_relabeling=True,
        optimizer_scheduler_RNG_data_not_loaded_into_inference_model=True,
        quality_gate_passed=False, independent_test_used=False,
        scope='Candidate AR planning/runtime diagnostics only. Actual AR and DiT/audio effect acceptance and a future qualified joint release remain separate.')
    return pipeline, report
