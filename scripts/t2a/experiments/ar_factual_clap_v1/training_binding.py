"""Explicit frozen CLAP dependency hooks for the existing AR training loop."""
import copy
import json
from pathlib import Path

from . import integration

native = integration.native
sha = integration.file_sha256
SCHEMA = 'editing_ar_factual_clap_training_binding_v1'
EFFECT_SCHEMA = 'clap44_factual_encoder_effect_acceptance_v1'
NATIVE20K_SHA = '844490fb1a2de0c98091a2dec2aef72a049f0984a4f97909b9a81bbf7bf55300'


def read(path):
    return json.loads(Path(path).read_text())


def effect_startup_check(cfg):
    """Enforce the user-defined encoder acceptance boundary before GPU setup.

    A protected native20k control remains available for matched adaptation.
    Factual candidates need a separate effect review of completed20k results;
    neither fit states nor the CPU interface proof constitute that review.
    """
    binding = cfg['clap_dependency']
    if binding['schema'] != SCHEMA:
        raise RuntimeError('AR frozen encoder dependency schema changed')
    reference = binding['checkpoint']
    if reference['format'] == 'native_clap44':
        if reference['step'] != 20000 or reference['sha256'] != NATIVE20K_SHA:
            raise RuntimeError('Only the protected native20k matched control is declared')
        report = read(binding['validation_report'])
        if ({k: report['checkpoint'][k] for k in ('path', 'sha256', 'step')}
                != {k: reference[k] for k in ('path', 'sha256', 'step')}):
            raise RuntimeError('Native control diagnostics refer to another encoder')
        native.audit_clap_validation(binding['validation_report'], reference,
            read(binding['preflight']), require_full=True)
        return {'scope': 'Protected native20k control, not a newly accepted factual encoder',
                'checkpoint_sha256': NATIVE20K_SHA, 'factual_encoder_effect_accepted': False}
    if reference['format'] != 'factual50k':
        raise RuntimeError('Unsupported AR source encoder format')
    selected = binding.get('effect_selection')
    if not selected:
        raise RuntimeError('Factual encoder effect acceptance is required before AR GPU training')
    path = Path(selected['path']).resolve(strict=True)
    if sha(path) != selected['sha256']:
        raise RuntimeError('Factual encoder effect review changed')
    value = read(path)
    if (value.get('schema') != EFFECT_SCHEMA or value.get('quality_gate_passed') is not True
            or value.get('independent_test_used') is not False
            or value.get('selected_encoder') != reference
            or value.get('semantic_retention') is not True
            or value.get('spatiotemporal_binding_improved') is not True):
        raise RuntimeError('The bound review does not accept this factual encoder')
    comparison = value['comparison']
    if sha(comparison['path']) != comparison['sha256']:
        raise RuntimeError('Accepted full20k comparison changed')
    compared = read(comparison['path'])
    if (compared.get('both_text_views_method_screens_passed') is not True
            or compared.get('pairs_per_view') != 20000 or compared.get('independent_test_used') is not False
            or compared['checkpoint']['checkpoint'] != reference['path']
            or compared['checkpoint']['sha256'] != reference['sha256']
            or compared['checkpoint']['new_updates'] != reference['step']
            or compared['checkpoint']['contract_sha256'] != reference['contract_sha256']):
        raise RuntimeError('Accepted comparison has different population or encoder identity')
    validations = value['validation_reports']
    if set(validations) != {'native', 'factual20'}:
        raise RuntimeError('Both complete20k text views are required')
    if Path(validations['native']['path']).resolve() != Path(binding['validation_report']).resolve():
        raise RuntimeError('AR startup must use the selected native-view diagnostics')
    for view, record in validations.items():
        report_path = Path(record['path']).resolve(strict=True)
        if sha(report_path) != record['sha256']:
            raise RuntimeError('Accepted encoder diagnostics changed')
        report = read(report_path)
        if (report['pairs'] != 20000 or report['audio_views'] != 40000
                or report['full_validation'] is not True or report['independent_test_used'] is not False
                or {k: report['checkpoint'][k] for k in ('path', 'sha256', 'step')}
                != {k: reference[k] for k in ('path', 'sha256', 'step')}):
            raise RuntimeError('Accepted diagnostics have different checkpoint or population')
        proof = read(report_path.parent / 'CAPTURE_PROVENANCE.json')
        if proof['caption_view'] != view or proof['raw_edit_request_consumed']:
            raise RuntimeError('Accepted CLAP text roles changed')
        native.audit_clap_validation(report_path, reference, read(binding['preflight']), require_full=True)
    return {'scope': 'Effect-accepted factual encoder for AR adaptation; full editing quality remains unaccepted',
            'effect_selection': selected, 'comparison': comparison,
            'validation_reports': validations, 'factual_encoder_effect_accepted': True}


class TrainingEncoderBinding:
    """Keep the actual CLAP contract outside the legacy metric configuration view."""
    def __init__(self, cfg):
        self.binding = copy.deepcopy(cfg['clap_dependency'])
        if self.binding['schema'] != SCHEMA:
            raise RuntimeError('AR frozen encoder binding is invalid')
        self.identity = None
        self.dependency = None

    def load(self, path, *, device='cpu'):
        reference = self.binding['checkpoint']
        if Path(path).resolve(strict=True) != Path(reference['path']).resolve(strict=True):
            raise RuntimeError('AR encoder argument differs from its bound selection')
        model, identity, record = integration.load_source_encoder(reference,
            preflight_path=self.binding['preflight'], device=device)
        self.identity, self.dependency = identity, record
        return model, identity

    def preflight_matches(self, identity, path):
        return (self.identity is not None and identity == self.identity
                and Path(path).resolve(strict=True) == Path(self.binding['preflight']).resolve(strict=True)
                and sha(path) == self.dependency['preflight_sha256'])

    def audit_validation(self, path, checkpoint, preflight, *, require_full=True):
        if Path(path).resolve(strict=True) != Path(self.binding['validation_report']).resolve(strict=True):
            raise RuntimeError('AR validation argument differs from the declared encoder report')
        if not require_full:
            raise RuntimeError('AR startup requires complete20k encoder diagnostics')
        if self.identity is None or checkpoint != self.identity:
            raise RuntimeError('AR validation must use the actual loaded encoder identity')
        return native.audit_clap_validation(path, checkpoint, preflight, require_full=True)

    def amend_contract(self, contract, cfg, parent_amend):
        parent_amend(contract, cfg)
        if self.identity is None or self.dependency is None:
            raise RuntimeError('AR contract was built before loading its frozen encoder')
        if contract['clap_checkpoint'] != self.dependency['checkpoint']:
            raise RuntimeError('AR contract and actual frozen encoder differ')
        if contract['runtime_inputs'] != ['source_foa_latent', 'raw_edit_request']:
            raise RuntimeError('Factual scene descriptions must stay outside AR inputs')
        contract['clap_dependency'] = copy.deepcopy(self.dependency)
        contract['clap_effect_selection'] = copy.deepcopy(self.binding.get('effect_selection'))
        contract['clap_dependency_scope'] = 'Frozen audio encoder only; no CLAP readout, optimizer, scheduler, RNG or data cursor enters AR state.'


def training_namespace(namespace, cfg, hooks):
    binding = TrainingEncoderBinding(cfg)
    original_amend = namespace['_text_contract']
    def amend(contract, config):
        binding.amend_contract(contract, config, original_amend)
    return {**namespace, 'load_clap44_checkpoint': binding.load,
        'audit_clap_validation': binding.audit_validation,
        '_clap_preflight_matches': binding.preflight_matches,
        '_text_contract': amend, '_factual_clap_binding': binding}
