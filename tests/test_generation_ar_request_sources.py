"""Input source features must not interpret quoted text as structural labels."""
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'stable_audio_tools/data/sceneplan_generation_ar_request_sources.py'
spec=importlib.util.spec_from_file_location('request_sources_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def clause(lead='Source 1',description='dog barking',coordinate='(10, 0, 1000)'):
    return f'{lead}: sound; description=“{description}”; active from 0 to 1000ms; static at {coordinate}.'


def test_numbered_and_ordinal_sources_and_global_closer():
    text='Create audio. '+clause()+' '+clause('Second source')+' Keep all gains at zero.'
    spans=module.request_source_spans(text)
    assert [s.source_number for s in spans]==[1,2]
    assert text[spans[-1].end:]==' Keep all gains at zero.'
    assert module.source_ids_for_offsets([(0,0),(0,6),(spans[0].start,spans[0].start+6),(spans[-1].end+1,len(text))],spans)==[0,0,1,0]


def test_source_header_inside_transcript_is_ignored():
    text=clause(description='A person says Source 2: sound; and then pauses.')+' '+clause('Source 2')
    spans=module.request_source_spans(text)
    assert [s.source_number for s in spans]==[1,2]


def test_nested_curly_quotes_and_decimal_coordinates():
    text=clause(description='A voice says “Source 3: sound;” twice.',coordinate='(-1.5, 0.2, 1000)')
    spans=module.request_source_spans(text)
    assert len(spans)==1 and spans[0].end==len(text)


def test_duplicate_missing_or_ambiguous_labels_fall_back():
    for text in [clause()+' '+clause(),clause()+' '+clause('Source 3'),clause(description='unclosed “ quote'), 'A dog barks beside a car.']:
        assert module.request_source_spans(text)==()


def test_features_do_not_depend_on_semantic_or_numeric_values():
    original=clause()+' '+clause('Source 2','car engine')
    edited=clause(description='cello playing',coordinate='(-150, 30, 4000)')+' '+clause('Source 2','rain')
    assert [(s.source_number,s.kind) for s in module.request_source_spans(original)]==[(s.source_number,s.kind) for s in module.request_source_spans(edited)]
