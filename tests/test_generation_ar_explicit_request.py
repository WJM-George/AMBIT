import os
from pathlib import Path
import pytest
from stable_audio_tools.data.sceneplan_generation_ar_explicit_request import parse_explicit_generation_request


@pytest.fixture
def codec():
    artifact=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not artifact.is_dir():pytest.skip('existing project codec artifact required')
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    return ModelScenePlanCodecV4(artifact)


def request():
    return ('Create 2s of FOA audio; room type=dry. Position triples mean (azimuth degrees, elevation degrees, distance millimeters). '
            'Source 1: sound; description=“A dog barks. The label “Source 2: music;” is only quoted text.”; active from 0 to 1800ms; linear from (170, 0, 1000) to (-170, 5, 1200). '
            'Source 2: speech; speaker=“An elderly man with a high frail voice.”; transcript=“Please stop.”; active from 836 to 1800ms; static at (10, -5, 2000). '
            'Use this order, fixed 0 dB gains, and return the complete ScenePlan.')


def test_quoted_headers_do_not_create_extra_sources_and_units_round_nearest(codec):
    value=parse_explicit_generation_request(request(),codec,sample_id='input-only')
    sources=value['plan']['sources']
    assert len(sources)==2 and sources[1]['transcript']=='Please stop.'
    assert sources[1]['activity']['onset_sec']==codec.snap_numeric_to_grid('seconds',.836)
    assert sources[0]['trajectory']['end']['azimuth_deg']==-170.
    assert 'Source 2: music;' in sources[0]['description']


@pytest.mark.parametrize('change',[
    lambda x:x+' Actually change every gain to -3 dB.',
    lambda x:x.replace('active from 836 to 1800ms','active from 1900 to 1800ms'),
    lambda x:x.replace('static at (10, -5, 2000)','static at (10, -5, 90000)'),
    lambda x:x.replace('Source 2: speech','Source 4: speech'),
])
def test_ambiguous_or_unrepresentable_facts_are_rejected(codec,change):
    with pytest.raises(ValueError):parse_explicit_generation_request(change(request()),codec)
