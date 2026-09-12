#!/usr/bin/env python3
"""Authored semantic-judge probes, independent of every AR data split/output."""
import argparse
import hashlib
import json
from pathlib import Path

# Each anchor has two meaning-preserving and two contradictory alternatives.
# Entire anchors stay in one fold. These are authored checks, not human labels.
ANCHORS = {
    'sound': [
        ('A dog barks repeatedly.', ['Repeated barking from a dog.', 'A dog lets out a series of barks.'], ['A cat meows repeatedly.', 'A dog laps water from a bowl.']),
        ('A kettle whistles as water boils.', ['The whistle of a boiling kettle.', 'A kettle gives a sustained boiling whistle.'], ['A train whistles as it approaches.', 'A kettle is set down with a clunk.']),
        ('A car door slams shut.', ['A car door closes with a slam.', 'Someone slams a vehicle door.'], ['A wooden cabinet door slams shut.', 'A car engine starts up.']),
        ('A glass bottle shatters on the floor.', ['A glass bottle breaks into pieces on the floor.', 'The crash of a glass bottle breaking on the ground.'], ['A ceramic plate shatters on the floor.', 'A glass bottle rolls across the floor.']),
        ('Footsteps crunch on loose gravel.', ['Someone walks over gravel with crunching steps.', 'Crunching footsteps across a gravel path.'], ['Footsteps tap on a wooden floor.', 'A car tire skids on gravel.']),
        ('Water drips slowly from a leaky tap.', ['Slow drops of water fall from a leaking faucet.', 'A dripping faucet releases water at a slow pace.'], ['Water gushes rapidly from a tap.', 'A clock ticks slowly.']),
        ('Rain patters against a metal roof.', ['Raindrops patter on metal roofing.', 'Rain hitting a roof made of metal.'], ['Rain patters against a glass window.', 'Hailstones smash through a metal roof.']),
        ('A bicycle bell rings twice.', ['Two rings from a bike bell.', 'A bell on a bicycle sounds two times.'], ['A church bell rings twice.', 'A bicycle chain rattles continuously.']),
    ],
    'music': [
        ('A solo cello plays long bowed notes.', ['A lone cellist bows sustained notes.', 'Long notes played with a bow on a single cello.'], ['A solo violin plays long bowed notes.', 'A solo cello plays short plucked notes.']),
        ('A muted trumpet plays short staccato phrases.', ['Brief detached phrases from a trumpet with a mute.', 'A muted trumpet performs clipped, short phrases.'], ['A muted trombone plays short staccato phrases.', 'An unmuted trumpet holds one long note.']),
        ('An acoustic guitar strums a steady chord rhythm.', ['Rhythmic chords strummed steadily on an acoustic guitar.', 'An acoustic guitarist strums chords at a regular pace.'], ['An electric bass plays a steady plucked rhythm.', 'An acoustic guitar plays isolated single-note harmonics.']),
        ('A grand piano plays an ascending scale.', ['A rising sequence of scale notes on a grand piano.', 'The notes of a scale climb upward on a grand piano.'], ['A flute plays an ascending scale.', 'A grand piano plays a descending scale.']),
        ('A flute plays a gentle flowing melody.', ['A soft, smoothly flowing tune on the flute.', 'Gentle melodic flute playing with connected notes.'], ['A clarinet plays a gentle flowing melody.', 'A flute produces sharp, harsh fluttering bursts.']),
        ('A hand drum plays a slow repeating beat.', ['A slow regular pattern played on a hand drum.', 'Repeated slow beats from a hand-played drum.'], ['A cymbal plays a slow repeating beat.', 'A hand drum plays a fast irregular roll.']),
        ('An electric bass plucks a low repeating riff.', ['A low repeated riff played by plucking an electric bass.', 'An electric bassist plucks the same low riff repeatedly.'], ['An acoustic guitar strums a high melody.', 'An electric bass sustains a single bowed note.']),
        ('A solo violin plays rapid tremolo.', ['A lone violinist plays with rapid repeated bowing.', 'Fast tremolo from a single violin.'], ['A solo cello plays rapid tremolo.', 'A solo violin plucks slow separate notes.']),
    ],
    'speech': [
        ('A young woman with a soft high-pitched voice.', ['A softly spoken young female voice in a high register.', 'A young adult female speaker with a gentle, high voice.'], ['A young man with a soft high-pitched voice.', 'An elderly woman with a rough low-pitched voice.']),
        ('An elderly man with a low raspy voice.', ['An older male speaker with a deep, gravelly voice.', 'An old man speaking in a low, hoarse register.'], ['An elderly woman with a low raspy voice.', 'A young man with a clear high-pitched voice.']),
        ('A young child with a high energetic voice.', ['A lively high-pitched voice of a small child.', 'An energetic young child speaking in a high register.'], ['An adult with a high energetic voice.', 'A young child with a quiet low-pitched voice.']),
        ('An adult woman with a low breathy voice.', ['A grown female speaker whose voice is deep and airy.', 'A woman speaking in a low register with a breathy tone.'], ['An adult man with a low breathy voice.', 'An adult woman with a piercing high-pitched voice.']),
        ('An adult man with a calm mid-pitched voice.', ['A grown male speaker with a relaxed voice in a middle register.', 'A man speaking calmly at a moderate pitch.'], ['An adult woman with a calm mid-pitched voice.', 'An adult man shouting in an agitated high-pitched voice.']),
        ('An elderly woman with a high quavering voice.', ['An older female speaker with a tremulous high voice.', 'An old woman speaking in a high, wavering register.'], ['An elderly man with a high quavering voice.', 'A young woman with a steady low-pitched voice.']),
        ('A teenage boy with a high nasal voice.', ['An adolescent male speaker with a high, nose-resonant voice.', 'A boy in his teens speaking with a high nasal tone.'], ['A teenage girl with a high nasal voice.', 'An elderly man with a low chesty voice.']),
        ('An adult woman with a firm mid-pitched voice.', ['A grown female speaker with an assertive voice in a middle register.', 'A woman speaking firmly at a moderate pitch.'], ['An adult man with a firm mid-pitched voice.', 'An adult woman whispering weakly in a very high register.']),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for kind, anchors in ANCHORS.items():
        for index, (reference, positives, negatives) in enumerate(anchors):
            for label, candidates in [('PASS', positives), ('FAIL', negatives)]:
                for variant, candidate in enumerate(candidates):
                    rows.append({'id': f'{kind}_{index}_{label.lower()}_{variant}', 'anchor': f'{kind}_{index}',
                                 'split': 'dev' if index % 2 == 0 else 'holdout', 'kind': kind,
                                 'reference': reference, 'candidate': candidate, 'label': label})
    assert len(rows) == 96 and len({r['id'] for r in rows}) == 96
    result = {'schema': 'generation_ar_semantic_authored_calibration_v1', 'rows': rows,
              'source': 'assistant-authored explicit synonym/contradiction examples; not independent human annotations',
              'ar_train_validation_test_used': False,
              'holdout_rule': 'whole odd-numbered anchors, 48 pairs; 24 positive and 24 negative',
              'prespecified_gate': {'holdout_accuracy_min': .95, 'holdout_false_accept_rate_max': .05, 'pass_probability_threshold': .5},
              'limitation': 'Passing these probes does not establish judge reliability on complex real AR outputs; a blinded candidate audit remains required.',
              'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'status': 'PREPARED', 'pairs': len(rows), 'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__': main()
