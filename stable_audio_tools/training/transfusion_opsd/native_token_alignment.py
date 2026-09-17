"""Validate native plans without replacing the student's visited token prefix.

Text segmentation is not unique: encode(decode(ids)) need not equal ids.
The Editing pipeline also aligns the decoded final frame to source duration.
Neither operation authorizes training on a re-encoded, unvisited prefix.
"""


def validate_native_plan_tokens(codec, tokens, plan):
    """Return canonical ids for comparison only, retaining original positions.

    Matching on the codec's executable grid preserves the previous contract
    for unquantized caller plans. The fallback still checks every field; it
    does not accept a different plan merely because the source count matches.
    """
    ids = list(map(int, tokens))
    canonical = codec.encode(plan)['input_ids'].tolist()
    if ids == canonical:
        return canonical
    decoded = codec.decode(ids, sample_id=plan['sample_id'])
    if codec.encode(decoded)['input_ids'].tolist() == canonical:
        return canonical
    from ...models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
    aligned = _align_decoded_sceneplan_to_audio_duration(decoded, plan['duration_sec'])
    if codec.encode(aligned)['input_ids'].tolist() != canonical:
        raise ValueError('Native tokens and plan describe different executable fields.')
    return canonical


def structured_positions(codec, tokens):
    """Locate the non-text skeleton without changing any native token offsets."""
    begin, end = codec._tid('<text_begin>'), codec._tid('<text_end>')
    positions, inside = [], False
    for position, token in enumerate(tokens):
        if token == end:
            inside = False
        if not inside:
            positions.append(position)
        if token == begin:
            inside = True
    return positions


def single_native_structured_edit(codec, tokens, canonical, candidate):
    """Map a single canonical field edit onto the actual visited sequence.

    Text differences are never proposed here. When boundary alignment changes
    the structure itself, this narrow one-choice proposal is unavailable.
    """
    ids = list(map(int, tokens))
    if len(canonical) != len(candidate):
        return None
    changed = [i for i, (a, b) in enumerate(zip(canonical, candidate)) if a != b]
    if len(changed) != 1:
        return None
    native_sites = structured_positions(codec, ids)
    canonical_sites = structured_positions(codec, canonical)
    if ([ids[i] for i in native_sites] != [canonical[i] for i in canonical_sites]
            or changed[0] not in canonical_sites):
        return None
    position = native_sites[canonical_sites.index(changed[0])]
    edited = ids.copy()
    edited[position] = candidate[changed[0]]
    return position, edited
