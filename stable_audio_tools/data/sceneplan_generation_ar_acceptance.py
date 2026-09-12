"""Request fidelity acceptance v1 for the established static/linear AR scope.

Semantic labels must come from a separately validated, frozen evaluator and
must be bound to the exact source pair. Missing labels remain explicitly
pending; they can never establish acceptance.
"""
from __future__ import annotations
import math
import numpy as np
from stable_audio_tools.data.model_sceneplan import validate_model_sceneplan

TOLERANCE = {'onset_s': .25, 'offset_s': .25, 'azimuth_deg': 10., 'elevation_deg': 5., 'distance_m': .25}
THRESHOLDS = {'valid': 1., 'count': .95, 'semantic_precision': .90, 'semantic_recall': .90,
              'motion': .95, 'onset': .95, 'offset': .95, 'start': .90, 'end': .90, 'joint': .80}


def endpoints(source):
    trajectory = source['trajectory']
    if trajectory['type'] == 'static': return trajectory['position'], trajectory['position']
    if trajectory['type'] == 'linear': return trajectory['start'], trajectory['end']
    raise ValueError('acceptance v1 implementation supports only established static/linear Generation AR scope')


def angular_delta(start, end):
    # Same shortest-angle convention as P10 _shortest_angle_radians.
    return (end-start+180.) % 360. - 180.


def motion_direction(reference, candidate):
    if reference['trajectory']['type'] != candidate['trajectory']['type']: return False, {}
    if reference['trajectory']['type'] == 'static': return True, {}
    ref_start, ref_end = endpoints(reference); hyp_start, hyp_end = endpoints(candidate)
    checks = {}
    for key in ('azimuth_deg', 'elevation_deg', 'distance_m'):
        delta = angular_delta(ref_start[key], ref_end[key]) if key == 'azimuth_deg' else ref_end[key]-ref_start[key]
        predicted = angular_delta(hyp_start[key], hyp_end[key]) if key == 'azimuth_deg' else hyp_end[key]-hyp_start[key]
        # Endpoint tolerance can reverse a tiny movement's apparent sign.
        # Score a direction only when it exceeds both endpoints' uncertainty.
        if abs(delta) > 2*TOLERANCE[key]: checks[key] = delta*predicted > 0.
    return all(checks.values()), checks


def score_scene(target, prediction, semantic_labels=None):
    target = validate_model_sceneplan(target)
    labels = semantic_labels or {}
    for ref in target['sources']: endpoints(ref)
    parse_error = None
    if prediction is not None:
        try:
            prediction = validate_model_sceneplan(prediction)
            for hyp in prediction['sources']: endpoints(hyp)
        except (ValueError, KeyError, TypeError) as exc:
            parse_error = f'{type(exc).__name__}: {exc}'
            prediction = None
    refs = target['sources']; preds = prediction['sources'] if prediction is not None else []
    by_id = {p['source_id']:p for p in preds}
    items = []
    for ref in refs:
        hyp = by_id.get(ref['source_id'])
        row = {'source_id':ref['source_id'], 'matched':hyp is not None, 'kind':False,
               'semantic':False if hyp is None else labels.get(ref['source_id']),
               'motion':False,'onset':False,'offset':False,'start':False,'end':False,'errors':{},'direction_checks':{}}
        if row['semantic'] is not None and not isinstance(row['semantic'], bool): raise TypeError('semantic labels must be bool or None')
        if hyp is not None:
            row['kind'] = ref['kind'] == hyp['kind']
            if not row['kind']: row['semantic'] = False
            row['motion'], row['direction_checks'] = motion_direction(ref, hyp)
            for edge in ('onset','offset'):
                error = abs(ref['activity'][edge+'_sec']-hyp['activity'][edge+'_sec'])
                row['errors'][edge+'_s'] = error
                row[edge] = error <= TOLERANCE[edge+'_s']
            for edge, left, right in zip(('start','end'), endpoints(ref), endpoints(hyp)):
                good = True
                for key in ('azimuth_deg','elevation_deg','distance_m'):
                    error = abs(angular_delta(left[key],right[key])) if key=='azimuth_deg' else abs(left[key]-right[key])
                    row['errors'][edge+'_'+key] = error
                    good = good and error <= TOLERANCE[key]
                row[edge] = good
        row['joint'] = all(row[key] is True for key in ('kind','semantic','motion','onset','offset','start','end'))
        items.append(row)
    count = prediction is not None and len(refs) == len(preds)
    return {'valid':prediction is not None,'count':count,'requested':len(refs),'predicted':len(preds),
            'missing':sum(not r['matched'] for r in items),'extra':len(set(by_id)-{r['source_id'] for r in refs}),
            'semantic_pending':sum(r['semantic'] is None for r in items),'sources':items,
            'joint':count and all(r['joint'] for r in items),'parse_error':parse_error}


def summarize_scenes(records, *, bootstrap_replicates=1000):
    if not records: raise ValueError('empty acceptance records')
    numerator = []; denominator = []
    keys = list(THRESHOLDS)
    for record in records:
        sources = record['sources']; correct_semantics = sum(r['semantic'] is True for r in sources)
        num = {'valid':int(record['valid']),'count':int(record['count']),'joint':int(record['joint']),
               'semantic_precision':correct_semantics,'semantic_recall':correct_semantics,
               **{k:sum(r[k] for r in sources) for k in ('motion','onset','offset','start','end')}}
        den = {k:1 if k in ('valid','count','joint') else record['predicted'] if k=='semantic_precision' else record['requested'] for k in keys}
        numerator.append([num[k] for k in keys]); denominator.append([den[k] for k in keys])
    numerator = np.asarray(numerator,dtype=np.float64); denominator = np.asarray(denominator,dtype=np.float64)
    ns = numerator.sum(0); ds = denominator.sum(0)
    rng = np.random.default_rng(42); samples = []
    for start in range(0, bootstrap_replicates, 50):
        indices = rng.integers(0,len(records),size=(min(50,bootstrap_replicates-start),len(records)))
        n = numerator[indices].sum(1); d = denominator[indices].sum(1)
        samples.append(np.divide(n,d,out=np.zeros_like(n),where=d>0))
    intervals = np.quantile(np.concatenate(samples), [.025,.975], axis=0) if samples else None
    pending = sum(r['semantic_pending'] for r in records)
    metrics = {key:{'successes':int(ns[i]),'denominator':int(ds[i]),'rate':float(ns[i]/ds[i]) if ds[i] else None,
                   'threshold':THRESHOLDS[key], 'ci95_scene_cluster_bootstrap':[float(x) for x in intervals[:,i]] if intervals is not None else None,
                   'complete':pending==0 if key in ('semantic_precision','semantic_recall','joint') else True} for i,key in enumerate(keys)}
    errors = {}
    for key in sorted({k for r in records for s in r['sources'] for k in s['errors']}):
        values = [s['errors'][key] for r in records for s in r['sources'] if s['matched']]
        assert all(math.isfinite(v) for v in values)
        errors[key] = {'matched_source_mae':float(np.mean(values)),'matched_source_p50':float(np.quantile(values,.5)),
                       'matched_source_p90':float(np.quantile(values,.9)),'matched_denominator':len(values)}
    return {'scenes':len(records),'requested_sources':sum(r['requested'] for r in records),
            'predicted_sources':sum(r['predicted'] for r in records),'missing_sources':sum(r['missing'] for r in records),
            'extra_sources':sum(r['extra'] for r in records),'semantic_pending':pending,'metrics':metrics,'field_errors':errors,
            'acceptance_pass':all(m['complete'] and m['rate'] is not None and m['rate']>=m['threshold'] for m in metrics.values()),
            'confidence_method':f'percentile bootstrap by whole scene, seed 42, {bootstrap_replicates} replicates; source dependencies within a scene retained',
            'limits':'Call separately for each 1–4-source group; labels require separately frozen and validated semantic evaluation. No publication claim follows from a diagnostic panel.'}
