"""Timing-only ranking from profiling data; streaming index avoids loading raw traces."""
import argparse
import collections
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .contract import dump


def feature(observation):
    """Only the first response marker is a ranking feature."""
    if observation.get('ok') is not True:
        raise ValueError('Incomplete observations cannot shrink the denominator')
    value = observation['markers'][0]['elapsed_ns'] / 1e6
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid first-marker time')
    return value


def fit_profile(rows, candidates, sigma_floor):
    # Welford: measured latencies may have a large mean and a small variance.
    moments = {int(candidate): [0, 0., 0.] for candidate in candidates}
    for candidate, value in rows:
        state = moments[int(candidate)]
        state[0] += 1
        delta = value - state[1]
        state[1] += delta / state[0]
        state[2] += delta * (value - state[1])
    if any(state[0] < 2 for state in moments.values()):
        raise ValueError('Every public candidate needs at least two profiling samples')
    return {'candidate_ids': list(candidates),
            'mean_ms': [moments[int(c)][1] for c in candidates],
            'sigma_ms': [max(sigma_floor, math.sqrt(max(0., moments[int(c)][2]) /
                                                        (moments[int(c)][0] - 1))) for c in candidates],
            'samples': [moments[int(c)][0] for c in candidates],
            'fit_partition': 'profile only; validation and test excluded',
            'ranking': 'log Gaussian compatibility; no probability calibration'}


def rank(times_ms, profile):
    ids = np.asarray(profile['candidate_ids'], dtype=np.int64)
    if len(ids) != len(np.unique(ids)) or np.any(ids[:-1] >= ids[1:]):
        raise ValueError('Public candidate IDs must be unique and ascending for deterministic ties')
    mu = np.asarray(profile['mean_ms'], dtype=float)
    sigma = np.asarray(profile['sigma_ms'], dtype=float)
    times = np.asarray(times_ms, dtype=float)
    if (not np.isfinite(times).all() or not np.isfinite(mu).all()
            or not np.isfinite(sigma).all() or np.any(sigma <= 0)):
        raise ValueError('Invalid timings/profiles')
    # Log compatibility avoids underflow. No 1/sigma density normalization.
    scores = -0.5 * ((times[:, None] - mu[None, :]) / sigma[None, :]) ** 2
    k = min(5, len(ids))
    # Stable sort makes ascending public ID the tie break.
    indices = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    return ids[indices], np.take_along_axis(scores, indices, axis=1)


def uncertainty(rows, seed, resamples):
    groups = collections.defaultdict(list)
    blocks = collections.defaultdict(list)
    for row in rows:
        groups[(row['context_id'], row['repeat'])].append(row)
        blocks[row['repeat']].append(row)
    if len(blocks) < 2:
        raise ValueError('At least two independent repetition blocks required for this analysis')
    counts = np.array([[sum(r['correct'] for r in block), len(block)]
                       for block in blocks.values()], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(counts), size=(resamples, len(counts)))
    totals = counts[sampled].sum(axis=1)
    bootstrap = totals[:, 0] / totals[:, 1]
    null = np.zeros(resamples)
    for group in groups.values():
        truth = np.array([r['truth'] for r in group], dtype=np.int64)
        predictions = np.array([r['prediction'] for r in group], dtype=np.int64)
        for draw in range(resamples):
            null[draw] += (rng.permutation(truth) == predictions).sum()
    null /= len(rows)
    observed = sum(r['correct'] for r in rows) / len(rows)
    return {'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'one_sided_permutation_p': float((1 + np.sum(null >= observed)) / (resamples + 1)),
            'bootstrap_unit': 'repetition block; fixed candidates, contexts, device and sessions',
            'permutation_unit': 'labels within context x repetition; fixed profile and decisions',
            'blocks': len(blocks), 'resamples': resamples}


def paired_interval(first, second, seed, resamples):
    key = lambda row: (row['context_id'], row['repeat'], row['truth'])
    other = {key(row): row for row in second}
    if len(other) != len(second) or set(other) != {key(row) for row in first}:
        raise ValueError('Conditions are not matched by context/repetition/candidate')
    blocks = collections.defaultdict(list)
    for row in first:
        blocks[row['repeat']].append(int(row['correct']) - int(other[key(row)]['correct']))
    values = np.array([np.mean(block) for block in blocks.values()])
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return {'difference': float(values.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'unit': 'paired repetition blocks, fixed contexts and candidates'}


def evaluate(output):
    if (output / 'summary.json').exists() or (output / 'evaluation_index.sqlite').exists():
        raise ValueError('Evaluation artifacts exist; refusing overwrite')
    cfg = json.loads((output / 'config.json').read_text())
    status = json.loads((output / 'status.json').read_text())
    planned = json.loads((output / 'plan.json').read_text())
    if status['state'] != 'collection_complete':
        raise ValueError('Collection incomplete; preserve failures without reducing denominator')
    db = sqlite3.connect(output / 'evaluation_index.sqlite')
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE truth (id TEXT PRIMARY KEY, phase TEXT, cond TEXT, rep INTEGER, ctx TEXT, cls INTEGER)')
    db.execute('CREATE TABLE seen (id TEXT PRIMARY KEY REFERENCES truth(id), time REAL NOT NULL)')
    with db:
        with (output / 'ground_truth.jsonl').open() as stream:
            for line in stream:
                r = json.loads(line)
                db.execute('INSERT INTO truth VALUES (?,?,?,?,?,?)',
                           (r['request_id'], r['phase'], r['condition'], r['repeat'], r['context_id'], r['class_id']))
        with (output / 'observations.jsonl').open() as stream:
            for line in stream:
                r = json.loads(line)
                if {'input_ids', 'class_id', 'generated_token_ids', 'host_visible_forward_and_argmax_ns'} & set(r):
                    raise ValueError('Privileged fields in the observer artifact')
                db.execute('INSERT INTO seen VALUES (?,?)', (r['request_id'], feature(r)))
    ntruth = db.execute('SELECT count(*) FROM truth').fetchone()[0]
    nseen = db.execute('SELECT count(*) FROM seen').fetchone()[0]
    if nseen != ntruth or nseen != planned['evaluation_requests']:
        raise ValueError('Missing/extra observations or labels; denominator is not complete')
    summary = {'protocol_id': cfg['protocol_id'], 'role': cfg['role'], 'task': cfg['task'],
               'observations': nseen, 'primary_feature': 'first_response_marker_ms',
               'access': cfg['observation']['access'], 'physical_cache_measured': False,
               'complete_prompt_reconstruction_evaluated': False,
               'historical_73_4_reproduced': False, 'pearson': 'not computed',
               'defense_overhead': 'not measured', 'conditions': {}}
    if cfg['task'] == 'corpus_trace':
        values = np.array([r[0] for r in db.execute('SELECT time FROM seen')])
        summary.update(target='raw timing of full inputs; no reconstruction predictions',
                       first_marker_ms={'mean': float(values.mean()),
                                        'quantiles_05_50_95': np.quantile(values, [.05, .5, .95]).tolist()})
        dump(output / 'summary.json', summary)
        db.close()
        return summary
    design = json.loads((output / 'design.json').read_text())
    candidates = design['candidate_ids']
    primary = {}
    with (output / 'predictions.jsonl').open('x') as predictions:
        for condition in cfg['design']['conditions']:
            query = ('SELECT cls,time FROM truth JOIN seen USING(id) '
                     'WHERE cond=? AND phase=?')
            profile = fit_profile(db.execute(query, (condition, 'profile')), candidates,
                                  cfg['analysis']['sigma_floor_ms'])
            dump(output / (condition + '_profile.json'), profile)
            summary['conditions'][condition] = {}
            for phase in ['validation', 'new_session_new_context']:
                cursor = db.execute('SELECT id,cls,ctx,rep,time FROM truth JOIN seen USING(id) '
                                    'WHERE cond=? AND phase=?', (condition, phase))
                rows = []
                while batch := cursor.fetchmany(128):
                    top, scores = rank([r[4] for r in batch], profile)
                    for i, (request_id, truth, context, repeat, unused_time) in enumerate(batch):
                        predictions.write(json.dumps({'request_id': request_id, 'condition': condition,
                            'phase': phase, 'topk': top[i].tolist(), 'log_compatibility': scores[i].tolist()}) + '\n')
                        rows.append({'truth': truth, 'prediction': int(top[i, 0]),
                                     'correct': bool(top[i, 0] == truth), 'top5': bool(truth in top[i]),
                                     'context_id': context, 'repeat': repeat})
                expected = next(p['requests_per_condition'] for p in planned['phases'] if p['name'] == phase)
                if len(rows) != expected:
                    raise ValueError('Phase denominator differs')
                metrics = {'requests': len(rows), 'correct_top1': sum(r['correct'] for r in rows),
                           'top1': sum(r['correct'] for r in rows) / len(rows),
                           'top5': sum(r['top5'] for r in rows) / len(rows),
                           'candidate_count': len(candidates), 'uniform_top1': 1 / len(candidates),
                           'uniform_top5': min(5, len(candidates)) / len(candidates)}
                if phase == 'new_session_new_context':
                    primary[condition] = rows
                    if cfg['role'] != 'technical_smoke':
                        metrics.update(uncertainty(rows, cfg['analysis']['seed'], cfg['analysis']['resamples']))
                    else:
                        metrics['inference'] = 'not computed; technical smoke is not scientific confirmation'
                summary['conditions'][condition][phase] = metrics
    if cfg['role'] != 'technical_smoke':
        summary['paired_primary'] = {
            control: paired_interval(primary['model'], primary[control],
                                     cfg['analysis']['seed'], cfg['analysis']['resamples'])
            for control in ['input_blinded', 'constant']}
    summary['interpretation'] = (
        'Technical execution only; no leakage claim.' if cfg['role'] == 'technical_smoke' else
        'Conditional estimates for one public input slot. Inspect controls and replicate independently; '
        'no automatic declaration of attack validation, passive access, or prompt reconstruction.')
    dump(output / 'summary.json', summary)
    db.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output.resolve()), indent=2), flush=True)


if __name__ == '__main__':
    main()
