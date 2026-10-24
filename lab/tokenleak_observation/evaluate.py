"""Fit on known requests; classify a single input token from response timing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['request_to_first_marker_ms', 'request_to_last_marker_ms', 'mean_marker_gap_ms',
            'std_marker_gap_ms', 'body_reads', 'max_markers_per_read']


def features(observation):
    """Consumes only the observer view; no truth, context or server trace argument."""
    if not observation['ok']:
        raise ValueError('Failed request must remain visible; incomplete runs are not evaluated')
    times = np.array([m['elapsed_ns'] for m in observation['markers']], dtype=float) / 1e6
    if len(times) < 2 or (np.diff(times) < 0).any():
        raise ValueError('Invalid monotonic marker sequence')
    return np.array([times[0], times[-1], np.diff(times).mean(), np.diff(times).std(),
                     observation['body_reads'], observation['max_markers_per_read']], dtype=float)


def interval_and_null(truth, predicted, rows, seed, repeats):
    hit = predicted == truth
    groups = sorted({r['repeat'] for r in rows})
    per_group = np.array([hit[[r['repeat'] == g for r in rows]].mean() for g in groups])
    rng = np.random.default_rng(seed)
    bootstrap = per_group[rng.integers(0, len(groups), (repeats, len(groups)))].mean(axis=1)
    blocks = []
    for context, repetition in sorted({(r['context_id'], r['repeat']) for r in rows}):
        blocks.append(np.array([i for i, r in enumerate(rows) if r['context_id'] == context and r['repeat'] == repetition]))
    null = []
    for _ in range(repeats):
        shuffled = truth.copy()
        for indices in blocks:
            shuffled[indices] = rng.permutation(truth[indices])
        null.append(float((predicted == shuffled).mean()))
    return {'cluster_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'cluster_unit': 'balanced repetition block; fixed candidates, contexts, machine and sessions',
            'bootstrap_groups': len(groups), 'randomizations': repeats,
            'conditional_permutation_p_one_sided': (1 + sum(x >= hit.mean() for x in null)) / (1 + repeats),
            'permutation_null_mean': float(np.mean(null)),
            'permutation_null_ci95': np.quantile(null, [.025, .975]).tolist(),
            'permutation_unit': 'labels shuffled within context x repetition, classifier fixed'}


def metrics(truth, probabilities, classes):
    rank = np.argsort(-probabilities, axis=1, kind='stable')
    predictions = classes[rank[:, 0]]
    return {'requests': len(truth), 'correct_top1': int((predictions == truth).sum()),
            'top1': float((predictions == truth).mean()),
            'top5': float((classes[rank[:, :5]] == truth[:, None]).any(axis=1).mean()),
            'balanced_accuracy': float(balanced_accuracy_score(truth, predictions)),
            'macro_f1': float(f1_score(truth, predictions, labels=classes, average='macro', zero_division=0)),
            'log_loss': float(log_loss(truth, probabilities, labels=classes)),
            'support': dict(Counter(truth.tolist()))}, predictions


def evaluate(output):
    cfg = json.loads((output/'config.json').read_text())
    truth_rows = [json.loads(x) for x in (output/'ground_truth.jsonl').read_text().splitlines()]
    observed = [json.loads(x) for x in (output/'observations.jsonl').read_text().splitlines()]
    truth_map = {r['request_id']: r for r in truth_rows}
    expected = len(truth_rows) * len(cfg['conditions'])
    if len(observed) != expected or len({(r['condition'], r['request_id']) for r in observed}) != expected:
        raise ValueError('Missing or duplicate observations; no requests dropped from denominator')
    if any('input_ids' in r or 'class_id' in r or 'model_forward_ns' in r for r in observed):
        raise ValueError('Ground truth or privileged trace leaked into observer view')
    partitions = {'validation_known_context': lambda r: r['session'] == 1 and r['context_id'] in ['X00', 'X01'],
                  'same_session_new_context': lambda r: r['session'] == 0 and r['context_id'] in ['X02', 'X03'],
                  'new_session_known_context': lambda r: r['session'] == 2 and r['context_id'] in ['X00', 'X01'],
                  'new_session_new_context': lambda r: r['session'] == 2 and r['context_id'] in ['X02', 'X03']}
    conditions, saved_predictions = {}, []
    for condition in cfg['conditions']:
        observations = [r for r in observed if r['condition'] == condition]
        rows = [truth_map[r['request_id']] for r in observations]
        x = np.stack([features(r) for r in observations])
        y = np.array([r['class_id'] for r in rows])
        training = np.array([r['session'] == 0 and r['context_id'] in ['X00', 'X01'] for r in rows])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000,
                              class_weight='balanced', solver='lbfgs', multi_class='multinomial'))
        model.fit(x[training], y[training])
        classes = model.classes_
        counts = Counter(y[training].tolist())
        prior = np.array([counts[c] for c in classes], dtype=float); prior /= prior.sum()
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        model_data = {'features': FEATURES, 'classes': classes.tolist(), 'training_requests': int(training.sum()),
                      'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                      'coefficients': classifier.coef_.tolist(), 'intercept': classifier.intercept_.tolist(),
                      'hyperparameters_selected_from_test': False,
                      'fit_partition': 'session 0, contexts X00/X01 only'}
        (output/f'model_{condition}.json').write_text(json.dumps(model_data, indent=2)+'\n')
        condition_metrics = {}
        for partition, predicate in partitions.items():
            mask = np.array([predicate(r) for r in rows])
            selected_rows = [r for r, selected in zip(rows, mask) if selected]
            target = y[mask]
            probabilities = model.predict_proba(x[mask])
            result, prediction = metrics(target, probabilities, classes)
            result.update(interval_and_null(target, prediction, selected_rows, cfg['analysis_seed'], cfg['resamples']))
            _, prior_prediction = metrics(target, np.tile(prior, (len(target), 1)), classes)
            result['frequency_prior_top1'] = float((prior_prediction == target).mean())
            result['uniform_top1_expectation'] = 1 / len(classes)
            result['uniform_top5_expectation'] = min(5, len(classes)) / len(classes)
            result['scope'] = 'one input-token identity among eight public candidates; fixed public slot; response timing only'
            condition_metrics[partition] = result
            for observation, row, vector, prob, pred in zip([r for r, ok in zip(observations, mask) if ok], selected_rows, x[mask], probabilities, prediction):
                saved_predictions.append({'condition': condition, 'partition': partition,
                    'request_id': row['request_id'], 'true_class': row['class_id'], 'predicted_class': str(pred),
                    'features': vector.tolist(), 'probabilities': prob.tolist(),
                    'feature_source': 'client observer only'})
        conditions[condition] = condition_metrics
    with (output/'predictions.jsonl').open('w') as f:
        for row in saved_predictions:
            f.write(json.dumps(row) + '\n')
    # Internal model timings remain diagnostics, not features available to this observer.
    diagnostics = {}
    for condition in cfg['conditions']:
        traces = [json.loads(line) for p in output.glob(f'session_*_{condition}_server_trace.jsonl')
                  for line in p.read_text().splitlines()]
        evaluation = [r for r in traces if r['purpose'] == 'evaluation']
        if len(evaluation) != len(truth_rows):
            raise ValueError('Privileged diagnostic trace coverage differs from observations')
        vectors = [r['model_forward_ns'] for r in evaluation if r['model_forward_ns']]
        diagnostics[condition] = {'evaluation_requests': len(evaluation), 'warmup_requests': len(traces)-len(evaluation),
                                 'privileged_not_used_as_features': True}
        if vectors:
            diagnostics[condition].update(prefill_and_argmax_ms_median=float(np.median([t[0] for t in vectors]))/1e6,
                decode_and_argmax_ms_median=float(np.median([v for t in vectors for v in t[1:]]))/1e6)
    summary = {'mode': 'measured_model_inference_loopback', 'model_id': cfg['model_id'], 'revision': cfg['model_revision'],
               'physical_cache_measured': False, 'complete_prompt_reconstruction_evaluated': False,
               'primary_partition': 'new_session_new_context', 'primary_condition': 'gpt2',
               'observations': len(observed), 'features': FEATURES, 'conditions': conditions,
               'privileged_diagnostics': diagnostics,
               'limits': ['GPT-2 small CPU feasibility pilot, not GPT-2 XL or the historical SPC-5K experiment.',
                          'Laboratory emits fixed-size token markers; not a production streaming endpoint.',
                          'Client timing excludes connection setup; includes request send, service and response transport.',
                          'Small fixed candidate set and public secret slot; no general prompt-position recovery.',
                          'Conditional block uncertainty; no inference about other machines or real private fields.']}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    verify(output)
    return summary


def verify(output):
    cfg = json.loads((output/'config.json').read_text())
    design = json.loads((output/'design.json').read_text())
    rows = [json.loads(x) for x in (output/'ground_truth.jsonl').read_text().splitlines()]
    lookup = {r['request_id']: r for r in rows}
    by_class = {r['class_id']: r['token_id'] for r in design['candidates']}
    if any(len(r['input_ids']) != cfg['input_length'] or r['input_ids'][cfg['secret_position']] != by_class[r['class_id']] for r in rows):
        raise ValueError('Secret identity, slot or input length differs from contract')
    observed = {(r['condition'], r['request_id']): r for r in
                [json.loads(x) for x in (output/'observations.jsonl').read_text().splitlines()]}
    checked = 0
    for line in (output/'predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        model = json.loads((output/f'model_{row["condition"]}.json').read_text())
        raw = observed[(row['condition'], row['request_id'])]
        vector = features(raw)
        if not np.array_equal(vector, row['features']):
            raise ValueError('Feature lineage mismatch')
        standardized = (vector-np.array(model['scaler_mean']))/np.array(model['scaler_scale'])
        logits = np.array(model['coefficients']) @ standardized + model['intercept']
        prediction = model['classes'][int(np.argmax(logits))]
        if prediction != row['predicted_class'] or row['true_class'] != lookup[row['request_id']]['class_id']:
            raise ValueError('Independent linear prediction or label join mismatch')
        checked += 1
    verification = {'passed': True, 'predictions_checked': checked,
                    'features_recomputed_from_observer': True, 'linear_decisions_recomputed_without_predict': True,
                    'labels_and_slot_checked': True, 'privileged_trace_excluded_from_features': True,
                    'observations': len(observed), 'ground_truth_requests': len(rows),
                    'physical_cache_validation': False}
    (output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file() and p.name not in ['manifest.json']}
    (output/'manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    return verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = evaluate(args.output.resolve())
    print(json.dumps({'evaluated': True, 'primary': summary['conditions']['gpt2']['new_session_new_context']}), flush=True)


if __name__ == '__main__':
    main()
