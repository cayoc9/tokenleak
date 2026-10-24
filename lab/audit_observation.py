"""Audit measured files without importing the collector's feature extractor."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def require(value, message):
    if not value:
        raise ValueError(message)


def vector(row):
    times = [m['elapsed_ns']/1e6 for m in row['markers']]
    gaps = np.diff(times)
    return np.array([times[0], times[-1], gaps.mean(), gaps.std(),
                     sum(r['body_bytes'] > 0 for r in row['reads']),
                     max(Counter(m['read_id'] for m in row['markers']).values())])


def audit(directory):
    cfg = json.loads((directory/'config.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, digest in manifest.items():
        require(hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, 'Artifact hash changed: '+name)
    truth = [json.loads(line) for line in (directory/'ground_truth.jsonl').read_text().splitlines()]
    by_id = {r['request_id']: r for r in truth}
    design = json.loads((directory/'design.json').read_text())
    classes = np.array(sorted(r['class_id'] for r in design['candidates']))
    blocks = Counter((r['session'], r['repeat'], r['context_id'], r['class_id']) for r in truth)
    require(len(blocks) == cfg['sessions']*cfg['repetitions']*4*8 and set(blocks.values()) == {1}, 'Design is not balanced')
    for context in ['X00', 'X01', 'X02', 'X03']:
        public = {tuple(r['input_ids'][:cfg['secret_position']] + r['input_ids'][cfg['secret_position']+1:])
                  for r in truth if r['context_id'] == context}
        require(len(public) == 1, 'More than the secret slot varies within a context')
    observed = [json.loads(line) for line in (directory/'observations.jsonl').read_text().splitlines()]
    require(len(observed) == len(truth)*len(cfg['conditions']), 'Incomplete observation count')
    lookup, diagnostics = {}, {}
    for condition in cfg['conditions']:
        rows = [r for r in observed if r['condition'] == condition]
        require({r['request_id'] for r in rows} == set(by_id) and len(rows)==len(truth), 'Condition coverage differs')
        for row in rows:
            require(set(row) == {'reads','markers','body_bytes','body_reads','max_markers_per_read','ok',
                                'request_body_bytes','request_id','session','condition','visibility'}, 'Observer contains extra fields')
            require(row['ok'] and len(row['markers']) == cfg['output_steps'], 'Missing output markers')
            require(row['body_bytes'] == cfg['output_steps']*64 and row['request_body_bytes']==cfg['request_body_bytes'], 'Payload size contract changed')
            require(row['session'] == by_id[row['request_id']]['session'], 'Session join mismatch')
            for i, marker in enumerate(row['markers']):
                require(marker['index']==i and marker['elapsed_ns']==row['reads'][marker['read_id']]['elapsed_ns'], 'Chunk/marker lineage failed')
            require(np.all(np.diff([r['elapsed_ns'] for r in row['reads']]) >= 0), 'Nonmonotonic client clock')
            lookup[(condition, row['request_id'])] = row
        model = json.loads((directory/f'model_{condition}.json').read_text())
        training = [r for r in rows if by_id[r['request_id']]['session']==0 and by_id[r['request_id']]['context_id'] in ['X00','X01']]
        x = np.stack([vector(r) for r in training])
        y = np.array([by_id[r['request_id']]['class_id'] for r in training])
        scaler = StandardScaler().fit(x)
        fitted = LogisticRegression(C=1., max_iter=2000, class_weight='balanced', solver='lbfgs', multi_class='multinomial').fit(scaler.transform(x), y)
        require(np.allclose(scaler.mean_, model['scaler_mean'], rtol=0, atol=1e-12), 'Scaler used a different partition')
        require(np.allclose(fitted.coef_, model['coefficients'], rtol=0, atol=1e-10), 'Weight fit differs from training-only replay')
        require(np.allclose(fitted.intercept_, model['intercept'], rtol=0, atol=1e-10), 'Intercept differs from replay')
        diagnostics[condition] = {'requests':len(rows),'train_requests':len(training),
            'requests_with_coalesced_markers':sum(r['max_markers_per_read'] > 1 for r in rows),
            'median_first_marker_ms':float(np.median([r['markers'][0]['elapsed_ns'] for r in rows]))/1e6,
            'median_last_marker_ms':float(np.median([r['markers'][-1]['elapsed_ns'] for r in rows]))/1e6,
            'max_markers_per_socket_read':max(r['max_markers_per_read'] for r in rows)}
    predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    recounted = defaultdict(list)
    for row in predictions:
        obs = lookup[(row['condition'], row['request_id'])]
        require(np.allclose(vector(obs), row['features'], rtol=0, atol=1e-12), 'External features differ')
        require(row['true_class'] == by_id[row['request_id']]['class_id'], 'Truth join failed')
        model = json.loads((directory/f'model_{row["condition"]}.json').read_text())
        x = (vector(obs)-model['scaler_mean'])/model['scaler_scale']
        scores = np.array(model['coefficients']) @ x + model['intercept']
        probabilities = np.exp(scores-scores.max()); probabilities /= probabilities.sum()
        require(np.allclose(probabilities, row['probabilities'], rtol=0, atol=1e-12), 'Probabilities differ from coefficient replay')
        require(model['classes'][int(scores.argmax())] == row['predicted_class'], 'Prediction differs')
        recounted[(row['condition'], row['partition'])].append(row)
    summary = json.loads((directory/'summary.json').read_text())
    for (condition, partition), rows in recounted.items():
        item = summary['conditions'][condition][partition]
        correct = sum(r['true_class']==r['predicted_class'] for r in rows)
        require(correct==item['correct_top1'] and len(rows)==item['requests'], 'Numerator/denominator mismatch')
        support = Counter(r['true_class'] for r in rows)
        recovered = Counter(r['true_class'] for r in rows if r['true_class']==r['predicted_class'])
        require(abs(np.mean([recovered[c]/support[c] for c in classes])-item['balanced_accuracy']) < 1e-12, 'Balanced accuracy mismatch')
    return {'passed':True,'observation_rows_checked':len(observed),'prediction_rows_checked':len(predictions),
            'training_only_refits_checked':len(cfg['conditions']),'fixed_payload_sizes_checked':True,
            'only_secret_slot_varies_within_context':True,'chunk_marker_lineage_checked':True,
            'features_recomputed_without_importing_evaluator':True,'conditions':diagnostics,
            'scope':'Software/data audit of a controlled inference pilot; not proof of a deployable side-channel attack'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():
        raise SystemExit('Report exists; choose a new name')
    report=audit(args.run)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
