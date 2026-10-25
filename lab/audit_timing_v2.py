"""Independent stdlib audit for v2 smokes, eight-candidate pilots and corpus traces."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import permutations
import json
import math
import random
from pathlib import Path
import statistics
import uuid


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def convolve(first, second):
    result = Counter()
    for a, wa in first.items():
        for b, wb in second.items():
            result[a + b] += wa * wb
    return result


def discrete_interval(distribution, denominator):
    total = sum(distribution.values())
    targets = [math.ceil(total * .025), math.ceil(total * .975)]
    answer, cumulative = [], 0
    for value, count in sorted(distribution.items()):
        cumulative += count
        while len(answer) < 2 and cumulative >= targets[len(answer)]:
            answer.append(value / denominator)
    return answer


def exact_block_interval(block_counts, requests_per_block):
    base = Counter(block_counts)
    distribution = Counter({0: 1})
    for unused in block_counts:
        distribution = convolve(distribution, base)
    return discrete_interval(distribution, len(block_counts) * requests_per_block)


@lru_cache(maxsize=64)
def null_histogram(pattern, k):
    """Labels occur once; permutation hit counts depend only on prediction multiplicities."""
    if k > 8:
        raise ValueError('Exact audit supports at most eight candidates')
    predictions = tuple(label for label, count in enumerate(pattern) for unused in range(count))
    assert len(predictions) == k
    result = Counter()
    for labels in permutations(range(k)):
        result[sum(a == b for a, b in zip(predictions, labels))] += 1
    return result


def exact_permutation_p(primary, k):
    groups = defaultdict(list)
    for row in primary:
        groups[(row['context_id'], row['repeat'])].append(row)
    distribution = Counter({0: 1})
    for group in groups.values():
        assert len(group) == k and len({r['truth'] for r in group}) == k
        pattern = tuple(sorted(Counter(r['prediction'] for r in group).values(), reverse=True))
        distribution = convolve(distribution, null_histogram(pattern, k))
    observed = sum(r['correct'] for r in primary)
    return sum(count for value, count in distribution.items() if value >= observed) / sum(distribution.values())


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo = math.floor(position); hi = math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def audit(directory):
    if not __debug__:
        raise RuntimeError('Run this assertion-based audit without Python optimization (-O)')
    cfg, plan, summary = (read(directory / (part + '.json')) for part in ['config', 'plan', 'summary'])
    if cfg['task'] == 'single_slot_ttft' and cfg['design']['candidate_count'] > 8:
        raise ValueError('This bounded stdlib audit supports up to eight candidates; full-candidate auditing requires a separate implementation')
    assert cfg['schema_version'] == 2
    assert read(directory / 'execution.json')['state'] == 'complete'
    assert read(directory / 'status.json')['state'] == 'collection_complete'
    assert summary['physical_cache_measured'] is False
    assert summary['complete_prompt_reconstruction_evaluated'] is False
    truth, observations = rows(directory / 'ground_truth.jsonl'), rows(directory / 'observations.jsonl')
    truth_by_id = {r['request_id']: r for r in truth}
    assert len(truth_by_id) == len(truth) == len(observations) == plan['evaluation_requests'] == summary['observations']
    assert set(truth_by_id) == {r['request_id'] for r in observations}
    times, request_seconds = {}, defaultdict(float)
    for observation in observations:
        identifier = observation['request_id']; uuid_value = uuid.UUID(identifier)
        assert uuid_value.version == 4
        expected = truth_by_id[identifier]
        assert observation['phase'] == expected['phase'] and observation['condition'] == expected['condition']
        assert observation['ok'] is True and observation['request_body_bytes'] == 2048
        assert observation['body_bytes'] == cfg['execution']['output_steps'] * 64
        assert not ({'input_ids', 'class_id', 'generated_token_ids', 'host_visible_forward_and_argmax_ns'} & set(observation))
        markers = observation['markers']; raw = {r['read_id']: r for r in observation['reads']}
        assert len(markers) == cfg['execution']['output_steps']
        assert sum(r['body_bytes'] for r in raw.values()) == observation['body_bytes']
        for index, marker in enumerate(markers):
            assert marker['index'] == index and marker['elapsed_ns'] == raw[marker['read_id']]['elapsed_ns']
            assert raw[marker['read_id']]['body_bytes'] > 0
        assert all(a['elapsed_ns'] <= b['elapsed_ns'] for a, b in zip(markers, markers[1:]))
        times[identifier] = markers[0]['elapsed_ns'] / 1e6
        request_seconds[observation['condition']] += markers[-1]['elapsed_ns'] / 1e9
    source_manifest = read(directory / 'source_manifest.json')
    source_root = (directory/'source_snapshot').resolve()
    for name, expected in source_manifest.items():
        path = (source_root/name).resolve()
        assert path.is_relative_to(source_root)
        assert digest(path) == expected
    diagnostics = []
    for path in directory.glob('*_privileged.jsonl'):
        records = rows(path)
        count = 0 if path.name.endswith('_constant_privileged.jsonl') else cfg['execution']['output_steps']
        assert all(len(r['host_visible_forward_and_argmax_ns']) == count for r in records)
        assert all(all(isinstance(t,int) and t > 0 for t in r['host_visible_forward_and_argmax_ns']) for r in records)
        diagnostics.extend(records)
    runtime_records = [read(p) for p in directory.glob('*_runtime.json')]
    assert len(runtime_records) == len(plan['phases']) * len(cfg['design']['conditions'])
    for runtime in runtime_records:
        if runtime['condition'] == 'constant':
            assert runtime['inference'] is False
        else:
            assert runtime['inference'] is True and runtime['device'] == cfg['execution']['device']
            assert runtime['dtype'] == 'torch.' + cfg['execution']['dtype']
            assert runtime['model_parameters'] > 0

    warmups = [r for r in diagnostics if r['purpose'] == 'warmup']
    evaluated = [r for r in diagnostics if r['purpose'] == 'evaluation']
    assert len(warmups) == plan['warmup_requests'] and len(evaluated) == len(truth)
    assert {r['request_id'] for r in evaluated} == set(truth_by_id)
    result = {'passed': True, 'role': cfg['role'], 'task': cfg['task'], 'observations': len(truth),
              'warmups': len(warmups), 'source_hashes': len(source_manifest),
              'raw_read_marker_association_verified': True, 'request_seconds_by_condition': dict(request_seconds),
              'physical_cache_validated': False, 'complete_prompt_reconstruction_validated': False,
              'conditions': {}, 'run_name': directory.name, 'auditor_sha256':digest(Path(__file__))}
    corpus = {r['prompt_id']: r for r in rows(directory / 'source_snapshot/data/corpus-v1/corpus.jsonl')}
    if cfg['task'] == 'corpus_trace':
        assert not (directory / 'predictions.jsonl').exists()
        assert len(truth) == cfg['design']['prompt_limit']
        assert len({r['prompt_id'] for r in truth}) == len(truth)
        expected_order = list(corpus)[:cfg['design']['prompt_limit']]
        random.Random(cfg['design']['seed']).shuffle(expected_order)
        assert [r['prompt_id'] for r in truth] == expected_order
        for row in truth:
            assert row['input_ids'] == corpus[row['prompt_id']]['token_ids'] and row['class_id'] is None
        assert math.isclose(summary['first_marker_ms']['mean'], statistics.mean(times.values()), rel_tol=1e-10)
        for probability, saved in zip([.05, .5, .95], summary['first_marker_ms']['quantiles_05_50_95']):
            assert math.isclose(quantile(times.values(), probability), saved, rel_tol=1e-10)
        result.update(corpus_inputs_and_latency_distribution_verified=True,
                      unique_input_token_types=len({token for row in truth for token in row['input_ids']}),
                      input_token_positions=sum(len(row['input_ids']) for row in truth),
                      observed_response_markers=len(truth)*cfg['execution']['output_steps'])
    else:
        candidates = read(directory / 'design.json')['candidate_ids']
        k = len(candidates)
        assert candidates == sorted(set(candidates)) and k == cfg['design']['candidate_count']
        vocabulary = read(directory/'source_snapshot/data/corpus-v1/vocabulary.json')['token_ids']
        assert candidates == sorted(random.Random(cfg['design']['candidate_seed']).sample(vocabulary,k))
        groups = defaultdict(Counter)
        for row in truth:
            groups[(row['condition'],row['phase'],row['context_id'],row['repeat'])][row['class_id']] += 1
        phases = [('profile','profile_contexts','profile_repeats'),('validation','profile_contexts','validation_repeats'),('new_session_new_context','test_contexts','test_repeats')]
        expected_groups = {(condition,phase,context,repeat) for condition in cfg['design']['conditions']
                           for phase,contexts,repeats in phases for context in cfg['design'][contexts]
                           for repeat in range(cfg['design'][repeats])}
        assert set(groups) == expected_groups
        assert all(group == Counter({candidate:1 for candidate in candidates}) for group in groups.values())
        predictions = rows(directory / 'predictions.jsonl')
        by_id = {r['request_id']: r for r in predictions}
        assert len(by_id) == len(predictions) == sum(r['phase'] != 'profile' for r in truth)
        for row in truth:
            expected = corpus[row['context_id']]['token_ids'][:16].copy()
            expected[cfg['design']['secret_position']] = row['class_id']
            assert row['input_ids'] == expected and row['class_id'] in candidates
        primary = {}
        for condition in cfg['design']['conditions']:
            profile = read(directory / (condition + '_profile.json'))
            assert profile['candidate_ids'] == candidates
            samples = defaultdict(list)
            for row in truth:
                if row['condition'] == condition and row['phase'] == 'profile':
                    samples[row['class_id']].append(times[row['request_id']])
            for index, candidate in enumerate(candidates):
                values = samples[candidate]
                assert len(values) == plan['calibration_samples_per_candidate'] == profile['samples'][index]
                assert math.isclose(statistics.mean(values), profile['mean_ms'][index], rel_tol=1e-10)
                assert math.isclose(max(.001, statistics.stdev(values)), profile['sigma_ms'][index], rel_tol=1e-9)
            result['conditions'][condition] = {}
            for phase in ['validation', 'new_session_new_context']:
                selected = [r for r in truth if r['condition'] == condition and r['phase'] == phase]
                decisions = []
                for row in selected:
                    t = times[row['request_id']]
                    scores = [(-.5*((t-mu)/sigma)**2, candidate) for candidate,mu,sigma in zip(candidates,profile['mean_ms'],profile['sigma_ms'])]
                    ranked = sorted(scores,key=lambda item:(-item[0],item[1]))[:min(5,k)]
                    saved = by_id[row['request_id']]
                    assert saved['topk'] == [candidate for score,candidate in ranked]
                    for (score, unused), actual in zip(ranked, saved['log_compatibility']):
                        assert math.isclose(score,actual,rel_tol=1e-9,abs_tol=1e-10)
                    decisions.append({**row,'truth':row['class_id'],'prediction':ranked[0][1],
                                      'correct':ranked[0][1]==row['class_id'],'top5':row['class_id'] in saved['topk']})
                metric = summary['conditions'][condition][phase]
                assert metric['requests'] == len(selected)
                correct = sum(r['correct'] for r in decisions)
                assert metric['correct_top1'] == correct and metric['top1'] == correct/len(selected)
                assert metric['top5'] == sum(r['top5'] for r in decisions)/len(selected)
                result['conditions'][condition][phase] = {'correct_top1':correct,'requests':len(selected)}
                if phase == 'new_session_new_context':
                    primary[condition] = decisions
                    if cfg['role'] != 'technical_smoke':
                        blocks = defaultdict(list)
                        for row in decisions: blocks[row['repeat']].append(row)
                        sizes = {len(block) for block in blocks.values()}; assert len(sizes)==1
                        exact_ci = exact_block_interval([sum(r['correct'] for r in block) for block in blocks.values()], sizes.pop())
                        assert max(abs(a-b) for a,b in zip(exact_ci, metric['ci95'])) <= 3/len(decisions)
                        exact_p = exact_permutation_p(decisions,k)
                        tolerance = 6*math.sqrt(exact_p*(1-exact_p)/cfg['analysis']['resamples'])+2/(cfg['analysis']['resamples']+1)
                        assert abs(exact_p-metric['one_sided_permutation_p']) <= tolerance
                        result['conditions'][condition][phase].update(exact_conditional_permutation_p=exact_p,exact_block_bootstrap_ci95=exact_ci,monte_carlo_statistics_consistent=True)
        if cfg['role'] != 'technical_smoke':
            result['paired'] = {}
            for control in ['input_blinded','constant']:
                key = lambda row:(row['context_id'],row['repeat'],row['truth'])
                other = {key(r):r for r in primary[control]}
                block_delta = Counter(); sizes = Counter()
                for row in primary['model']:
                    block_delta[row['repeat']] += int(row['correct'])-int(other[key(row)]['correct'])
                    sizes[row['repeat']] += 1
                assert len(set(sizes.values()))==1
                ci = exact_block_interval(list(block_delta.values()),next(iter(sizes.values())))
                metric = summary['paired_primary'][control]
                assert math.isclose(metric['difference'],sum(block_delta.values())/len(primary['model']),abs_tol=1e-12)
                assert max(abs(a-b) for a,b in zip(ci,metric['ci95'])) <= 3/len(primary['model'])
                result['paired'][control] = {'difference':metric['difference'],'exact_ci95':ci}
        result['profiles_rankings_and_metrics_recounted'] = True
    names = ['config.json','plan.json','summary.json','ground_truth.jsonl','observations.jsonl','source_manifest.json']
    if (directory/'predictions.jsonl').exists(): names.append('predictions.jsonl')
    result['artifact_sha256'] = {name:digest(directory/name) for name in names}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists(): raise ValueError('Audit report exists; refusing overwrite')
    result = audit(args.run.resolve())
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
