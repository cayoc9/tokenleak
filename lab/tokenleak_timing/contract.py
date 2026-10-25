"""Validate and materialize the registered design without loading ML libraries."""
import hashlib
import json
from pathlib import Path
import random
import uuid

APP = Path(__file__).resolve().parents[1]
CONDITIONS = ('model', 'input_blinded', 'constant')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_config(path, app=APP):
    cfg = json.loads(Path(path).read_text())
    validate(cfg)
    corpus_path = app / 'data/corpus-v1/corpus.jsonl'
    vocabulary_path = app / 'data/corpus-v1/vocabulary.json'
    splits_path = app / 'data/corpus-v1/splits.json'
    for key, source in [('corpus_sha256', corpus_path), ('vocabulary_sha256', vocabulary_path),
                        ('splits_sha256', splits_path)]:
        if sha(source) != cfg['data'][key]:
            raise ValueError('Frozen data identity differs: ' + key)
    reference = app / 'assets/model_references' / (cfg['model']['reference'] + '.json')
    model = json.loads(reference.read_text())
    if (cfg['model']['id'], cfg['model']['revision']) != (model['model_id'], model['revision']):
        raise ValueError('Model reference and protocol identity differ')
    return cfg


def validate(cfg):
    if cfg.get('schema_version') != 2 or cfg.get('task') not in {'single_slot_ttft', 'corpus_trace'}:
        raise ValueError('Only the registered v2 task types are supported')
    if cfg.get('role') not in {'technical_smoke', 'exploratory', 'registered_measurement', 'diagnostic'}:
        raise ValueError('Explicit execution role required')
    if cfg['observation'] != {'access': 'coordinator_instrumented_loopback',
                              'primary_feature': 'first_response_marker_ms',
                              'physical_cache': False, 'input_position_alignment': False}:
        raise ValueError('This runner implements an instrumental timing-only view')
    if cfg['model']['reference'] not in {'gpt2-small', 'gpt2-xl'}:
        raise ValueError('Unsupported model reference')
    device, dtype = cfg['execution']['device'], cfg['execution']['dtype']
    if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('Select cpu or one explicit cuda:N device; no automatic offload')
    if dtype not in {'float32', 'float16', 'bfloat16'} or (device == 'cpu' and dtype != 'float32'):
        raise ValueError('CPU uses float32; GPU dtype must be explicit')
    if not 1 <= cfg['execution']['threads'] <= 32:
        raise ValueError('Invalid explicit thread budget')
    if not 1 <= cfg['execution']['output_steps'] <= 64 or cfg['execution']['request_bytes'] != 2048:
        raise ValueError('Invalid fixed transport budget')
    if cfg['execution']['warmup_requests'] < 0 or cfg['execution']['max_seconds'] <= 0:
        raise ValueError('Invalid warmup or time budget')
    if cfg['task'] == 'single_slot_ttft':
        d = cfg['design']
        if not 2 <= d['candidate_count'] <= 8431:
            raise ValueError('Candidate count must be 2..8431')
        if d['input_length'] != 16 or not 0 <= d['secret_position'] < d['input_length']:
            raise ValueError('This design uses a public slot in a 16-token input')
        if set(d['conditions']) != set(CONDITIONS) or len(d['conditions']) != 3:
            raise ValueError('Model, input-blinded and no-inference controls are required')
        if len(d['profile_contexts']) != 2 or len(d['test_contexts']) != 2:
            raise ValueError('Two profiling and two held-out public contexts required')
        if set(d['profile_contexts']) & set(d['test_contexts']):
            raise ValueError('Profiling and held-out contexts must differ')
        if d['profile_repeats'] < 1 or d['validation_repeats'] < 1 or d['test_repeats'] < 1:
            raise ValueError('Nonempty phases required')
        if d['profile_repeats'] * len(d['profile_contexts']) < 2:
            raise ValueError('Variance estimation requires at least two samples per type')
        if cfg['analysis'] != {'ranking': 'log_gaussian_compatibility', 'sigma_floor_ms': 0.001,
                               'primary_partition': 'new_session_new_context',
                               'resamples': cfg['analysis']['resamples'],
                               'seed': cfg['analysis']['seed']}:
            raise ValueError('Unexpected analysis or feature definition')
        if cfg['analysis']['resamples'] < 99:
            raise ValueError('At least 99 resamples required; smoke results remain technical')
        if cfg['role'] != 'technical_smoke' and d['test_repeats'] < 8:
            raise ValueError('Measured analysis requires at least eight repetition blocks')
    else:
        if not 1 <= cfg['design']['prompt_limit'] <= 5000:
            raise ValueError('Corpus trace count must be 1..5000')
        if cfg['design']['conditions'] != ['model']:
            raise ValueError('Corpus trace is a diagnostic model-only collection, not reconstruction')


def materialize(cfg, app=APP):
    corpus = [json.loads(line) for line in (app / 'data/corpus-v1/corpus.jsonl').read_text().splitlines()]
    vocabulary = json.loads((app / 'data/corpus-v1/vocabulary.json').read_text())['token_ids']
    if len(corpus) != 5000 or len(vocabulary) != 8431 or len(set(vocabulary)) != 8431:
        raise ValueError('Corpus cardinality differs from registered data')
    if cfg['task'] == 'corpus_trace':
        selected = corpus[:cfg['design']['prompt_limit']]
        return {'corpus': selected, 'candidate_ids': [], 'phases': [
            {'name': 'corpus_trace', 'session': 0, 'repeats': 1, 'contexts': []}],
            'target': 'raw response timing for whole corpus inputs; no input-token alignment'}
    index = {row['prompt_id']: row for row in corpus}
    d = cfg['design']
    contexts = {}
    for name in d['profile_contexts'] + d['test_contexts']:
        row = index[name]
        if len(row['token_ids']) < d['input_length']:
            raise ValueError('Public context is shorter than registered input')
        contexts[name] = {'input_ids': row['token_ids'][:d['input_length']],
                          'source_group_id': row['source_group_id']}
    profile_groups = {contexts[name]['source_group_id'] for name in d['profile_contexts']}
    test_groups = {contexts[name]['source_group_id'] for name in d['test_contexts']}
    if profile_groups & test_groups:
        raise ValueError('Profiling and test instruction families overlap')
    candidate_ids = sorted(random.Random(d['candidate_seed']).sample(vocabulary, d['candidate_count']))
    return {'contexts': contexts, 'candidate_ids': candidate_ids, 'phases': [
        {'name': 'profile', 'session': 0, 'repeats': d['profile_repeats'], 'contexts': d['profile_contexts']},
        {'name': 'validation', 'session': 1, 'repeats': d['validation_repeats'], 'contexts': d['profile_contexts']},
        {'name': 'new_session_new_context', 'session': 2, 'repeats': d['test_repeats'], 'contexts': d['test_contexts']}],
        'target': 'identity of one input token in a public fixed slot; direct input_ids laboratory interface'}


def plan(cfg, data):
    phases = []
    for phase in data['phases']:
        n = (len(data['corpus']) if cfg['task'] == 'corpus_trace' else
             len(data['candidate_ids']) * len(phase['contexts']) * phase['repeats'])
        phases.append({**phase, 'requests_per_condition': n})
    conditions = len(cfg['design']['conditions'])
    total = sum(x['requests_per_condition'] for x in phases) * conditions
    return {'protocol_id': cfg['protocol_id'], 'role': cfg['role'], 'task': cfg['task'],
            'model': cfg['model'], 'execution': cfg['execution'], 'phases': phases,
            'conditions': cfg['design']['conditions'], 'candidate_count': len(data['candidate_ids']),
            'evaluation_requests': total,
            'warmup_requests': len(phases) * conditions * cfg['execution']['warmup_requests'],
            'calibration_samples_per_candidate': (
                len(data['phases'][0]['contexts']) * data['phases'][0]['repeats']
                if cfg['task'] == 'single_slot_ttft' else None),
            'input_token_positions_are_not_observed_times': True,
            'model_inference_executed_by_plan': False,
            'physical_cache_measured': False, 'complete_prompt_reconstruction_evaluated': False}


def jobs(cfg, data, phase, condition):
    if cfg['task'] == 'corpus_trace':
        order = list(data['corpus'])
        random.Random(cfg['design']['seed']).shuffle(order)
        for row in order:
            yield {'request_id': str(uuid.uuid4()), 'phase': phase['name'], 'session': phase['session'],
                   'condition': condition, 'repeat': 0, 'context_id': row['prompt_id'],
                   'input_ids': row['token_ids'], 'class_id': None, 'prompt_id': row['prompt_id']}
        return
    seed = cfg['design']['seed'] + 10000 * phase['session'] + 100 * CONDITIONS.index(condition)
    rng = random.Random(seed)
    for repeat in range(phase['repeats']):
        block = [(context, candidate) for context in phase['contexts'] for candidate in data['candidate_ids']]
        rng.shuffle(block)
        for context, candidate in block:
            ids = list(data['contexts'][context]['input_ids'])
            ids[cfg['design']['secret_position']] = candidate
            yield {'request_id': str(uuid.uuid4()), 'phase': phase['name'], 'session': phase['session'],
                   'condition': condition, 'repeat': repeat, 'context_id': context,
                   'input_ids': ids, 'class_id': candidate}


def verify_model(model_dir, cfg, app=APP):
    reference = json.loads((app / 'assets/model_references' / (cfg['model']['reference'] + '.json')).read_text())
    local = json.loads((model_dir / 'manifest.json').read_text())
    if local['model_id'] != reference['model_id'] or local['revision'] != reference['revision']:
        raise ValueError('Local model identity differs')
    # Always verify against the shipped reference, never trust user-editable local hashes.
    for item in reference['files']:
        path = model_dir / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Model asset integrity differs: ' + item['path'])
    return reference
