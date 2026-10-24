"""Corpus novo, controlado por palavras publicas; nao e o SPC-5K historico."""
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from tokenizers import Tokenizer

from .io import digest, write_json, manifest, check_manifest


TOPICS = ['agriculture', 'architecture', 'astronomy', 'biology', 'cooking',
          'cybersecurity', 'economics', 'education', 'energy', 'engineering',
          'environment', 'finance', 'geography', 'history', 'law', 'literature',
          'mathematics', 'medicine', 'music', 'transportation']
TEMPLATES = ['Discuss {topic}:', 'Explain {topic}:', 'Study {topic}:',
             'Review {topic}:', 'Explore {topic}:', 'Outline {topic}:',
             'Summarize {topic}:', 'Compare {topic} terms:',
             'List {topic} concepts:', 'Describe {topic} topics:']


def build(config, output, workspace, app):
    """Fixa o vocabulario ANTES das particoes e de qualquer resultado de ataque."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = config['dataset']
    n = cfg['n_prompts']
    if n % 200:
        raise ValueError('n_prompts must be a multiple of 20 topics x 10 template families')
    per_cell = n // 200
    reference = Path(workspace) / cfg['tokenizer_directory']
    reference_manifest = json.loads((reference / 'manifesto.json').read_text())
    for item in reference_manifest['files']:
        if digest(Path(workspace) / item['path']) != item['sha256']:
            raise ValueError('Tokenizer reference was changed')
    tokenizer = Tokenizer.from_file(str(reference / 'tokenizer.json'))
    lexicon = Path(app) / 'assets/cmudict'
    lexicon_manifest = json.loads((lexicon / 'manifesto.json').read_text())
    for item in lexicon_manifest['files']:
        if digest(lexicon / item['path']) != item['sha256']:
            raise ValueError('Public dictionary was changed')
    words = {re.sub(r'\(\d+\)$', '', line.split()[0]).lower()
             for line in (lexicon / 'cmudict.dict').read_text().splitlines()
             if line and not line.startswith(';;;')}
    prefixes = [(c, t, TEMPLATES[t].format(topic=topic))
                for c, topic in enumerate(TOPICS) for t in range(10)]
    base_counts = Counter()
    for _, _, prefix in prefixes:
        for token in tokenizer.encode(prefix, add_special_tokens=False).ids:
            base_counts[token] += per_cell
    base_ids = set(base_counts)
    candidates = []
    for token, idx in tokenizer.get_vocab().items():
        if idx in base_ids or not token.startswith('Ġ'):
            continue
        word = tokenizer.decode([idx], skip_special_tokens=False)
        if not re.fullmatch(r' [a-z]{2,}', word) or word.strip() not in words:
            continue
        if tokenizer.encode(word, add_special_tokens=False).ids == [idx]:
            candidates.append((idx, word.strip()))
    candidates.sort()
    size = cfg['unique_token_types']
    requested = size - len(base_ids)
    if not 0 < requested <= len(candidates):
        raise ValueError(f'Need {requested} stable words; available {len(candidates)}')
    rng_vocab = np.random.default_rng(cfg['vocabulary_seed'])
    chosen = [candidates[i] for i in rng_vocab.choice(len(candidates), requested, replace=False)]
    frequent_total = round(size * cfg['frequent_type_fraction'])
    frequent_slots = frequent_total - len(base_ids)
    if frequent_slots <= 0 or min(base_counts.values()) < cfg['frequent_min_count']:
        raise ValueError('Corpus too small to enforce the frequency/type contract')
    rng_generation = np.random.default_rng(cfg['generation_seed'])
    descriptors = []
    for c, t, prefix in prefixes:
        prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False).ids)
        for sample in range(per_cell):
            desired = int(rng_generation.integers(cfg['min_tokens'], cfg['max_tokens'] + 1))
            if desired - prefix_len < 1:
                raise ValueError('A template leaves no word slot at the specified minimum length')
            descriptors.append((c, t, sample, prefix, prefix_len, desired - prefix_len))
    total_slots = sum(x[-1] for x in descriptors)
    rng_frequency = np.random.default_rng(cfg['frequency_seed'])
    rare_choices = np.arange(1, cfg['frequent_min_count'])
    rare_p = rare_choices.astype(float) ** -cfg['zipf_exponent']
    rare_counts = rng_frequency.choice(rare_choices, requested - frequent_slots, p=rare_p / rare_p.sum())
    remaining = total_slots - frequent_slots * cfg['frequent_min_count'] - int(rare_counts.sum())
    if remaining < 0:
        raise ValueError('Insufficient positions for the declared frequency constraints')
    rank_p = np.arange(1, frequent_slots + 1, dtype=float) ** -cfg['zipf_exponent']
    high_counts = cfg['frequent_min_count'] + rng_frequency.multinomial(remaining, rank_p / rank_p.sum())
    counts = np.concatenate([high_counts, rare_counts])
    slots = np.repeat(np.arange(requested, dtype=np.int32), counts)
    # Retry only for duplicate text, never for attack performance. Counts remain fixed.
    for attempt in range(20):
        rng_generation.shuffle(slots)
        records = []
        cursor = 0
        for c, t, sample, prefix, prefix_len, length in descriptors:
            picked = slots[cursor:cursor + length]
            cursor += length
            text = prefix + ''.join(' ' + chosen[i][1] for i in picked)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            expected = tokenizer.encode(prefix, add_special_tokens=False).ids + [chosen[i][0] for i in picked]
            if ids != expected:
                raise ValueError('Tokenization boundary changed the intended controlled sequence')
            # Proxy of a protected field; not an actual personal identifier or original-paper annotation.
            protected_len = min(3, length)
            records.append({'prompt_id': f'C{c+1:02d}-T{t:02d}-{sample:03d}',
                            'category_id': f'C{c+1:02d}', 'category_name': TOPICS[c],
                            'source_group_id': f'T{t:02d}', 'text': text, 'token_ids': ids,
                            'tokenizer_revision': cfg['tokenizer_revision'],
                            'protected_spans': [{'kind': 'synthetic_protected_keyword_span',
                                                 'start': len(ids) - protected_len, 'length': protected_len}],
                            'provenance': {'generation_method': cfg['method'],
                                           'generation_seed': cfg['generation_seed'],
                                           'lexicon_commit': lexicon_manifest['commit']}})
        if len({r['text'] for r in records}) == n:
            break
    else:
        raise ValueError('Could not construct a duplicate-free corpus')
    family_order = np.random.default_rng(cfg['split_seed']).permutation(10)
    groups = {'train': [f'T{x:02d}' for x in family_order[:6]],
              'validation': [f'T{x:02d}' for x in family_order[6:8]],
              'test': [f'T{x:02d}' for x in family_order[8:]]}
    split = {name: [r['prompt_id'] for r in records if r['source_group_id'] in values]
             for name, values in groups.items()}
    split['profiling'] = split['train'] + split['validation']
    with (output / 'corpus.jsonl').open('w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    frequencies = Counter(tok for record in records for tok in record['token_ids'])
    vocabulary = sorted(frequencies)
    write_json(output / 'vocabulary.json', {'token_ids': vocabulary,
               'selection': 'public closed vocabulary frozen before split and attack evaluation',
               'frequencies': {str(k): frequencies[k] for k in vocabulary}})
    write_json(output / 'splits.json', {'groups': groups, 'members': split,
               'seed': cfg['split_seed'], 'group_unit': 'global instruction template family'})
    write_json(output / 'generation.json', {'config': cfg, 'lexicon': lexicon_manifest,
               'tokenizer': reference_manifest, 'available_word_candidates': len(candidates),
               'duplicate_retry_attempt': attempt, 'distribution': 'constrained Zipf-like counts; not the original unknown generator',
               'limitations': ['Keyword-list tasks; not a natural-conversation benchmark.',
                              'Categories label instructions; sampled words are shared across categories.',
                              'Protected spans are synthetic keyword proxies, not real personal data.']})
    write_json(output / 'manifest.json', manifest(output))
    audit = audit_corpus(output, config, tokenizer)
    write_json(output / 'audit.json', audit)
    write_json(output / 'manifest.json', manifest(output))
    return audit


def load(directory):
    directory = Path(directory)
    check_manifest(directory)
    records = [json.loads(line) for line in (directory / 'corpus.jsonl').read_text().splitlines()]
    split = json.loads((directory / 'splits.json').read_text())
    vocab = json.loads((directory / 'vocabulary.json').read_text())
    return records, split, vocab


def audit_corpus(directory, config, tokenizer=None):
    records, split, vocab = load(directory)
    cfg = config['dataset']
    ids = [r['prompt_id'] for r in records]
    frequencies = Counter(t for r in records for t in r['token_ids'])
    lengths = [len(r['token_ids']) for r in records]
    checks = {'prompt_count': len(records) == cfg['n_prompts'], 'unique_ids': len(set(ids)) == len(ids),
              'unique_texts': len({r['text'] for r in records}) == len(records),
              'category_count': len({r['category_id'] for r in records}) == 20,
              'lengths': min(lengths) >= cfg['min_tokens'] and max(lengths) <= cfg['max_tokens'],
              'vocabulary_size': len(frequencies) == cfg['unique_token_types'],
              'frequency_contract': sum(n >= cfg['frequent_min_count'] for n in frequencies.values()) == round(cfg['unique_token_types'] * cfg['frequent_type_fraction']),
              'train_validation_test_disjoint': not (set(split['members']['train']) & set(split['members']['validation']) or set(split['members']['train']) & set(split['members']['test']) or set(split['members']['validation']) & set(split['members']['test'])),
              'group_disjoint': len(set(sum(split['groups'].values(), []))) == 10,
              'split_80_20': len(split['members']['profiling']) == 4 * len(split['members']['test']),
              'all_members_accounted': set(ids) == set(split['members']['profiling']) | set(split['members']['test']),
              'retokenization': tokenizer is None or all(tokenizer.encode(r['text'], add_special_tokens=False).ids == r['token_ids'] for r in records)}
    if not all(checks.values()):
        raise ValueError(f'Corpus audit failed: {checks}')
    return {'checks': checks, 'passed': True, 'n_prompts': len(records), 'token_positions': sum(lengths),
            'unique_token_types': len(frequencies), 'length_min': min(lengths), 'length_max': max(lengths),
            'length_mean': float(np.mean(lengths)),
            'frequent_types': sum(n >= cfg['frequent_min_count'] for n in frequencies.values()),
            'rare_types': sum(n < cfg['frequent_min_count'] for n in frequencies.values()),
            'partition_sizes': {k: len(v) for k, v in split['members'].items()},
            'by_category': dict(Counter(r['category_id'] for r in records))}
