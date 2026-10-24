"""Balanced randomized input-token experiment; ground truth is stored separately."""
import random
import uuid


def design(cfg, tokenizer):
    candidates = []
    for i, word in enumerate(cfg['candidate_words']):
        ids = tokenizer.encode(' ' + word, add_special_tokens=False).ids
        if len(ids) != 1:
            raise ValueError(f'Candidate is not a single BPE token: {word}')
        candidates.append({'class_id': f'C{i:02d}', 'word': word, 'token_id': ids[0]})
    suffix = tokenizer.encode(cfg['suffix'], add_special_tokens=False).ids
    if len(suffix) != cfg['input_length'] - cfg['secret_position'] - 1:
        raise ValueError('Suffix length differs from contract')
    contexts = []
    for i, text in enumerate(cfg['context_prefixes']):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < cfg['secret_position']:
            raise ValueError('Context too short')
        contexts.append({'context_id': f'X{i:02d}', 'prefix_token_ids': ids[:cfg['secret_position']]})
    jobs = []
    for session in range(cfg['sessions']):
        rng = random.Random(cfg['design_seed'] + session)
        for repeat in range(cfg['repetitions']):
            block = []
            for context in contexts:
                for candidate in candidates:
                    token_ids = context['prefix_token_ids'] + [candidate['token_id']] + suffix
                    block.append({'request_id': str(uuid.uuid4()), 'session': session, 'repeat': repeat,
                                  'context_id': context['context_id'], 'class_id': candidate['class_id'],
                                  'input_ids': token_ids})
            rng.shuffle(block)
            jobs.extend(block)
    return {'candidates': candidates, 'contexts': contexts, 'jobs': jobs,
            'randomization': 'balanced context x candidate block per repetition; opaque UUID4 request IDs'}
