"""Metricas de classificacao, sequencia e recuperacao do segredo sintetico."""
from collections import Counter
import numpy as np


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(token != other)))
        previous = current
    return previous[-1]


def evaluate(records, predictions, frequencies, with_edit=True):
    truth = np.array([t for r in records for t in r['token_ids']], dtype=np.int32)
    if predictions.shape != (len(truth), 5):
        raise ValueError('Prediction count differs from the full reference denominator')
    correct1 = predictions[:, 0] == truth
    correct5 = (predictions == truth[:, None]).any(axis=1)
    lengths = np.array([len(r['token_ids']) for r in records])
    boundaries = np.r_[0, np.cumsum(lengths)]
    per_prompt = []
    for i, record in enumerate(records):
        a, b = boundaries[i:i+2]
        exact = bool(correct1[a:b].all())
        span_correct = []
        for span in record.get('protected_spans', []):
            start, stop = a+span['start'], a+span['start']+span['length']
            if start < a or stop > b or stop <= start:
                raise ValueError('Invalid protected-span annotation')
            span_correct.append(bool(correct1[start:stop].all()))
        per_prompt.append({'prompt_id': record['prompt_id'], 'category_id': record['category_id'],
                           'length': int(b-a), 'top1': float(correct1[a:b].mean()),
                           'top5': float(correct5[a:b].mean()), 'exact_match': exact,
                           'protected_span_exact': span_correct,
                           'normalized_edit_distance': (0. if exact else edit_distance(truth[a:b].tolist(), predictions[a:b, 0].tolist())/(b-a)) if with_edit else None})
    support = Counter(truth.tolist())
    recovered = Counter(truth[correct1].tolist())
    balanced = float(np.mean([recovered[t]/support[t] for t in support]))
    pred_support = Counter(predictions[:, 0].tolist())
    macro_f1 = float(np.mean([2*recovered[t]/(support[t]+pred_support[t]) for t in support]))
    # Full-label single-label multiclass: micro-F1 equals ordinary accuracy.
    spans = [value for item in per_prompt for value in item['protected_span_exact']]
    rare_mask = np.array([frequencies[str(t)] < 10 for t in truth])
    per_category = {}
    for category in sorted({r['category_id'] for r in records}):
        items = [p for p in per_prompt if p['category_id'] == category]
        per_category[category] = {'prompts': len(items), 'top1_prompt_macro': float(np.mean([p['top1'] for p in items]))}
    summary = {'positions': len(truth), 'prompts': len(records), 'correct_top1': int(correct1.sum()),
               'correct_top5': int(correct5.sum()), 'top1_micro': float(correct1.mean()),
               'top5_micro': float(correct5.mean()), 'top1_prompt_macro': float(np.mean([p['top1'] for p in per_prompt])),
               'top5_prompt_macro': float(np.mean([p['top5'] for p in per_prompt])),
               'prompt_exact_match': float(np.mean([p['exact_match'] for p in per_prompt])),
               'synthetic_protected_span_exact_match': float(np.mean(spans)) if spans else None,
               'protected_span_count': len(spans), 'token_balanced_recall': balanced,
               'token_macro_f1_observed_true_classes': macro_f1,
               'rare_token_count': int(rare_mask.sum()),
               'rare_token_top1': float(correct1[rare_mask].mean()) if rare_mask.any() else None,
               'frequent_token_top1': float(correct1[~rare_mask].mean()) if (~rare_mask).any() else None,
               'normalized_edit_distance': float(np.mean([p['normalized_edit_distance'] for p in per_prompt])) if with_edit else None,
               'by_category': per_category,
               'scope': 'Known input positions, synthetic signals and synthetic protected keyword spans; no physical leakage demonstrated.'}
    return summary, per_prompt
