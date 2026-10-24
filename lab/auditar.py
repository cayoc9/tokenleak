"""Auditoria independente dos arquivos de uma execucao congelada, sem retreino."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(abs(actual - expected) <= 1e-12, message)


def check_manifest(directory):
    expected = json.loads((directory / 'manifest.json').read_text())
    actual = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
              if p.is_file() and p.name != 'manifest.json'}
    require(actual == expected, f'Manifesto incompleto ou alterado: {directory}')
    return len(expected)


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        row = [i]
        for j, other in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = row
    return previous[-1]


def audit(output, workspace):
    """Nao importa o avaliador de metricas nem o reconstrutor da execucao."""
    cfg = json.loads((output / 'config.json').read_text())
    require(cfg['mode'] == 'fully_simulated', 'Modo fora do escopo desta auditoria')
    require(cfg['alignment'] == 'oracle_input_positions', 'Alinhamento fora do escopo')
    require(cfg['evaluation']['confidence_level'] == .95 and cfg['trcs']['top_k'] == 5,
            'Esta versao implementa IC de 95% e Top-5')
    directory = output / 'corpus'
    files_checked = check_manifest(directory)
    records = [json.loads(line) for line in (directory / 'corpus.jsonl').read_text().splitlines()]
    split = json.loads((directory / 'splits.json').read_text())
    vocabulary = json.loads((directory / 'vocabulary.json').read_text())
    tokens = np.array(vocabulary['token_ids'], dtype=np.int32)
    frequencies = Counter(t for r in records for t in r['token_ids'])
    require(tokens.tolist() == sorted(frequencies), 'Vocabulario nao corresponde ao corpus')
    require({str(k): v for k, v in frequencies.items()} == vocabulary['frequencies'], 'Frequencias divergentes')
    require(len(records) == cfg['dataset']['n_prompts'], 'Contagem de prompts incorreta')
    require(len(tokens) == cfg['dataset']['unique_token_types'], 'Tipos incorretos')
    require(len({r['prompt_id'] for r in records}) == len(records), 'IDs repetidos')
    require(len({r['text'] for r in records}) == len(records), 'Textos repetidos')
    require(set(Counter(r['category_id'] for r in records).values()) == {len(records) // 20}, 'Categorias desequilibradas')
    require(sum(v >= cfg['dataset']['frequent_min_count'] for v in frequencies.values()) ==
            round(len(tokens) * cfg['dataset']['frequent_type_fraction']), 'Contrato de frequencia incorreto')
    tokenizer_dir = workspace / cfg['dataset']['tokenizer_directory']
    if not tokenizer_dir.exists():
        # Historical v1 artifacts used a dissertation-relative asset path.
        tokenizer_dir = workspace / 'lab/assets/tokenizer/gpt2-xl'
    token_manifest = json.loads((tokenizer_dir / 'manifesto.json').read_text())
    require(token_manifest['revision'] == cfg['dataset']['tokenizer_revision'], 'Revisao do tokenizador divergente')
    for item in token_manifest['files']:
        require(sha(workspace / item['path']) == item['sha256'], 'Referencia do tokenizador alterada')
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / 'tokenizer.json'))
    for record in records:
        require(tokenizer.encode(record['text'], add_special_tokens=False).ids == record['token_ids'], 'Retokenizacao divergente')
        require(record['tokenizer_revision'] == cfg['dataset']['tokenizer_revision'], 'Revisao divergente')
        require(cfg['dataset']['min_tokens'] <= len(record['token_ids']) <= cfg['dataset']['max_tokens'], 'Comprimento invalido')
        for span in record['protected_spans']:
            require(0 <= span['start'] < span['start'] + span['length'] <= len(record['token_ids']), 'Trecho invalido')
    partition = {}
    for name, ratio in [('train', .6), ('validation', .2), ('test', .2)]:
        expected = [r for r in records if r['source_group_id'] in split['groups'][name]]
        require({r['prompt_id'] for r in expected} == set(split['members'][name]), 'Membro fora da familia declarada')
        require(len(expected) == len(split['members'][name]) == round(len(records) * ratio), 'Split incorreto')
        partition[name] = expected
    all_groups = sum(split['groups'].values(), [])
    require(len(all_groups) == len(set(all_groups)) == 10, 'Familias compartilhadas')
    require(set(split['members']['profiling']) == set(split['members']['train'] + split['members']['validation']), 'Profiling incorreto')
    require(len(split['members']['profiling']) == round(.8 * len(records)), 'Duplicacao no profiling')
    train_ids = np.array([t for r in partition['train'] for t in r['token_ids']], dtype=np.int32)
    truth = np.array([t for r in partition['test'] for t in r['token_ids']], dtype=np.int32)
    lengths = np.array([len(r['token_ids']) for r in partition['test']])
    borders = np.r_[0, lengths.cumsum()]
    result_dir = output / 'results'
    files_checked += check_manifest(result_dir)
    reference = arrays(result_dir / 'evaluation_reference.npz')
    require(np.array_equal(truth, reference['true_tokens']), 'Referencia mudou')
    require(np.array_equal(lengths, reference['lengths']), 'Limites dos prompts mudaram')
    aggregate = json.loads((result_dir / 'summary.json').read_text())
    round_reports, recount_rows = [], []
    scalar_positions = edit_positions = 0
    # Importa apenas o gerador congelado para conferir o replay dos sinais por seed.
    sys.path.insert(0, str(output / 'source_snapshot'))
    from tokenleak_lab.signals import SyntheticEnvironment, rng_for
    environment = SyntheticEnvironment(tokens, cfg['signals'])
    width = cfg['signals']['cache_width']
    prior_freq = Counter(train_ids.tolist())
    prior = sorted(tokens.tolist(), key=lambda t: (-prior_freq[t], t))[:5]
    for seed in cfg['evaluation']['round_seeds']:
        rd = result_dir / f'round_{seed:02d}'
        check_manifest(rd)
        raw = arrays(rd / 'profile_measurements.npz')
        p = arrays(rd / 'profiles.npz')
        require(np.array_equal(p['token_ids'], tokens) and np.array_equal(raw['token_ids'], tokens), 'Perfis de outro vocabulario')
        require(raw['timing_ms'].shape == (len(tokens), cfg['signals']['profile_samples_per_type']), 'Orcamento de perfis divergente')
        require(np.array_equal(raw['timing_ms'].mean(axis=1), p['mu']), 'Media nao vem das amostras')
        require(np.array_equal(np.maximum(raw['timing_ms'].std(axis=1, ddof=1), 1e-6), p['sigma']), 'Sigma divergente')
        # Recalculo em lotes: nao expande todos os bits do perfil simultaneamente.
        for start in range(0, len(tokens), 128):
            cached = np.unpackbits(raw['cache_packed'][start:start + 128], axis=-1)[..., :width].mean(axis=1)
            require(np.array_equal(cached, p['cache'][start:start + 128]), 'Perfil cache nao vem das amostras')
        training = arrays(rd / 'weight_training.npz')
        require(np.array_equal(train_ids[training['source_positions']], training['true_tokens']), 'Treino acessou outra particao')
        require(np.array_equal(training['candidate_ids'][:, 0], training['true_tokens']), 'Pares positivos incorretos')
        require(not (training['candidate_ids'][:, 1:] == training['true_tokens'][:, None]).any(), 'Negativo inclui rotulo positivo')
        observed = arrays(rd / 'observations.npz')
        require(set(observed) == {'timing_ms', 'cache_packed'}, 'Observacao carrega campo extra ou verdade')
        obs_cache = np.unpackbits(observed['cache_packed'], axis=1)[:, :width]
        replay = environment.observe(truth, seed, 7)
        require(np.array_equal(replay.timing, observed['timing_ms']) and np.array_equal(replay.cache, obs_cache), 'Replay do gerador divergiu')
        audit_arrays = arrays(rd / 'candidate_audit.npz')
        permutation = audit_arrays['observation_permutation']
        require(np.array_equal(permutation, rng_for(seed, 9).permutation(len(truth))), 'Permutacao de outro seed')
        predictions = arrays(rd / 'predictions.npz')
        require(np.array_equal(predictions['permuted_observation_pairs'], predictions['fusion_learned'][permutation]), 'Controle permutado divergente')
        require(np.all(predictions['frequency_prior'] == prior), 'Prior nao e o do treino')
        require(np.all(predictions['constant_signals'] == predictions['constant_signals'][0]), 'Controle constante varia por rotulo')
        learned = json.loads((rd / 'trcs_weights.json').read_text())
        weights = np.array(learned['weights'])
        require((weights >= 0).all() and abs(weights.sum() - 1) < 1e-12, 'Pesos invalidos')
        # Confere ranking sem importar as funcoes de reconstrucao.
        for index in sorted({0, len(truth) // 2, len(truth) - 1}):
            timing = np.exp(-.5 * ((observed['timing_ms'][index] - p['mu']) / p['sigma']) ** 2)
            denominator = np.linalg.norm(p['cache'], axis=1) * np.linalg.norm(obs_cache[index])
            cache = np.divide(p['cache'] @ obs_cache[index], denominator,
                              out=np.zeros(len(tokens)), where=denominator > 0)
            scores = weights[0] * timing + weights[1] * cache
            ranking = np.lexsort((tokens, -scores))[:5]
            require(np.array_equal(tokens[ranking], predictions['fusion_learned'][index]), 'Ranking independente divergente')
            scalar_positions += 1
        for start in range(0, len(truth), 128):
            stop = start + 128
            allowed = np.abs(observed['timing_ms'][start:stop, None] - p['mu']) <= cfg['trcs']['prune_sigma'] * p['sigma']
            fallback = ~allowed.any(axis=1)
            allowed[fallback] = True
            require(np.array_equal(allowed.sum(axis=1), audit_arrays['counts'][start:stop]), 'Contagem de candidatos divergente')
            require(np.array_equal(fallback, audit_arrays['fallback'][start:stop]), 'Fallback divergente')
            indices = np.searchsorted(tokens, truth[start:stop])
            require(np.array_equal(allowed[np.arange(len(indices)), indices], audit_arrays['true_candidate_survived'][start:stop]), 'Recall de candidatos divergente')
        report_metrics = json.loads((rd / 'metrics.json').read_text())
        for stored in report_metrics:
            name = stored['condition']
            pred = predictions[name]
            require(pred.shape == (len(truth), 5), 'Falhas omitidas do denominador')
            require(np.isin(pred, np.r_[-1, tokens]).all(), 'ID predito fora do vocabulario')
            hit1, hit5 = pred[:, 0] == truth, (pred == truth[:, None]).any(axis=1)
            per1 = np.add.reduceat(hit1, borders[:-1]) / lengths
            per5 = np.add.reduceat(hit5, borders[:-1]) / lengths
            exact = np.array([hit1[a:b].all() for a, b in zip(borders[:-1], borders[1:])])
            spans = [hit1[a + span['start']:a + span['start'] + span['length']].all()
                     for a, record in zip(borders[:-1], partition['test']) for span in record['protected_spans']]
            labels, inverse = np.unique(truth, return_inverse=True)
            support = np.bincount(inverse)
            recovered = np.bincount(inverse, weights=hit1)
            rare = np.array([frequencies[int(t)] < 10 for t in truth])
            values = {'top1_micro': hit1.mean(), 'top5_micro': hit5.mean(),
                      'top1_prompt_macro': per1.mean(), 'top5_prompt_macro': per5.mean(),
                      'prompt_exact_match': exact.mean(),
                      'synthetic_protected_span_exact_match': np.mean(spans),
                      'token_balanced_recall': np.mean(recovered / support),
                      'token_macro_f1_observed_true_classes': f1_score(truth, pred[:, 0], labels=labels, average='macro', zero_division=0),
                      'rare_token_top1': hit1[rare].mean()}
            require(stored['correct_top1'] == int(hit1.sum()) and stored['correct_top5'] == int(hit5.sum()), 'Contagem Top-k divergente')
            require(stored['rare_token_count'] == int(rare.sum()) and stored['protected_span_count'] == len(spans), 'Denominador de estrato divergente')
            for metric, value in values.items():
                close(float(value), stored[metric], f'Metrica divergente: {seed}/{name}/{metric}')
                recount_rows.append((name, metric, float(value)))
            per_prompt = json.loads((rd / f'per_prompt_{name}.json').read_text())
            for i, item in enumerate(per_prompt):
                close(item['top1'], per1[i], 'Macro prompt divergente')
                require(item['exact_match'] == bool(exact[i]), 'Exact match divergente')
            if name.startswith('fusion'):
                # Amostra deterministica; media completa ja preservada pelo avaliador.
                for i in sorted({0, len(lengths) // 2, len(lengths) - 1}):
                    a, b = borders[i:i + 2]
                    ned = distance(truth[a:b].tolist(), pred[a:b, 0].tolist()) / (b - a)
                    close(float(ned), per_prompt[i]['normalized_edit_distance'], 'Distancia de edicao divergente')
                    edit_positions += 1
        survived = audit_arrays['true_candidate_survived']
        round_reports.append({'seed': seed, 'passed': True, 'conditions_recounted': len(report_metrics),
                              'candidate_recall': float(survived.mean()),
                              'mean_candidates': float(audit_arrays['counts'].mean()),
                              'fallback_positions': int(audit_arrays['fallback'].sum()),
                              'top1_abstentions': {name: int((predictions[name][:, 0] == -1).sum()) for name in predictions}})
        print(json.dumps({'audited_seed': seed, 'passed': True}), flush=True)
    for name, metric in sorted({(n, m) for n, m, _ in recount_rows}):
        values = np.array([v for n, m, v in recount_rows if n == name and m == metric])
        stored = aggregate['conditions'][name][metric]
        close(float(values.mean()), stored['mean'], 'Media agregada divergente')
        if len(values) > 1:
            half = stats.t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            close(float(half), stored['ci95_half_width'], 'IC divergente')
    paired = []
    for left, right in [('fusion_learned', 'cache_only'), ('fusion_learned', 'fusion_fixed'),
                        ('fusion_learned_pruned', 'fusion_learned')]:
        for metric in ['top1_micro', 'prompt_exact_match', 'synthetic_protected_span_exact_match']:
            a = np.array([v for n, m, v in recount_rows if n == left and m == metric])
            b = np.array([v for n, m, v in recount_rows if n == right and m == metric])
            delta = a - b
            half = (float(stats.t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
                    if len(delta) > 1 else None)
            paired.append({'left': left, 'right': right, 'metric': metric,
                           'mean_difference': float(delta.mean()), 'ci95_half_width': half,
                           'unit': 'paired measurement seeds; fixed corpus and latent channel'})
    return {'passed': True, 'mode': cfg['mode'], 'physical_validation': False,
            'full_retokenization': True, 'group_membership_checked_against_records': True,
            'profile_estimates_recomputed_from_raw': True, 'test_observations_replayed_from_seeds': True,
            'macro_f1_compared_with_sklearn': True, 'files_checked': files_checked,
            'independent_rank_positions': scalar_positions, 'sampled_edit_distances': edit_positions,
            'metric_condition_round_checks': len(recount_rows),
            'test_positions_per_round': len(truth), 'test_prompts_per_round': len(lengths),
            'test_token_types': len(set(truth.tolist())), 'rounds': round_reports,
            'paired_comparisons': paired,
            'limits': ['No physical attack, model inference or empirical timing measured.',
                       'Edit distances checked on a deterministic sample, not all prompts.',
                       'Observation replay shares the frozen generator; metric and ranking checks do not import the evaluator.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Diretorio da execucao')
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), 'O relatorio ja existe; preserve e escolha outro nome')
    result = audit(args.output.resolve(), args.workspace.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'report': str(args.report)}), flush=True)


if __name__ == '__main__':
    main()
