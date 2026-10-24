"""Orquestracao congelada da simulacao, controles e conferencia dos artefatos."""
from collections import Counter
import csv
import datetime
import json
from pathlib import Path
import time

import numpy as np
from scipy import stats

from . import corpus
from .io import write_json, manifest, check_manifest, digest
from .signals import SyntheticEnvironment, Observations, rng_for
from .reconstruction import Reconstructor, fit_weights
from .metrics import evaluate


CONDITIONS = ['frequency_prior', 'uniform_random', 'timing_only', 'cache_only',
              'fusion_fixed', 'fusion_learned', 'fusion_learned_pruned',
              'constant_signals', 'permuted_observation_pairs']


def flatten(records):
    return np.array([t for r in records for t in r['token_ids']], dtype=np.int32)


def subset(records, members):
    wanted = set(members)
    return [r for r in records if r['prompt_id'] in wanted]


def scalar_reference(profiles, observation, weights):
    """Equacoes independentes, sem a matriz vetorizada e sem consultar rotulos."""
    scores = []
    c = observation.cache[0].astype(float)
    for i, token in enumerate(profiles.token_ids):
        timing = float(np.exp(-0.5*((observation.timing[0]-profiles.mu[i])/profiles.sigma[i])**2))
        denominator = float(np.linalg.norm(c)*np.linalg.norm(profiles.cache[i]))
        cache = float(np.dot(c, profiles.cache[i])/denominator) if denominator else 0.
        scores.append(((weights[0]*timing+weights[1]*cache)/sum(weights), int(token)))
    return [token for _, token in sorted(scores, key=lambda item: (-item[0], item[1]))[:5]]


def run(config, output):
    output = Path(output)
    result_dir = output / 'results'
    result_dir.mkdir(exist_ok=False)
    records, split, vocabulary = corpus.load(output / 'corpus')
    train = subset(records, split['members']['train'])
    validation = subset(records, split['members']['validation'])
    test = subset(records, split['members']['test'])
    train_tokens, validation_tokens, test_tokens = map(flatten, [train, validation, test])
    np.savez_compressed(result_dir / 'evaluation_reference.npz', true_tokens=test_tokens,
                        lengths=np.array([len(r['token_ids']) for r in test]))
    environment = SyntheticEnvironment(vocabulary['token_ids'], config['signals'])
    frequencies = Counter(train_tokens.tolist())
    prior = sorted(environment.tokens.tolist(), key=lambda t: (-frequencies[t], t))[:5]
    all_metrics = []
    started = time.monotonic()
    for seed in config['evaluation']['round_seeds']:
        rd = result_dir / f'round_{seed:02d}'
        rd.mkdir()
        round_started = time.monotonic()
        write_json(result_dir / 'status.json', {'state': 'running', 'active_seed': seed,
                    'complete_rounds': len(all_metrics)//len(CONDITIONS)})
        print(json.dumps({'event': 'round_start', 'seed': seed}), flush=True)
        profiles = environment.profile(seed, rd)
        reconstructor = Reconstructor(profiles)
        train_rng = rng_for(seed, 3)
        indices = train_rng.choice(len(train_tokens), min(len(train_tokens), config['trcs']['training_positions']), replace=False)
        training_observations = environment.observe(train_tokens[indices], seed, 4)
        weights, learned, candidate_pairs = fit_weights(profiles, training_observations, train_tokens[indices], config['trcs'], train_rng)
        write_json(rd / 'trcs_weights.json', learned)
        np.savez_compressed(rd / 'weight_training.npz', source_positions=indices,
                            true_tokens=train_tokens[indices], candidate_ids=candidate_pairs,
                            timing=training_observations.timing,
                            cache_packed=np.packbits(training_observations.cache, axis=1))
        val_indices = rng_for(seed, 5).choice(len(validation_tokens), min(len(validation_tokens), config['trcs']['validation_positions']), replace=False)
        val_obs = environment.observe(validation_tokens[val_indices], seed, 6)
        val_hits = []
        for a in range(0, len(val_indices), config['evaluation']['batch_size']):
            b = a + config['evaluation']['batch_size']
            t, c = reconstructor.features(Observations(val_obs.timing[a:b], val_obs.cache[a:b]))
            pred, _, _, _ = reconstructor.rank_features(t, c, weights)
            val_hits.extend((pred[:, 0] == validation_tokens[val_indices[a:b]]).tolist())
        write_json(rd / 'validation.json', {'partition': 'validation', 'positions': len(val_hits),
                   'top1': float(np.mean(val_hits)), 'used_to_select_hyperparameters': False,
                   'weights_trained_on_test': False})
        obs = environment.observe(test_tokens, seed, 7)
        np.savez_compressed(rd / 'observations.npz', timing_ms=obs.timing,
                            cache_packed=np.packbits(obs.cache, axis=1))
        predictions = {name: np.empty((len(test_tokens), 5), dtype=np.int32) for name in CONDITIONS}
        primary_scores = {}
        counts = np.empty(len(test_tokens), dtype=np.int32)
        fallback = np.empty(len(test_tokens), dtype=bool)
        constant_t, constant_c = reconstructor.features(Observations(np.array([10.]), np.zeros((1, config['signals']['cache_width']), dtype=bool)))
        constant_pred, _, _, _ = reconstructor.rank_features(constant_t, constant_c, weights)
        predictions['constant_signals'][:] = constant_pred[0]
        predictions['frequency_prior'][:] = prior
        uniform_rng = rng_for(seed, 8)
        uniform_indices = uniform_rng.integers(0, len(environment.tokens), (len(test_tokens), 5))
        duplicate_rows = np.array([len(set(row)) != 5 for row in uniform_indices])
        for row in np.flatnonzero(duplicate_rows):
            uniform_indices[row] = uniform_rng.choice(len(environment.tokens), 5, replace=False)
        predictions['uniform_random'][:] = environment.tokens[uniform_indices]
        for name in ['fusion_fixed', 'fusion_learned', 'fusion_learned_pruned']:
            primary_scores[name] = np.empty(len(test_tokens))
        for a in range(0, len(test_tokens), config['evaluation']['batch_size']):
            b = min(a + config['evaluation']['batch_size'], len(test_tokens))
            timing, cache = reconstructor.features(Observations(obs.timing[a:b], obs.cache[a:b]))
            for name, w in [('timing_only', [1., 0.]), ('cache_only', [0., 1.]),
                            ('fusion_fixed', config['trcs']['fixed_weights']), ('fusion_learned', weights)]:
                pred, scores, _, _ = reconstructor.rank_features(timing, cache, w)
                predictions[name][a:b] = pred
                if name in primary_scores:
                    primary_scores[name][a:b] = scores
            pred, scores, candidate_count, did_fallback = reconstructor.rank_features(
                timing, cache, weights, obs.timing[a:b], config['trcs']['prune_sigma'])
            predictions['fusion_learned_pruned'][a:b] = pred
            primary_scores['fusion_learned_pruned'][a:b] = scores
            counts[a:b], fallback[a:b] = candidate_count, did_fallback
        permutation = rng_for(seed, 9).permutation(len(test_tokens))
        predictions['permuted_observation_pairs'][:] = predictions['fusion_learned'][permutation]
        # An identical row-wise decoder means paired signal permutation equals row permutation.
        reference_ok = []
        for index in [0, len(test_tokens)//2, len(test_tokens)-1]:
            ref = scalar_reference(profiles, Observations(obs.timing[index:index+1], obs.cache[index:index+1]), weights)
            reference_ok.append(ref == predictions['fusion_learned'][index].tolist())
        if not all(reference_ok):
            raise ValueError('Scalar reference and vectorized reconstruction disagree')
        write_json(rd / 'scalar_conference.json', {'positions_compared': 3, 'top5_exactly_equal': reference_ok})
        true_profile_index = np.searchsorted(profiles.token_ids, test_tokens)
        survived = np.abs(obs.timing-profiles.mu[true_profile_index]) <= config['trcs']['prune_sigma']*profiles.sigma[true_profile_index]
        survived |= fallback
        np.savez_compressed(rd / 'predictions.npz', **predictions)
        np.savez_compressed(rd / 'candidate_audit.npz', counts=counts, fallback=fallback,
                            true_candidate_survived=survived, observation_permutation=permutation,
                            **{name+'_top1_score': value for name, value in primary_scores.items()})
        round_metrics = []
        for name in CONDITIONS:
            summary, per_prompt = evaluate(test, predictions[name], vocabulary['frequencies'],
                                           with_edit=name.startswith('fusion'))
            summary.update({'condition': name, 'seed': seed})
            round_metrics.append(summary)
            write_json(rd / f'per_prompt_{name}.json', per_prompt)
        write_json(rd / 'metrics.json', round_metrics)
        write_json(rd / 'round_manifest.json', {'seed': seed, 'elapsed_seconds': time.monotonic()-round_started,
                   'profiling_policy': config['signals']['profiling_policy'],
                   'known_token_profile_requests': len(environment.tokens)*config['signals']['profile_samples_per_type'],
                   'candidate_recall': float(survived.mean()), 'fallback_positions': int(fallback.sum()),
                   'weights': weights.tolist(), 'signal_mode': 'fully_simulated',
                   'alignment': 'oracle_input_positions',
                   'independent_rng_components': {'profiling': 2, 'training_selection': 3, 'training_signals': 4,
                                                'validation_selection': 5, 'validation_signals': 6,
                                                'test_signals': 7, 'uniform_control': 8, 'permutation_control': 9}})
        write_json(rd / 'manifest.json', manifest(rd))
        all_metrics.extend(round_metrics)
        print(json.dumps({'event': 'round_complete', 'seed': seed, 'seconds': time.monotonic()-round_started,
                          'weights': weights.tolist(), 'fusion_top1': round_metrics[5]['top1_micro'],
                          'protected_span_exact': round_metrics[5]['synthetic_protected_span_exact_match']}), flush=True)
    write_json(result_dir / 'metrics_per_run.json', all_metrics)
    aggregate = summarize(all_metrics, config)
    write_json(result_dir / 'summary.json', aggregate)
    columns = ['condition', 'seed', 'positions', 'top1_micro', 'top5_micro', 'top1_prompt_macro',
               'prompt_exact_match', 'synthetic_protected_span_exact_match', 'normalized_edit_distance',
               'token_balanced_recall', 'token_macro_f1_observed_true_classes', 'rare_token_top1']
    with (result_dir / 'metrics_per_run.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(all_metrics)
    plot(aggregate, result_dir)
    write_json(result_dir / 'status.json', {'state': 'complete', 'complete_rounds': len(config['evaluation']['round_seeds']),
               'elapsed_seconds': time.monotonic()-started, 'mode': 'fully_simulated'})
    write_json(result_dir / 'manifest.json', manifest(result_dir))
    return aggregate


def summarize(rows, config):
    result = {'mode': 'fully_simulated', 'alignment': 'oracle_input_positions',
              'interval_scope': config['evaluation']['interval_unit'], 'conditions': {},
              'paper_numbers_are_references_not_parameter_targets': True}
    metrics = ['top1_micro', 'top5_micro', 'top1_prompt_macro', 'top5_prompt_macro',
               'prompt_exact_match', 'synthetic_protected_span_exact_match', 'normalized_edit_distance',
               'token_balanced_recall', 'token_macro_f1_observed_true_classes', 'rare_token_top1']
    for condition in CONDITIONS:
        samples = [r for r in rows if r['condition'] == condition]
        item = {'rounds': len(samples)}
        for metric in metrics:
            values = np.array([r[metric] for r in samples if r[metric] is not None], dtype=float)
            if not len(values):
                continue
            half = float(stats.t.ppf(0.975, len(values)-1)*values.std(ddof=1)/len(values)**.5) if len(values) > 1 else None
            item[metric] = {'mean': float(values.mean()), 'ci95_half_width': half}
        result['conditions'][condition] = item
    return result


def plot(aggregate, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(aggregate['conditions'])
    labels = ['Prior', 'Random', 'Timing', 'Cache', 'Fixed fusion', 'Learned fusion', 'Pruned fusion', 'Constant', 'Permuted']
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    x = np.arange(len(names))
    for axis, metrics in zip(axes, [('top1_micro', 'top5_micro'), ('prompt_exact_match', 'synthetic_protected_span_exact_match')]):
        for j, metric in enumerate(metrics):
            values = [100*aggregate['conditions'][name][metric]['mean'] for name in names]
            errors = [100*(aggregate['conditions'][name][metric]['ci95_half_width'] or 0) for name in names]
            axis.bar(x+(j-.5)*.35, values, .35, yerr=errors, capsize=2, label=metric)
        axis.set_xticks(x, labels, rotation=15); axis.set_ylabel('Percent'); axis.set_ylim(0, 105)
        axis.grid(axis='y', alpha=.2); axis.legend(fontsize=8)
    fig.suptitle('SPC5K reconstructed v1 — synthetic signals and known input positions\nProtected spans are synthetic keyword proxies; no physical attack measured', fontsize=11)
    for extension in ['png', 'pdf']:
        fig.savefig(directory/f'methodology_metrics.{extension}', dpi=150)
    plt.close(fig)


def verify(config, output):
    output = Path(output)
    records, split, vocabulary = corpus.load(output / 'corpus')
    test = subset(records, split['members']['test'])
    truth = flatten(test)
    reports = []
    files_checked = check_manifest(output / 'results')
    for seed in config['evaluation']['round_seeds']:
        rd = output / 'results' / f'round_{seed:02d}'
        check_manifest(rd)
        stored = json.loads((rd / 'metrics.json').read_text())
        with np.load(rd / 'predictions.npz', allow_pickle=False) as pred:
            for item in stored:
                name = item['condition']
                top1 = int((pred[name][:, 0] == truth).sum())
                top5 = int((pred[name] == truth[:, None]).any(axis=1).sum())
                if top1 != item['correct_top1'] or top5 != item['correct_top5'] or len(truth) != item['positions']:
                    raise ValueError('Independent count conference failed')
                if abs(top1/len(truth) - item['top1_micro']) > 1e-14:
                    raise ValueError('Micro denominator conference failed')
            perm = np.load(rd / 'candidate_audit.npz', allow_pickle=False)['observation_permutation']
            if not np.array_equal(pred['permuted_observation_pairs'], pred['fusion_learned'][perm]):
                raise ValueError('Paired permutation conference failed')
        reports.append({'seed': seed, 'conditions_recounted': len(stored), 'passed': True})
    return {'passed': True, 'manifest_files_checked': files_checked, 'rounds': reports,
            'evaluated_positions_per_round': len(truth), 'full_corpus_prompts': len(records),
            'vocabulary_size': len(vocabulary['token_ids']), 'physical_validation': False}
