import copy
import json
import inspect
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tokenleak_lab.corpus import build, load, audit_corpus
from tokenleak_lab.metrics import evaluate, edit_distance
from tokenleak_lab.reconstruction import topk, Reconstructor, fit_weights
from tokenleak_lab.signals import Profiles, Observations, SyntheticEnvironment
from tokenleak_lab.pipeline import scalar_reference


class ScientificContracts(unittest.TestCase):
    def test_collision_sensitivity_really_shares_cache_patterns(self):
        app = Path(__file__).resolve().parents[1]
        cfg = json.loads((app/'configs/simulacao_v1.json').read_text())['signals']
        cfg['cache_fingerprint_groups'] = 2
        environment = SyntheticEnvironment(np.arange(16), cfg)
        np.testing.assert_array_equal(environment.patterns[0], environment.patterns[2])
        self.assertFalse(np.array_equal(environment.patterns[0], environment.patterns[1]))
        cfg['cache_fingerprint_groups'] = None
        distinct = SyntheticEnvironment(np.arange(16), cfg)
        self.assertEqual(len({row.tobytes() for row in distinct.patterns}), 16)

    def test_decoder_has_no_truth_and_paired_permutation_is_equivalent(self):
        self.assertNotIn('labels', inspect.signature(Reconstructor.features).parameters)
        self.assertNotIn('truth', inspect.signature(Reconstructor.rank_features).parameters)
        p = Profiles(np.arange(8), np.zeros(8), np.ones(8), np.eye(8))
        r = Reconstructor(p)
        o = Observations(np.zeros(8), np.eye(8))
        t, c = r.features(o)
        direct, _, _, _ = r.rank_features(t, c, [0, 1])
        perm = np.roll(np.arange(8), 1)
        t, c = r.features(Observations(o.timing[perm], o.cache[perm]))
        shuffled, _, _, _ = r.rank_features(t, c, [0, 1])
        np.testing.assert_array_equal(shuffled, direct[perm])
        self.assertTrue(np.all(direct[:, 0] == p.token_ids))
        self.assertTrue(np.all(shuffled[:, 0] != p.token_ids))

    def test_weight_learning_identifies_informative_channel(self):
        p = Profiles(np.arange(16), np.zeros(16), np.ones(16), np.eye(16))
        labels = np.tile(p.token_ids, 4)
        o = Observations(np.zeros(len(labels)), np.eye(16)[labels])
        cfg = {'uniform_negatives_per_position': 8, 'hard_timing_negatives_per_position': 4,
               'nonnegative_logistic_l2': .001}
        weights, _, _ = fit_weights(p, o, labels, cfg, np.random.default_rng(42))
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(weights.sum()), 1.)

    def test_accuracy_can_hide_a_secret_failure(self):
        records = [{'prompt_id': 'a', 'category_id': 'C01', 'token_ids': [1]*9+[2],
                    'protected_spans': [{'start': 9, 'length': 1}]}]
        predictions = np.tile([1, 3, 4, 5, 6], (10, 1))
        m, _ = evaluate(records, predictions, {'1': 90, '2': 1})
        self.assertEqual(m['top1_micro'], .9)
        self.assertEqual(m['synthetic_protected_span_exact_match'], 0)
        self.assertEqual(m['prompt_exact_match'], 0)
        self.assertEqual(m['token_balanced_recall'], .5)

    def test_partial_secret_leak_with_low_global_accuracy(self):
        records = [{'prompt_id': 'a', 'category_id': 'C01', 'token_ids': list(range(10)),
                    'protected_spans': [{'start': 7, 'length': 3}]}]
        p = np.tile([99, 90, 91, 92, 93], (10, 1)); p[7:, 0] = [7, 8, 9]
        m, _ = evaluate(records, p, {str(i): 1 for i in range(10)})
        self.assertEqual(m['top1_micro'], .3)
        self.assertEqual(m['synthetic_protected_span_exact_match'], 1)

    def test_macro_prompt_differs_from_micro_token(self):
        records = [{'prompt_id': 'a', 'category_id': 'C01', 'token_ids': [1, 1]},
                   {'prompt_id': 'b', 'category_id': 'C01', 'token_ids': [2]*8}]
        p = np.tile([1, 3, 4, 5, 6], (10, 1))
        m, _ = evaluate(records, p, {'1': 2, '2': 8})
        self.assertEqual(m['top1_micro'], .2)
        self.assertEqual(m['top1_prompt_macro'], .5)

    def test_edit_distance_handles_an_insertion(self):
        self.assertEqual(edit_distance([1, 2, 3], [1, 9, 2, 3]), 1)

    def test_topk_boundary_ties_choose_lowest_token_ids(self):
        ids = np.array([8, 2, 5, 1, 9, 7, 3])
        predictions, _ = topk(np.array([[0., 1., 1., 1., 1., 1., 1.]]), ids)
        self.assertEqual(predictions[0].tolist(), [1, 2, 3, 5, 7])

    def test_vectorized_rank_equals_independent_scalar(self):
        p = Profiles(np.arange(6), np.array([4., 5., 8., 12., 16., 20.]), np.ones(6), np.eye(6))
        o = Observations(np.array([8.]), np.array([[0, 0, 1, 0, 0, 0]], dtype=bool))
        r = Reconstructor(p); t, c = r.features(o)
        pred, _, _, _ = r.rank_features(t, c, [.55, .45])
        self.assertEqual(pred[0].tolist(), scalar_reference(p, o, [.55, .45]))
        self.assertEqual(pred[0, 0], 2)

    def test_pruning_can_exclude_correct_cache_candidate(self):
        p = Profiles(np.arange(6), np.arange(6, dtype=float)*10, np.ones(6), np.eye(6))
        o = Observations(np.array([0.]), np.array([[0, 0, 0, 0, 0, 1]], dtype=bool))
        r = Reconstructor(p); t, c = r.features(o)
        full, _, _, _ = r.rank_features(t, c, [0., 1.])
        pruned, _, _, _ = r.rank_features(t, c, [0., 1.], o.timing, 3.)
        self.assertEqual(full[0, 0], 5)
        self.assertEqual(pruned[0, 0], 0)

    def test_wrong_prediction_count_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            evaluate([{'token_ids': [1, 2]}], np.ones((1, 5)), {'1': 1, '2': 1})

    def test_corpus_determinism_frequency_split_and_immutability(self):
        app = Path(__file__).resolve().parents[1]
        cfg = json.loads((app/'configs/simulacao_v1.json').read_text())
        cfg['dataset']['n_prompts'] = 200
        cfg['dataset']['unique_token_types'] = 512
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            build(cfg, a, os.environ['TOKENLEAK_WORKSPACE'], app)
            build(cfg, b, os.environ['TOKENLEAK_WORKSPACE'], app)
            self.assertEqual((a/'corpus.jsonl').read_bytes(), (b/'corpus.jsonl').read_bytes())
            self.assertEqual((a/'splits.json').read_bytes(), (b/'splits.json').read_bytes())
            self.assertTrue(audit_corpus(a, cfg)['passed'])
            with self.assertRaises(FileExistsError):
                build(cfg, a, os.environ['TOKENLEAK_WORKSPACE'], app)
            corrupt = json.loads((a/'splits.json').read_text())
            corrupt['members']['test'].append(corrupt['members']['train'][0])
            (a/'splits.json').write_text(json.dumps(corrupt))
            with self.assertRaises(ValueError):
                load(a)


if __name__ == '__main__':
    unittest.main()
