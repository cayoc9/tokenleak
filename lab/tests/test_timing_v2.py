"""Scientific boundaries: data identity, access, budget, ranking and denominators."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tokenleak_observation.transport import Observer, marker
from tokenleak_timing.contract import APP, dump, load_config, materialize, plan, validate, verify_model
from tokenleak_timing.evaluate import evaluate, feature, fit_profile, paired_interval, rank, uncertainty
from timing_experiment import package, run, verify_package


def config():
    return load_config(APP / 'configs/timing_smoke_cpu_v2.json')


def fixture(directory, missing=False):
    cfg = config()
    dump(directory / 'config.json', cfg)
    dump(directory / 'design.json', {'candidate_ids': [25, 262]})
    phases = ['profile', 'validation', 'new_session_new_context']
    dump(directory / 'plan.json', {'evaluation_requests': 36,
                                   'phases': [{'name': phase, 'requests_per_condition': 4} for phase in phases]})
    dump(directory / 'status.json', {'state': 'collection_complete'})
    truth, observations = [], []
    for condition in ['model', 'input_blinded', 'constant']:
        for phase in phases:
            for context in ['a', 'b']:
                for candidate in [25, 262]:
                    request_id = f'{condition}-{phase}-{context}-{candidate}'
                    truth.append({'request_id': request_id, 'condition': condition, 'phase': phase,
                                  'context_id': context, 'repeat': 0, 'class_id': candidate})
                    time_ns = 1000000 if candidate == 25 or condition != 'model' else 8000000
                    observations.append({'request_id': request_id, 'ok': True,
                                         'markers': [{'elapsed_ns': time_ns}]})
    (directory / 'ground_truth.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in truth))
    if missing:
        observations.pop()
    (directory / 'observations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in observations))


class TimingV2Contracts(unittest.TestCase):
    def test_full_budget_counts_samples_not_prompts(self):
        cfg = load_config(APP / 'configs/timing_8431_gpt2xl_v2.json')
        result = plan(cfg, materialize(cfg))
        self.assertEqual(result['calibration_samples_per_candidate'], 50)
        self.assertEqual(result['phases'][0]['requests_per_condition'], 421550)
        self.assertEqual(result['phases'][2]['requests_per_condition'], 134896)
        self.assertEqual(result['evaluation_requests'], 1770510)
        self.assertFalse(result['complete_prompt_reconstruction_evaluated'])
        self.assertFalse(result['model_inference_executed_by_plan'])

    def test_no_silent_change_to_physical_cache_access_or_dtype(self):
        cfg = config(); cfg['observation']['physical_cache'] = True
        with self.assertRaises(ValueError):
            validate(cfg)
        cfg = config(); cfg['execution']['dtype'] = 'float16'
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_public_context_families_are_held_out(self):
        cfg = config(); cfg['design']['test_contexts'] = ['C03-T00-000', 'C04-T02-000']
        with self.assertRaises(ValueError):
            materialize(cfg)

    def test_frozen_data_hash_rejects_changed_vocabulary(self):
        cfg = config(); cfg['data']['vocabulary_sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'; dump(path, cfg)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_feature_uses_first_completed_marker_not_privileged_metadata(self):
        observer = Observer(2)
        observer.feed(b'HTTP/1.0 200 OK\r\n\r\n' + marker(0) + marker(1), 8000000)
        clean = observer.finish(); altered = copy.deepcopy(clean)
        altered.update(class_id=999, input_ids=[123], session=999, context_id='secret',
                       host_visible_forward_and_argmax_ns=[1], body_reads=99999)
        self.assertEqual(feature(clean), 8.)
        self.assertEqual(feature(clean), feature(altered))
        self.assertEqual(list(inspect.signature(feature).parameters), ['observation'])

    def test_rank_is_stable_without_probability_or_label_inputs(self):
        profile = fit_profile([(25, 1.), (25, 1.1), (262, 8.), (262, 8.1)], [25, 262], .001)
        predicted, scores = rank([1.05, 8.05], profile)
        np.testing.assert_array_equal(predicted[:, 0], [25, 262])
        self.assertTrue(np.isfinite(scores).all())
        tied = {'candidate_ids': [25, 262], 'mean_ms': [1., 1.], 'sigma_ms': [.1, .1]}
        np.testing.assert_array_equal(rank([2.], tied)[0], [[25, 262]])

    def test_missing_request_cannot_shrink_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); fixture(directory, missing=True)
            with self.assertRaises(ValueError):
                evaluate(directory)
            self.assertFalse((directory / 'summary.json').exists())

    def test_conditional_resampling_and_paired_blocks_have_known_extremes(self):
        # Deliberately artificial decisions test statistics, not measured leakage.
        rows = [{'truth': candidate, 'prediction': candidate, 'correct': True,
                 'context_id': context, 'repeat': repeat}
                for repeat in range(8) for context in ['a', 'b'] for candidate in range(8)]
        result = uncertainty(rows, seed=42, resamples=99)
        self.assertEqual(result['ci95'], [1., 1.])
        self.assertEqual(result['one_sided_permutation_p'], .01)
        control = [{**row, 'correct': False} for row in rows]
        paired = paired_interval(rows, list(reversed(control)), seed=42, resamples=99)
        self.assertEqual(paired['difference'], 1.)
        self.assertEqual(paired['ci95'], [1., 1.])

    def test_heldout_labels_do_not_change_profiles_or_decisions(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = Path(first), Path(second); fixture(a); fixture(b)
            rows = [json.loads(line) for line in (b / 'ground_truth.jsonl').read_text().splitlines()]
            for row in rows:
                if row['phase'] == 'new_session_new_context':
                    row['class_id'] = 262 if row['class_id'] == 25 else 25
            (b / 'ground_truth.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            original, changed = evaluate(a), evaluate(b)
            self.assertEqual((a / 'model_profile.json').read_bytes(), (b / 'model_profile.json').read_bytes())
            self.assertEqual((a / 'predictions.jsonl').read_bytes(), (b / 'predictions.jsonl').read_bytes())
            self.assertEqual(original['conditions']['model']['new_session_new_context']['top1'], 1.)
            self.assertEqual(changed['conditions']['model']['new_session_new_context']['top1'], 0.)
            self.assertEqual(original['conditions']['constant']['new_session_new_context']['top1'], .5)

    def test_local_manifest_cannot_replace_authoritative_asset_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); model = directory / 'model'; model.mkdir()
            app = directory / 'app'; (app / 'assets/model_references').mkdir(parents=True)
            ref = {'model_id': 'fixture', 'revision': 'fixed',
                   'files': [{'path': 'asset', 'bytes': 4, 'sha256': '0' * 64}]}
            dump(app / 'assets/model_references/gpt2-small.json', ref)
            (model / 'asset').write_bytes(b'fake')
            dump(model / 'manifest.json', {**ref, 'files': []})
            with self.assertRaises(ValueError):
                verify_model(model, {'model': {'reference': 'gpt2-small'}}, app=app)

    def test_large_run_requires_declared_budget_before_loading_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'not-created'
            with self.assertRaisesRegex(ValueError, 'evaluation requests'):
                run(APP / 'configs/timing_8431_gpt2xl_v2.json', Path('/no-model'), output, 2000)
            self.assertFalse(output.exists())

    def test_bundle_is_checksummed_and_excludes_runs_and_private_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'handoff.tar.gz'
            created = package(path); verified = verify_package(path)
            self.assertEqual(created['sha256'], verified['sha256'])
            import tarfile
            with tarfile.open(path) as archive:
                names = archive.getnames()
            self.assertFalse(any('/runs/' in name or '/registros/' in name or '/results/' in name
                                 or '.ssh' in name or '/Thesis_' in name for name in names))


if __name__ == '__main__':
    unittest.main()
