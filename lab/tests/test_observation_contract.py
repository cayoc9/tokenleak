import copy
import inspect
import hashlib
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np
from tokenizers import Tokenizer

from tokenleak_observation.design import design
from tokenleak_observation.evaluate import features, interval_and_null, metrics
from tokenleak_observation.transport import Observer, marker
from observe import verify_assets


class ObservationContracts(unittest.TestCase):
    def test_checkpoint_hash_and_registered_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = b'fixture, not real weights'
            (root/'weights').write_bytes(body)
            reference = {'model_id':'fixture', 'revision':'frozen',
                         'files':[{'path':'weights','bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}]}
            (root/'manifest.json').write_text(json.dumps(reference))
            cfg = {'model_id':'fixture','model_revision':'frozen','device':'cpu','dtype':'float32',
                   'sessions':3,'candidate_words':list(range(8)),'context_prefixes':list(range(4))}
            verify_assets(root, cfg)
            (root/'weights').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                verify_assets(root, cfg)

    def test_coalesced_markers_are_one_socket_read(self):
        observer = Observer(3)
        observer.feed(b'HTTP/1.0 200 OK\r\n\r\n' + marker(0) + marker(1), 1000000)
        observer.feed(marker(2), 2000000)
        result = observer.finish()
        self.assertEqual(result['body_reads'], 2)
        self.assertEqual(result['max_markers_per_read'], 2)
        self.assertEqual(result['markers'][0]['elapsed_ns'], result['markers'][1]['elapsed_ns'])
        self.assertEqual(result['body_bytes'], 3 * 64)

    def test_feature_whitelist_is_independent_of_truth_metadata(self):
        observer = Observer(3)
        observer.feed(b'HTTP/1.0 200 OK\r\n\r\n'+marker(0), 1000000)
        observer.feed(marker(1)+marker(2), 2000000)
        a = observer.finish(); b = copy.deepcopy(a)
        b.update(class_id='SENSITIVE', input_ids=[123], request_id='different', session=99)
        np.testing.assert_array_equal(features(a), features(b))
        self.assertEqual(list(inspect.signature(features).parameters), ['observation'])

    def test_incomplete_stream_cannot_shrink_denominator(self):
        observer = Observer(3)
        observer.feed(b'HTTP/1.0 200 OK\r\n\r\n'+marker(0), 1000000)
        with self.assertRaises(ValueError):
            observer.finish()

    def test_balanced_fixed_slot_design_has_no_class_in_request_id(self):
        app = Path(__file__).resolve().parents[1]
        cfg = json.loads((app/'configs/observacao_smoke_v1.json').read_text())
        tokenizer = Tokenizer.from_file(str(app/'assets/tokenizer/gpt2-xl/tokenizer.json'))
        data = design(cfg, tokenizer)
        self.assertEqual(len(data['jobs']), 3*4*8)
        self.assertEqual(len({r['request_id'] for r in data['jobs']}), len(data['jobs']))
        for session in range(3):
            for context in ['X00', 'X01', 'X02', 'X03']:
                rows = [r for r in data['jobs'] if r['session']==session and r['context_id']==context]
                self.assertEqual(len(rows), 8)
                self.assertEqual(len({r['class_id'] for r in rows}), 8)
                for row in rows:
                    self.assertEqual(len(row['input_ids']), 16)

    def test_permutation_null_detects_perfect_candidate_classification(self):
        truth = np.tile(np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']), 8)
        rows = [{'context_id': 'X02', 'repeat': i//8} for i in range(len(truth))]
        result = interval_and_null(truth, truth.copy(), rows, 42, 200)
        self.assertEqual(result['cluster_bootstrap_ci95'], [1., 1.])
        self.assertLess(result['conditional_permutation_p_one_sided'], .01)
        self.assertAlmostEqual(result['permutation_null_mean'], 1/8, delta=.03)


if __name__ == '__main__':
    unittest.main()
