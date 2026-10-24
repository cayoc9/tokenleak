"""Coordinate victim requests and record a timing-only view for later decoding."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

from tokenizers import Tokenizer
from .design import design
from .transport import observe


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def collect(cfg, model_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'config.json', cfg)
    tokenizer = Tokenizer.from_file(str(model_dir/'tokenizer.json'))
    planned = design(cfg, tokenizer)
    blind = tokenizer.encode(' object', add_special_tokens=False).ids
    if len(blind) != 1:
        raise ValueError('Blinding token contract failed')
    cfg['blinding_token_id'] = blind[0]
    dump(output/'config.json', cfg)
    dump(output/'design.json', {k: v for k, v in planned.items() if k != 'jobs'})
    with (output/'ground_truth.jsonl').open('w') as f:
        for job in planned['jobs']:
            f.write(json.dumps(job) + '\n')
    dump(output/'environment.json', {'python': platform.python_version(), 'machine': platform.machine(),
         'processor': platform.processor(), 'torch_threads': cfg['torch_threads'],
         'clock': str(time.get_clock_info('perf_counter')), 'transport': 'loopback TCP; TCP_NODELAY; padded requests and fixed-size response markers',
         'server_features_available_to_attacker': False, 'input_output_alignment': 'aggregate response predicts one candidate input token at a public fixed position'})
    observations = output/'observations.jsonl'
    completed = 0
    started = time.monotonic()
    for session in range(cfg['sessions']):
        modes = list(cfg['conditions'])
        random.Random(cfg['design_seed'] + 1000 + session).shuffle(modes)
        for condition in modes:
            tag = f'session_{session:02d}_{condition}'
            port_file, trace = output/(tag+'_port.json'), output/(tag+'_server_trace.jsonl')
            command = [sys.executable, '-m', 'tokenleak_observation.server', '--config', str(output/'config.json'),
                       '--model', str(model_dir), '--condition', condition,
                       '--port-file', str(port_file), '--trace', str(trace)]
            dump(output/(tag+'_command.json'), {'command': command})
            with (output/(tag+'_server.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 180
                    while not port_file.exists():
                        if process.poll() is not None:
                            raise RuntimeError(f'Server exited: {tag}; inspect server log')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Server startup')
                        time.sleep(.1)
                    port = json.loads(port_file.read_text())['port']
                    warm_ids = [tokenizer.token_to_id('<|endoftext|>')] * cfg['input_length']
                    for i in range(cfg['warmup_requests']):
                        observe(port, {'request_id': str(uuid.uuid4()), 'input_ids': warm_ids,
                                       'steps': cfg['output_steps'], 'purpose': 'warmup'},
                                cfg['output_steps'], cfg['request_body_bytes'])
                    for job in [j for j in planned['jobs'] if j['session'] == session]:
                        measured = observe(port, {'request_id': job['request_id'], 'input_ids': job['input_ids'],
                                           'steps': cfg['output_steps'], 'purpose': 'evaluation'},
                                           cfg['output_steps'], cfg['request_body_bytes'])
                        measured.update({'request_id': job['request_id'], 'session': session,
                                         'condition': condition, 'visibility': 'timing-only observer view'})
                        with observations.open('a') as f:
                            f.write(json.dumps(measured) + '\n')
                        completed += 1
                        if completed % 32 == 0:
                            status = {'state': 'collecting', 'session': session, 'condition': condition,
                                      'observations': completed, 'elapsed_seconds': time.monotonic()-started}
                            dump(output/'status.json', status); print(json.dumps(status), flush=True)
                    # Single-thread server completes previous trace before answering this barrier.
                    barrier = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
                    barrier.request('GET', '/health'); barrier.getresponse().read(); barrier.close()
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
    status = {'state': 'collection_complete', 'observations': completed, 'elapsed_seconds': time.monotonic()-started,
              'model_inference_measured': True, 'physical_cache_measured': False}
    dump(output/'status.json', status)
    print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    collect(json.loads(args.config.read_text()), args.model.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
