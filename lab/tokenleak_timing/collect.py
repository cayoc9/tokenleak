"""Fresh phases/processes, fixed payload, opaque IDs, and raw timings preserved."""
import argparse
import http.client
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

from tokenleak_observation.transport import observe
from .contract import dump, jobs, load_config, materialize, plan


def collect(cfg, model, output):
    data = materialize(cfg)
    planned = plan(cfg, data)
    dump(output / 'design.json', {key: value for key, value in data.items() if key != 'corpus'})
    dump(output / 'plan.json', planned)
    count = 0
    started = time.monotonic()
    execution = cfg['execution']
    with (output / 'ground_truth.jsonl').open('x') as truth, (output / 'observations.jsonl').open('x') as observations:
        for phase in data['phases']:
            conditions = list(cfg['design']['conditions'])
            random.Random(cfg['design']['seed'] + phase['session'] + 100).shuffle(conditions)
            for condition in conditions:
                tag = f"{phase['session']:02d}_{condition}"
                port_file = output / (tag + '_port.json')
                command = [sys.executable, '-B', '-m', 'tokenleak_timing.server',
                           '--config', str(output / 'config.json'), '--model', str(model),
                           '--condition', condition, '--port-file', str(port_file),
                           '--trace', str(output / (tag + '_privileged.jsonl'))]
                with (output / (tag + '_server.log')).open('x') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        deadline = time.monotonic() + execution['startup_seconds']
                        while not port_file.exists():
                            if process.poll() is not None:
                                raise RuntimeError('Victim process failed; inspect ' + tag + '_server.log')
                            if time.monotonic() > deadline:
                                raise TimeoutError('Victim startup deadline')
                            time.sleep(0.05)
                        ready = json.loads(port_file.read_text())
                        dump(output / (tag + '_runtime.json'), ready['runtime'])
                        port = ready['port']
                        for unused in range(execution['warmup_requests']):
                            observe(port, {'request_id': str(uuid.uuid4()), 'input_ids': [50256] * 16,
                                           'steps': execution['output_steps'], 'purpose': 'warmup'},
                                    execution['output_steps'], execution['request_bytes'])
                        for job in jobs(cfg, data, phase, condition):
                            if time.monotonic() - started > execution['max_seconds']:
                                raise TimeoutError('Registered collection wall-time budget')
                            truth.write(json.dumps(job) + '\n'); truth.flush()
                            measured = observe(port, {'request_id': job['request_id'], 'input_ids': job['input_ids'],
                                                       'steps': execution['output_steps'], 'purpose': 'evaluation'},
                                               execution['output_steps'], execution['request_bytes'])
                            measured.update(request_id=job['request_id'], phase=phase['name'],
                                            condition=condition, visibility='instrumental timing-only view')
                            observations.write(json.dumps(measured) + '\n'); observations.flush()
                            count += 1
                            if count % 32 == 0:
                                status = {'state': 'collecting', 'observations': count,
                                          'elapsed_seconds': time.monotonic() - started}
                                dump(output / 'status.json', status)
                                print(json.dumps(status), flush=True)
                        barrier = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
                        barrier.request('GET', '/health')
                        barrier.getresponse().read(); barrier.close()
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
    if count != planned['evaluation_requests']:
        raise ValueError('Collected denominator differs from frozen plan')
    dump(output / 'status.json', {'state': 'collection_complete', 'observations': count,
         'elapsed_seconds': time.monotonic() - started, 'role': cfg['role'],
         'model_inference_measured': True, 'physical_cache_measured': False,
         'complete_prompt_reconstruction_evaluated': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(load_config(args.config), args.model.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
