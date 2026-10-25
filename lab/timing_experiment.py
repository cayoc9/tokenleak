"""Portable v2 runner: inspect, plan, verified download, measured run, and handoff bundle."""
import argparse
import datetime
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request

from tokenleak_timing.contract import APP, dump, load_config, materialize, plan, sha, verify_model


def source_files():
    """Explicit public export; no workstation records, previous runs, or model weights."""
    selected = {}
    for directory in ['tokenleak_timing']:
        for path in (APP / directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.md'} and '__pycache__' not in path.parts:
                selected[str(path.relative_to(APP))] = path
    for pattern in ['configs/timing_*_v2.json', 'tests/test_timing_v2.py',
                    'docs/TIMING_PROTOCOL_V2.md', 'docs/RUN_TIMING_V2.md',
                    'docs/RESULTS_TIMING_V2_GPU.md']:
        for path in APP.glob(pattern):
            selected[str(path.relative_to(APP))] = path
    for name in ['timing_experiment.py', 'audit_timing_v2.py', 'probe_host.py', 'requirements-timing.lock', 'requirements-timing-gpu.lock',
                 'tokenleak_observation/__init__.py', 'tokenleak_observation/transport.py',
                 'data/corpus-v1/corpus.jsonl', 'data/corpus-v1/vocabulary.json',
                 'data/corpus-v1/splits.json']:
        selected[name] = APP / name
    for path in (APP / 'assets/model_references').glob('*.json'):
        selected[str(path.relative_to(APP))] = path
    for path in (APP / 'assets/cmudict').glob('*'):
        if path.is_file() and ('licen' in path.name.lower() or path.name == 'README'):
            selected[str(path.relative_to(APP))] = path
    return dict(sorted(selected.items()))


def prepare_model(reference_name, output):
    reference = json.loads((APP / 'assets/model_references' / (reference_name + '.json')).read_text())
    if reference['model_id'] not in {'openai-community/gpt2', 'openai-community/gpt2-xl'}:
        raise ValueError('Unsupported public model repository')
    output.mkdir(parents=True, exist_ok=True)
    for item in reference['files']:
        name = item['path']
        if Path(name).name != name:
            raise ValueError('Model paths must be flat safe asset names')
        expected = ('https://huggingface.co/' + reference['model_id'] + '/resolve/' +
                    reference['revision'] + '/' + name)
        if item['source_url'] != expected:
            raise ValueError('Only immutable URLs from the registered public repository are allowed')
        target = output / name
        if target.exists():
            if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
                raise ValueError('Existing model asset differs; no overwrite: ' + name)
            print('Verified ' + name, flush=True)
            continue
        part = target.with_name(target.name + '.part')
        # Preserve interrupted downloads. Select a fresh destination or handle the partial explicitly.
        with part.open('xb') as stream, urllib.request.urlopen(expected, timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
        if part.stat().st_size != item['bytes'] or sha(part) != item['sha256']:
            raise ValueError('Download integrity failed: ' + name)
        part.rename(target)
        print('Downloaded and verified ' + name, flush=True)
    dump(output / 'manifest.json', reference)


def check_runtime(cfg):
    required = {'torch': '2.3.1', 'transformers': '4.41.2', 'numpy': '1.26.0'}
    actual = {}
    for name, expected in required.items():
        actual[name] = importlib.metadata.version(name)
        if actual[name].split('+')[0] != expected:
            raise ValueError(f'Use an isolated validated runtime: {name}={actual[name]}, expected {expected}')
    if cfg['execution']['device'].startswith('cuda:'):
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Explicit CUDA execution requested, but CUDA is unavailable')
    return actual


def run(config, model, output, max_requests):
    cfg = load_config(config)
    data = materialize(cfg)
    planned = plan(cfg, data)
    if planned['evaluation_requests'] > max_requests:
        raise ValueError(f"Plan has {planned['evaluation_requests']} evaluation requests; "
                         f"explicit execution budget is {max_requests}. Inspect plan before increasing it.")
    if output.exists():
        raise ValueError('Output exists; every execution needs a fresh directory')
    versions = check_runtime(cfg)
    reference = verify_model(model, cfg)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    files = source_files()
    for relative, source in files.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dump(output / 'config.json', cfg)
    dump(output / 'plan.json', planned)
    dump(output / 'model_manifest.json', reference)
    dump(output / 'source_manifest.json', {relative: sha(snapshot / relative) for relative in files})
    from probe_host import inventory
    dump(output / 'environment.json', inventory())
    execution = {'state': 'prepared', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'config_sha256': sha(output / 'config.json'), 'runtime_versions': versions,
                 'role': cfg['role'], 'parent_protocol_v1_modified': False}
    dump(output / 'execution.json', execution)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': str(cfg['execution']['threads']),
           'MKL_NUM_THREADS': str(cfg['execution']['threads']), 'PYTHONPATH': str(snapshot)}
    commands = [
        ['-m', 'tokenleak_timing.collect', '--config', str(output / 'config.json'),
         '--model', str(model), '--output', str(output)],
        ['-m', 'tokenleak_timing.evaluate', '--output', str(output)]]
    try:
        for stage, arguments in zip(['collection', 'evaluation'], commands):
            command = [sys.executable, '-B'] + arguments
            execution[stage + '_command'] = command
            dump(output / 'execution.json', execution)
            with (output / (stage + '.log')).open('x') as log:
                result = subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
            execution[stage + '_returncode'] = result.returncode
            dump(output / 'execution.json', execution)
            if result.returncode:
                raise RuntimeError(stage + ' failed; inspect its preserved log')
        execution.update(state='complete', ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        dump(output / 'execution.json', execution)
    except Exception as exc:
        execution.update(state='failed', error_type=type(exc).__name__)
        dump(output / 'execution.json', execution)
        raise
    return {'output': str(output), 'state': execution['state'], 'role': cfg['role'],
            'summary': str(output / 'summary.json')}


def package(output):
    files = {'lab/' + key: value for key, value in source_files().items()}
    license_path = APP.parent / 'LICENSE'
    if license_path.exists():
        files['LICENSE'] = license_path
    manifest = {'kind': 'portable timing v2 source/data bundle; no model weights or private records',
                'files': {key: {'sha256': sha(value), 'bytes': value.stat().st_size}
                          for key, value in files.items()}}
    prefix = 'tokenleak-timing-v2/'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            bodies = [(key, value.read_bytes()) for key, value in sorted(files.items())]
            bodies.append(('bundle_manifest.json', (json.dumps(manifest, indent=2) + '\n').encode()))
            for name, body in bodies:
                info = tarfile.TarInfo(prefix + name)
                info.size = len(body); info.mode = 0o644; info.mtime = 0
                info.uid = info.gid = 0; info.uname = info.gname = ''
                archive.addfile(info, io.BytesIO(body))
    return {'archive': str(output), 'sha256': sha(output), 'files': len(files)}


def verify_package(path):
    prefix = 'tokenleak-timing-v2/'
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive paths')
        for member in members:
            parts = Path(member.name).parts
            if (not member.isfile() or not member.name.startswith(prefix)
                    or '..' in parts or Path(member.name).is_absolute()):
                raise ValueError('Unsafe archive member')
        manifest = json.load(archive.extractfile(prefix + 'bundle_manifest.json'))
        expected = {prefix + name for name in manifest['files']} | {prefix + 'bundle_manifest.json'}
        if set(names) != expected:
            raise ValueError('Unexpected or missing archive files')
        for name, item in manifest['files'].items():
            body = archive.extractfile(prefix + name).read()
            if len(body) != item['bytes'] or hashlib.sha256(body).hexdigest() != item['sha256']:
                raise ValueError('Bundle checksum differs: ' + name)
    return {'verified': True, 'files': len(manifest['files']), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect'); p.add_argument('--output', type=Path)
    p = sub.add_parser('plan'); p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('prepare-model')
    p.add_argument('--reference', choices=['gpt2-small', 'gpt2-xl'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', type=Path, required=True); p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--max-requests', type=int, default=2000)
    p = sub.add_parser('evaluate'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('package'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify-package'); p.add_argument('archive', type=Path)
    args = parser.parse_args()
    if args.command == 'inspect':
        from probe_host import inventory
        result = inventory()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                stream.write(json.dumps(result, indent=2) + '\n')
            return
    elif args.command == 'plan':
        cfg = load_config(args.config); result = plan(cfg, materialize(cfg))
    elif args.command == 'prepare-model':
        prepare_model(args.reference, args.output.resolve()); return
    elif args.command == 'run':
        result = run(args.config.resolve(), args.model.resolve(), args.output.resolve(), args.max_requests)
    elif args.command == 'evaluate':
        from tokenleak_timing.evaluate import evaluate
        result = evaluate(args.output.resolve())
    elif args.command == 'package':
        result = package(args.output.resolve())
    else:
        result = verify_package(args.archive.resolve())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
