"""Run the registered GPT-2 small CPU observation pilot with frozen sources."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from run import IMAGE
from download_model import sha


def verify_assets(model, config):
    reference = json.loads((model/'manifest.json').read_text())
    if reference['model_id'] != config['model_id'] or reference['revision'] != config['model_revision']:
        raise ValueError('Checkpoint identity differs from registered configuration')
    for item in reference['files']:
        path = model/item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Model asset hash differs: '+item['path'])
    if config['device'] != 'cpu' or config['dtype'] != 'float32':
        raise ValueError('This version implements CPU/float32 only')
    if config['sessions'] != 3 or len(config['candidate_words']) != 8 or len(config['context_prefixes']) != 4:
        raise ValueError('This evaluation contract uses three sessions, eight candidates and four contexts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path.home()/'.cache/tokenleak/observation-v1')
    p.add_argument('--model', type=Path, default=Path.home()/'.cache/tokenleak/models/gpt2-607a30d783df')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    app = Path(__file__).resolve().parent
    config = (args.config or app/'configs/observacao_real_v1.json').resolve()
    output, runtime, model = args.output.resolve(), args.runtime.resolve(), args.model.resolve()
    if not (runtime/'freeze.txt').exists() or not (model/'model.safetensors').exists():
        raise SystemExit('Prepare the observation runtime and verified model; see lab/README.md')
    common = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
              '--cpus', '2', '--memory', '4g', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1',
              '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
              '-e', 'OPENBLAS_NUM_THREADS=2', '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2',
              '-v', str(runtime)+':/runtime:ro', '-v', str(model)+':/model:ro',
              '-v', str(output/'source_snapshot')+':/app:ro', '-v', str(output)+':/output', '-w', '/app',
              IMAGE, '/runtime/venv/bin/python']
    commands = [common+['-m', 'tokenleak_observation.collect', '--config', '/output/config.json',
                        '--model', '/model', '--output', '/output/run'],
                common+['-m', 'tokenleak_observation.evaluate', '--output', '/output/run']]
    print(json.dumps(commands, indent=2), flush=True)
    if args.dry_run:
        return
    verify_assets(model, json.loads(config.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(app, output/'source_snapshot', ignore=shutil.ignore_patterns('runs', 'results', 'data', '__pycache__', '*.pyc'))
    shutil.copy2(config, output/'config.json')
    shutil.copy2(runtime/'freeze.txt', output/'environment_freeze.txt')
    if (model/'manifest.json').exists():
        reference = json.loads((model/'manifest.json').read_text())
        reference.pop('path', None)
        (output/'model_manifest.json').write_text(json.dumps(reference, indent=2)+'\n')
    record = {'commands': commands, 'image': IMAGE, 'network': 'none', 'cpus': 2, 'memory': '4g',
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': {str(f.relative_to(output/'source_snapshot')): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in (output/'source_snapshot').rglob('*') if f.is_file()}}
    (output/'command.json').write_text(json.dumps(record, indent=2)+'\n')
    for stage, command in zip(['collection', 'evaluation'], commands):
        print(json.dumps({'stage': stage, 'state': 'start'}), flush=True)
        with (output/(stage+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record[stage+'_returncode'] = result.returncode
        (output/'command.json').write_text(json.dumps(record, indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    record.update(returncode=0, ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (output/'command.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'complete': True, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
