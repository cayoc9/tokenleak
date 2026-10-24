"""Portable offline Docker runner for the independent simulation laboratory."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

IMAGE = 'python@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['setup', 'test', 'all', 'build', 'run', 'verify', 'audit'])
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--runtime', type=Path, default=Path.home()/'.cache/tokenleak/metodologia-v1')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    app = Path(__file__).resolve().parent
    repo = app.parent
    runtime = args.runtime.resolve()
    docker = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
              '--cpus', '3', '--memory', '4g', '-v', str(runtime)+':/runtime',
              '-v', str(app)+':/app:ro', '-w', '/app']
    if args.command == 'setup':
        if (runtime/'venv').exists():
            raise SystemExit('Runtime exists; reuse it or select a new --runtime. No packages changed.')
        cmds = [docker+[IMAGE, 'python', '-m', 'venv', '/runtime/venv'],
                docker+[IMAGE, '/runtime/venv/bin/pip', 'install', '--no-cache-dir', '-r', '/app/requirements-simulation.lock']]
        print(json.dumps(cmds, indent=2))
        if args.dry_run:
            return
        runtime.mkdir(parents=True, exist_ok=True)
        for cmd in cmds:
            subprocess.run(cmd, check=True)
        frozen = subprocess.check_output(docker+[IMAGE, '/runtime/venv/bin/pip', 'freeze'], text=True)
        (runtime/'freeze.txt').write_text(frozen)
        return
    if not (runtime/'freeze.txt').exists():
        raise SystemExit('Run setup first or select a prepared --runtime.')
    output = (args.output or app/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    source = app if args.command in ['test', 'audit'] else output/'source_snapshot'
    common = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
              '--cpus', '3', '--memory', '4g', '-e', 'PYTHONDONTWRITEBYTECODE=1',
              '-e', 'PYTHONUNBUFFERED=1', '-e', 'TOKENIZERS_PARALLELISM=false',
              '-e', 'OPENBLAS_NUM_THREADS=2', '-e', 'OMP_NUM_THREADS=2',
              '-e', 'MPLCONFIGDIR=/tmp/matplotlib', '-e', 'TOKENLEAK_WORKSPACE=/repo',
              '-v', str(runtime)+':/runtime:ro', '-v', str(repo)+':/repo:ro',
              '-v', str(source)+':/app:ro', '-w', '/app']
    if args.command == 'test':
        command = common+[IMAGE, '/runtime/venv/bin/python', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    elif args.command == 'audit':
        command = common+['-v', str(output)+':/output', IMAGE, '/runtime/venv/bin/python',
                          '/app/auditar.py', '--workspace', '/repo', '--output', '/output',
                          '--report', '/output/additional_audit.json']
    else:
        command = common+['-v', str(output)+':/output', IMAGE, '/runtime/venv/bin/python', '-m', 'tokenleak_lab',
                          args.command, '--config', '/output/config.json', '--output', '/output']
    print(json.dumps(command, indent=2), flush=True)
    if args.dry_run:
        return
    if args.command == 'test':
        raise SystemExit(subprocess.run(command).returncode)
    if args.command in ['all', 'build']:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(app, source, ignore=shutil.ignore_patterns('runs', 'results', 'data', '__pycache__', '*.pyc'))
        shutil.copy2(args.config or app/'configs/simulacao_v1.json', output/'config.json')
        shutil.copy2(runtime/'freeze.txt', output/'environment_freeze.txt')
    elif not output.is_dir():
        raise SystemExit('Select an existing execution with --output.')
    record = output/(args.command+'_public_command.json')
    if record.exists():
        raise SystemExit('Command record exists; refusing to overwrite.')
    metadata = {'command': command, 'image': IMAGE, 'network': 'none',
                'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    record.write_text(json.dumps(metadata, indent=2)+'\n')
    with (output/(args.command+'_public.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(returncode=result.returncode, ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    record.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'returncode': result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
