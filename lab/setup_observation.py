"""Create a separate pinned CPU runtime; refuse to mutate an existing environment."""
import argparse
import os
from pathlib import Path
import subprocess

from run import IMAGE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, default=Path.home()/'.cache/tokenleak/observation-v1')
    args = p.parse_args()
    runtime = args.runtime.resolve()
    if (runtime/'venv').exists():
        raise SystemExit('Runtime exists; reuse it or choose a fresh --runtime. No packages changed.')
    runtime.mkdir(parents=True, exist_ok=True)
    app = Path(__file__).resolve().parent
    common = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '2', '--memory', '3g',
              '-v', str(runtime)+':/runtime', '-v', str(app)+':/app:ro', IMAGE]
    subprocess.run(common+['python', '-m', 'venv', '/runtime/venv'], check=True)
    subprocess.run(common+['/runtime/venv/bin/pip', 'install', '--no-cache-dir', 'torch==2.3.1+cpu',
                          '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
    subprocess.run(common+['/runtime/venv/bin/pip', 'install', '--no-cache-dir', '-r', '/app/requirements-observation.lock'], check=True)
    frozen = subprocess.check_output(common+['/runtime/venv/bin/pip', 'freeze'], text=True)
    (runtime/'freeze.txt').write_text(frozen)


if __name__ == '__main__':
    main()
