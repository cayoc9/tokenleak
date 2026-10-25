"""Read-only experiment inventory. Uses stdlib; never loads a model or runs a GPU job."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args, timeout=8):
    if not shutil.which(args[0]):
        return {"available": False, "command": args[0]}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"available": True, "returncode": result.returncode,
                "stdout": result.stdout.strip()[:16000], "stderr": result.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": type(exc).__name__}


def read_number(path):
    try:
        value = Path(path).read_text().strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def inventory():
    cpu, flags, memory = None, [], {}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            key, _, value = line.partition(':')
            if key.strip() == 'model name' and cpu is None:
                cpu = value.strip()
            if key.strip() == 'flags' and not flags:
                flags = value.split()
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in {'MemTotal', 'MemAvailable'}:
                memory[key + '_bytes'] = int(value.split()[0]) * 1024
    except OSError:
        pass
    packages = {}
    for name in ['torch', 'transformers', 'tokenizers', 'numpy', 'scipy', 'scikit-learn', 'safetensors']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    usage = shutil.disk_usage('.')
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    try:
        quota = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        quota_cpus = None if quota[0] == 'max' else int(quota[0]) / int(quota[1])
    except (OSError, ValueError):
        quota_cpus = None
    return {
        'schema_version': 1,
        'record_kind': 'read_only_host_inventory',
        'collected_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'cpu': {'model': cpu, 'logical': os.cpu_count(), 'affinity': affinity,
                'cgroup_quota_cpus': quota_cpus, 'clflush_flag': 'clflush' in flags},
        'memory': {**memory, 'cgroup_limit_bytes': read_number('/sys/fs/cgroup/memory.max')},
        'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
        'disk': {'free_bytes': usage.free, 'total_bytes': usage.total},
        'python': {'version': platform.python_version(), 'executable': sys.executable,
                   'packages_in_this_interpreter': packages},
        'commands': {name: bool(shutil.which(name)) for name in ['docker', 'nvidia-smi', 'git', 'tmux']},
        'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,utilization.gpu',
                        '--format=csv,noheader,nounits']),
        'container': Path('/.dockerenv').exists(),
        'claims': {'model_loaded': False, 'gpu_computation_executed': False,
                   'cache_primitive_validated': False, 'shared_pages_proven': False},
        'limits': ['Physical inventory is not a reservation or a performance test.',
                   'GPU/CLFLUSH availability does not establish adversarial cache access.',
                   'Package versions refer only to the executing Python interpreter.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = json.dumps(inventory(), indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(value)
    else:
        print(value, end='')


if __name__ == '__main__':
    main()
