"""Persistencia deterministica e manifestos; nunca sobrescreve uma execucao."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def manifest(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}


def check_manifest(directory):
    directory = Path(directory)
    expected = json.loads((directory / 'manifest.json').read_text())
    errors = [name for name, h in expected.items()
              if not (directory / name).is_file() or digest(directory / name) != h]
    if errors:
        raise ValueError(f'Artifact integrity failure: {errors}')
    return len(expected)
