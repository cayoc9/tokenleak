"""Download only the pinned public safetensors checkpoint; verify every SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path.home()/'.cache/tokenleak/models/gpt2-607a30d783df')
    args = p.parse_args()
    reference = json.loads((Path(__file__).resolve().parent/'assets/model_references/gpt2-small.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for item in reference['files']:
        target = args.output/item['path']
        if target.exists():
            if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
                raise ValueError('Existing model asset differs from pinned reference: '+item['path'])
            print('Verified existing '+item['path'], flush=True)
            continue
        part = target.with_suffix(target.suffix+'.part')
        with urllib.request.urlopen(item['source_url'], timeout=120) as response, part.open('wb') as f:
            for chunk in iter(lambda: response.read(1024*1024), b''):
                f.write(chunk)
        if part.stat().st_size != item['bytes'] or sha(part) != item['sha256']:
            raise ValueError('Downloaded asset differs from reference: '+item['path'])
        part.rename(target)
        print('Downloaded and verified '+item['path'], flush=True)
    (args.output/'manifest.json').write_text(json.dumps(reference, indent=2)+'\n')


if __name__ == '__main__':
    main()
