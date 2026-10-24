import argparse
import json
from pathlib import Path
import os

from . import corpus, pipeline
from .io import write_json, digest


def main():
    parser = argparse.ArgumentParser(description='Corpus reconstruido e avaliacao metodologica TokenLeak')
    parser.add_argument('command', choices=['build', 'run', 'verify', 'all'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config['mode'] != 'fully_simulated' or config['alignment'] != 'oracle_input_positions':
        raise ValueError('This implementation does not assert a measured or physically aligned attack')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'config.json'
    if frozen.exists() and digest(frozen) != digest(args.config):
        raise ValueError('Configuration differs from the frozen execution')
    if not frozen.exists():
        write_json(frozen, config)
    if args.command in ['build', 'all']:
        print(json.dumps(corpus.build(config, output/'corpus', os.environ['TOKENLEAK_WORKSPACE'], Path(__file__).resolve().parents[1])), flush=True)
    if args.command in ['run', 'all']:
        pipeline.run(config, output)
    if args.command in ['verify', 'all']:
        result = pipeline.verify(config, output)
        write_json(output / 'verification.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
