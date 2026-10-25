# TokenLeak — reproducible experiments and measured-observation pilot

Research fork of [ufxa/tokenleak](https://github.com/ufxa/tokenleak), maintained by Cayo Murilo. The upstream scientific code, data and paper remain preserved. The original README is retained in [docs/upstream/README_071bf8e.md](docs/upstream/README_071bf8e.md).

The new **[laboratory in `lab/`](lab/README.md)** separates three experimental objects:

| Track | Evidence | Scope |
|---|---|---|
| Public upstream replay | 65.453% Top-1 / 70.810% Top-5 without injected timing noise; published CSV reproduced | Reduced synthetic evaluator, 500 token types; historical 73.4%/91.2% configuration remains unreproduced |
| Independent simulation v1 | 5,000 controlled prompts, 8,431 types, ten measurement seeds, nine conditions | New closed-vocabulary corpus, estimated profiles, learned weights, known input positions; no model inference |
| Measured observation pilot M5 | GPT-2 small CPU, socket timings and negative controls | One controlled input-token identity among eight public candidates; distinct from complete prompt reconstruction or physical cache probing |

The independent simulation reaches 99.577% Top-1 under a strongly discriminative synthetic cache. Cache alone already reaches 99.475%. Sharing 128 cache fingerprints between token types reduces learned fusion to 4.638% in a paired two-seed exploratory comparison. These figures characterize the assumed channel and are not physical attack accuracy. See [results and limits](lab/docs/RESULTS_SIMULATION.md).

## Run the independent simulation

Python 3, Git and Docker are required. From this repository root:

```bash
python3 lab/run.py setup
python3 lab/run.py test
python3 lab/run.py all --config lab/configs/piloto_v1.json
python3 lab/run.py all
```

Setup creates a separate pinned runtime and refuses to modify an existing one. Experiments run offline in Docker. Every execution gets a new output directory, frozen code/configuration, observations, predictions and manifests. `--dry-run` displays the command. See [the laboratory guide](lab/README.md) for audit commands and data availability.

## Run the observation pilot

```bash
python3 lab/setup_observation.py
python3 lab/download_model.py
python3 lab/observe.py --config lab/configs/observacao_smoke_v1.json --output lab/runs/observation-smoke
python3 lab/observe.py --output lab/runs/observation-main
```

Only setup/model download use the internet. Collection runs in a container with external networking disabled; its service binds to loopback. Model weights are pinned safetensors and checked against the published hashes. Read the [observation contract](lab/docs/OBSERVATION_PROTOCOL.md) before interpreting its metrics.

## Portable timing v2

The new [v2 contract](lab/docs/TIMING_PROTOCOL_V2.md) and [operator handoff](lab/docs/RUN_TIMING_V2.md) prepare a separate timing-only experiment for GPT-2 XL on one explicit CUDA device. They preserve M5 v1. The design fixes the first-response-marker feature, profiling-only Gaussian compatibility, new-process/context tests and two negative controls. A separate routine collects the 5,000 full corpus inputs without assuming input-token timing alignment.

```bash
python3 lab/probe_host.py --output /path/to/private-records/host.json
python3 lab/timing_experiment.py plan --config lab/configs/timing_pilot_gpt2xl_v2.json
python3 lab/timing_experiment.py package --output /path/to/handoff/tokenleak-timing-v2.tar.gz
```

Inspection, planning and packaging need only Python's standard library. The eight-candidate pilot has 1,680 evaluation requests; the full 8,431-candidate design has 1,770,510. Technical CPU integration is checked separately; GPU execution still needs its technical smoke on the intended hardware. These tools do not establish physical cache leakage, passive-adversary access, full-prompt reconstruction or historical 73.4% reproduction.

## Data, provenance and interpretation

- [Controlled corpus and partitions](lab/data/corpus-v1/): synthetic public-word prompts, not the historical SPC-5K.
- [Per-run results](lab/results/), [simulation analysis](lab/docs/RESULTS_SIMULATION.md), [metric definitions](lab/docs/METRICS.md) and [reproducibility notes](lab/docs/REPRODUCIBILITY.md).
- Complete public experimental intermediates are distributed in the [repository releases](https://github.com/cayoc9/tokenleak/releases), with SHA-256 manifest and reproducible code. Model weights are downloaded from the original public publisher, rather than redistributed here.
- The [upstream manuscript](paper/) contains reported claims and publication metadata preserved as received. This fork's measured evidence is identified separately in `lab/`; it does not verify the publication metadata or all historical results.

The experiment code is for controlled academic security research. No production service or third-party infrastructure is probed. The fork contains experimental software/data and the inherited upstream article; it does not contain the dissertation sources or private conversation archive.

MIT license for code; see [LICENSE](LICENSE). CMUdict provenance and its license are retained in [lab/assets/cmudict](lab/assets/cmudict/).
