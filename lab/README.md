# TokenLeak laboratory

This is the self-contained development tree for the independent methodology. It requires no dissertation checkout. The original `src/` evaluator remains a separate reference. The [root guide](../README.md) presents the three experimental tracks; [metric explanations in Portuguese](docs/METRICS.md) connect claims and denominators.

## Simulation

`tokenleak_lab/` implements controlled corpus generation, raw calibration samples, estimated profiles, training-only nonnegative logistic fusion, deterministic ranking, pruning, nine controls/ablations, and per-prompt metrics. The default corpus has 5,000 public-word prompts, 8,431 token types, 20 topics and lengths 8–64. Global instruction families split 3,000/1,000/1,000 into training/validation/test. The attacker knows a public closed vocabulary and receives fifty additional known-token calibration samples per type. Signal alignment to input positions is supplied by the simulator.

The corpus uses the frozen public GPT-2 XL tokenizer and public CMUdict assets with hashes/licenses. It is a controlled keyword-list benchmark, not a natural-conversation corpus or recovered historical SPC-5K. Frequencies are constrained Zipf-like counts fixed before attack evaluation. All token types receive profiles, including rare types.

```bash
python3 lab/run.py setup
python3 lab/run.py test
python3 lab/run.py all --config lab/configs/piloto_v1.json --output lab/runs/pilot
python3 lab/run.py audit --output lab/runs/pilot
python3 lab/run.py all --output lab/runs/simulation-main
python3 lab/run.py all --config lab/configs/colisoes128_v1.json --output lab/runs/collisions128
```

Commands refuse to overwrite executions and audit reports. `build`, `run` and `verify` allow separate phases, with `--output` selecting an existing frozen execution for the latter two. `--runtime` selects an isolated runtime; the default is `~/.cache/tokenleak/metodologia-v1`. Setup uses the full [simulation lock](requirements-simulation.lock), Python 3.11.9 and a pinned Docker image. Execution has external networking disabled, a three-CPU quota and a 4 GiB memory limit. These are declared test conditions, not proven hardware minima.

Saved data include corpus, vocabulary/frequencies, partitions, raw profile timings/cache bits, estimated profiles, training examples, weights, held-out observations, Top-5 predictions, candidate counts/fallback, per-prompt metrics and aggregates. Validation observations can be regenerated from their seeds but are not saved raw in this v1 pipeline. Full per-channel/candidate score matrices can be recomputed from observations/profiles; saved score arrays contain Top-1 fusion scores. Confidence intervals are Student-t across measurement seeds conditional on the fixed corpus and latent channel. This version implements Top-5 and 95% intervals.

`auditar.py` adds full retokenization, group-membership checks, raw-to-profile recalculation, observation replay, external metric recounts and sampled independent ranking/edit-distance checks. Its replay shares the frozen generator; metric/ranking calculations do not import the evaluator/reconstructor. Historical artifacts with the old asset directory resolve to the same pinned tokenizer shipped here. [Results](docs/RESULTS_SIMULATION.md) identify exactly what was checked.

## Measured observation

`tokenleak_observation/` contains a loopback server, socket-read observer, randomized balanced design and training/evaluation. [OBSERVATION_PROTOCOL.md](docs/OBSERVATION_PROTOCOL.md) is the scientific contract. This pilot uses GPT-2 small CPU and response-level features to classify one candidate input token at a fixed public position. It does not provide per-input-position timing alignment, probe physical cache or reconstruct full prompts.

```bash
python3 lab/setup_observation.py
python3 lab/download_model.py
python3 lab/observe.py --config lab/configs/observacao_smoke_v1.json --output lab/runs/observation-smoke
python3 lab/observe.py --output lab/runs/observation-main
```

The CPU runtime is separate: `~/.cache/tokenleak/observation-v1`, pinned by [requirements-observation.lock](requirements-observation.lock). CPU PyTorch comes from the official CPU wheel index. The model [reference manifest](assets/model_references/gpt2-small.json) pins safetensors, configuration/tokenizer and hashes. Downloaded weights live outside Git. Both setup scripts refuse to mutate an existing runtime; reuse it or pass a fresh `--runtime`.

The observation wrapper freezes code, configuration, model manifest and runtime versions before collection. Its observer file excludes labels, input IDs and server diagnostics. Actual model timings remain privileged diagnostic traces. Three conditions investigate content dependence: GPT-2, input-blinded GPT-2, and no-inference transport control. New-session/new-context requests are the primary test. No hyperparameters are selected using them.

After an observation execution, run an external audit in the prepared CPU runtime:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
  -v "$HOME/.cache/tokenleak/observation-v1:/runtime:ro" \
  -v "$PWD/lab:/app:ro" -v "$PWD/lab/runs/observation-main:/output" -w /app \
  python@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 \
  /runtime/venv/bin/python /app/audit_observation.py \
  --run /output/run --report /output/additional_audit.json
```

The audit reconstructs features directly from socket-read/marker records, refits the declared training-only model, checks fixed sizes and class balance, and recounts saved decisions. Passing software/data checks does not establish a generalizable attack. See the result report for the observed controls and scope.

## Portable timing v2

`timing_experiment.py` is the separate portable v2 entry point. Its [scientific contract](docs/TIMING_PROTOCOL_V2.md) fixes a single-slot first-marker timing hypothesis after the negative M5 result; the [operator guide](docs/RUN_TIMING_V2.md) covers read-only host inspection, a new isolated runtime, verified model download, CPU/CUDA technical smokes, the eight-candidate pilot, a separate full-input corpus trace and the larger candidate budget. It supports GPT-2 small CPU and GPT-2 XL on an explicit CUDA device. Actual GPU execution has not yet been validated.

The v2 bundle includes its exact source, twelve unit tests, six configurations, frozen input data and checkpoint references. It excludes prior runs and private host records. `run` freezes the executed source and refuses output reuse. V1 code and results remain preserved. A measured response marker does not supply physical-cache observations or input-position alignment.

## Public data and artifact layout

`data/corpus-v1/` contains immutable reference text/labels and splits. `results/` contains compact summaries, per-run metrics, data provenance and uncertainty. The release bundle holds the larger raw profiles, observations, predictions, server diagnostics, manifests and execution source snapshots. Private messages, credentials, workstation history and dissertation sources are excluded by an explicit export file list.

The historical simulation code snapshot remains in the release so its results can be audited. The portable runner changes asset/output paths and packaging, preserving scientific equations and random sampling; its pilot produced nineteen byte-identical scientific artifacts compared with the preserved initial pilot. The current maintained implementation is this `lab/` tree.
