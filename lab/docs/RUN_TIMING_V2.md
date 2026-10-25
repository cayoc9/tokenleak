# Portable timing experiment: operator handoff

Read [the scientific contract](TIMING_PROTOCOL_V2.md) first. CPU integration is tested independently of the original M5; CUDA code is provided for a technical smoke on the intended GPU and has not yet been executed on a GPU. Keep the original v1 outputs intact.

Verification on 2026-10-24: all 30 laboratory unit tests passed. An extracted portable bundle passed its v2 tests and completed the two CPU integration paths offline with the existing small checkpoint: 36 evaluation requests plus nine warmups for the single-slot task, and three full-input requests plus one warmup for the corpus path. An external standard-library recount verified raw read/marker association, source hashes, profiles, predictions and denominators. These are technical checks; they do not extend M5's scientific conclusions. The tested collector/evaluator source is unchanged by the final addition of the resampling unit test and this verification note.

## 1. Inspect without loading a model

From the repository root or the extracted `tokenleak-timing-v2/` bundle:

```bash
python3 lab/probe_host.py --output /path/to/private-records/host.json
python3 lab/timing_experiment.py plan --config lab/configs/timing_pilot_gpt2xl_v2.json
python3 lab/timing_experiment.py plan --config lab/configs/timing_8431_gpt2xl_v2.json
```

These commands use only Python's standard library. Inspection reads CPU/memory/quota, disk space, installed package metadata and `nvidia-smi` if present. It neither installs software nor allocates a model or GPU job. Missing `nvidia-smi` is not proof that the physical host lacks a GPU. A VM must expose its GPU and allow a local Python process; access only to an inference UI or hypervisor manager is insufficient for this collector. Keep actual host inventories private.

For GPT-2 XL, verify a supported NVIDIA GPU/driver, available VRAM, host RAM, local writable storage, Python access and the permitted job window. The pinned checkpoint has a 6,431,829,964-byte safetensors file, plus small configuration/tokenizer files. Downloads, the environment, transient loading and raw outputs require additional storage. Memory and runtime minima have not been benchmarked on GPU: inspect, then run the small technical smoke before estimating capacity for larger jobs.

## 2. Prepare an isolated environment and checkpoint

Use a **new** virtual environment. Python 3.11.9 is the CPU-tested interpreter. Do not change a shared model-serving environment or host driver. If the GPU cannot run PyTorch 2.3.1/CUDA 12.1, record that incompatibility and define a versioned environment change rather than silently substituting packages.

Example CUDA environment using an already available compatible Python 3.11:

```bash
python3.11 -m venv /path/to/new-tokenleak-venv
/path/to/new-tokenleak-venv/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
/path/to/new-tokenleak-venv/bin/python -m pip install -r lab/requirements-timing.lock
/path/to/new-tokenleak-venv/bin/python -m pip check
/path/to/new-tokenleak-venv/bin/python -m pip freeze > /path/to/private-records/pip-freeze.txt
```

For a CPU environment replace the PyTorch wheel index with `https://download.pytorch.org/whl/cpu`. The version/index combinations come from [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v231). The lock fixes the Python dependencies inherited from the tested CPU runtime; CUDA wheel dependencies and the host driver must also be recorded by the operator. The runner checks Torch/Transformers/NumPy versions, device and dtype, and refuses a silent CPU fallback.

The explicit download command verifies every asset against the immutable shipped reference, including all SHA-256 values. It requires internet access; collection loads only local assets and sets Hugging Face offline mode. Network isolation can additionally be supplied by the host/container policy.

```bash
python3 lab/timing_experiment.py prepare-model --reference gpt2-xl --output /path/to/models/gpt2-xl-fixed
```

Use `--reference gpt2-small` for the CPU smoke. Downloads never overwrite a conflicting asset; interrupted `.part` files remain available for inspection. Model weights are not in Git or the portable bundle. The GPT-2 XL identity is [this fixed publisher revision](https://huggingface.co/openai-community/gpt2-xl/tree/15ea56dee5df4983c59b2538573817e1667135e2).

## 3. Check software and run the technical smoke

The bundle includes the twelve v2 unit tests, which require NumPy but do not load Torch or a model:

```bash
cd lab
/path/to/new-tokenleak-venv/bin/python -B -m unittest discover -s tests -p test_timing_v2.py -v
cd ..
```

Then select the GPU smoke with a **fresh** output directory:

```bash
/path/to/new-tokenleak-venv/bin/python -B lab/timing_experiment.py run \
  --config lab/configs/timing_smoke_gpt2xl_cuda_v2.json \
  --model /path/to/models/gpt2-xl-fixed \
  --output /path/to/private-runs/cuda-smoke-001 --max-requests 100
```

The CPU equivalent uses `timing_smoke_cpu_v2.json` and the small checkpoint. `timing_corpus_smoke_cpu_v2.json` exercises three full prompts. A technical smoke checks loading, process boundaries, markers, labels, raw records, profiles and evaluation. Its accuracy has no scientific interpretation and it computes no significance test. Review `execution.json`, `status.json`, `summary.json`, per-process `*_runtime.json` and logs before larger runs. GPU completion occurs before each marker through the argmax scalar `.item()`; diagnostics are host-visible forward/argmax durations, not CUDA-event kernel timing.

## 4. Run a fresh exploratory pilot and separate corpus trace

```bash
/path/to/new-tokenleak-venv/bin/python -B lab/timing_experiment.py run \
  --config lab/configs/timing_pilot_gpt2xl_v2.json \
  --model /path/to/models/gpt2-xl-fixed \
  --output /path/to/private-runs/xl-pilot-001 --max-requests 2000

/path/to/new-tokenleak-venv/bin/python -B lab/timing_experiment.py run \
  --config lab/configs/timing_corpus5000_gpt2xl_v2.json \
  --model /path/to/models/gpt2-xl-fixed \
  --output /path/to/private-runs/xl-corpus-001 --max-requests 5000
```

Use a persistent terminal session for long work. Every process is limited to the configured two Torch threads; a host scheduler/container can additionally constrain CPU/RAM and GPU assignment. No resource exclusivity is enforced by this script. Record other GPU work and thermal/power conditions; scheduling changes can affect timings. The observer service binds only to loopback and never exposes a listening public endpoint.

Do not run the pilot concurrently with the corpus trace on the same device. Preserve all failed observations. There is no resume option: use a fresh run after diagnosing failure. The collection time limit is checked between requests; socket operations and model startup have additional bounded timeouts. Analysis is separate and can take extra time. A run reaching the cap is incomplete and does not publish reduced-denominator accuracy.

## 5. Assess the full 8,431-candidate budget

First inspect pilot throughput, variance and negative controls. Whether the real channel is useful is an empirical question. The full configuration requires **1,770,510 evaluation requests**, not 5,000. At ten requests/second this alone is about 49.2 hours, exceeding the default 24-hour collection limit. That is an arithmetic example, not measured GPU throughput. Plan capacity before execution. Any time/resource change must be saved as a new explicit configuration and frozen with the run; do not tune features or candidate selection against the test labels.

If the declared design fits the agreed job window:

```bash
/path/to/new-tokenleak-venv/bin/python -B lab/timing_experiment.py run \
  --config lab/configs/timing_8431_gpt2xl_v2.json \
  --model /path/to/models/gpt2-xl-fixed \
  --output /path/to/private-runs/xl-8431-001 --max-requests 1770510
```

This measures the single-slot timing hypothesis under all public candidates. It does not reproduce the paper's historical fused 73.4%, supply a physical cache channel or reconstruct complete prompts. A negative result remains a valid outcome.

## 6. Package and return evidence

```bash
python3 lab/timing_experiment.py package --output /path/to/handoff/tokenleak-timing-v2.tar.gz
python3 lab/timing_experiment.py verify-package /path/to/handoff/tokenleak-timing-v2.tar.gz
```

The explicit bundle contains v2 source/tests/configurations/contracts, the frozen corpus/vocabulary/splits, model reference hashes, the minimal transport module and licenses. It excludes model weights, prior runs, private records, credentials and dissertation sources. The archive itself has a returned SHA-256 and an internal file manifest. Verification checks integrity and safe paths; authenticity requires obtaining the expected archive hash from a trusted delivery channel. Preserve the Git revision or archive hash used.

Return the complete run directory, including failures, logs and source snapshot. Truth files hold controlled synthetic inputs; environment/runtime records identify the host. Review these records before public release. `evaluate --output /path/to/run` is available only when collection completed and no evaluation artifacts exist; it never overwrites a previous analysis. A partial analysis should remain preserved while a fresh copy is assessed. No command here performs remote access, modifies VPN routing, installs a cache attack primitive or contacts another person.
