# Instrumental GPT-2 XL timing on a CUDA GPU — 2026-10-25

The v2 software now has a completed GPU technical smoke, an exploratory eight-candidate pilot and a diagnostic trace of all 5,000 reconstructed corpus inputs. **The pilot did not demonstrate classification advantage under this view.** This is real inference and measured socket timing, with controlled synthetic inputs; it is distinct from physical cache validation or reproduction of the historical fused 73.4%.

## Environment and source

Measurements executed the unchanged source in commit `ffed170d4b7bb7ab27152006b69d7c015f1566f6`, from source bundle SHA-256 `90b3e09d306c2ef904dae7a794d5c598526cb0a22cd4e014cdb151e758f368dc`. The current publication adds an independent auditor, reference results, an actual GPU dependency lock and documentation; it preserves those execution snapshots and the earlier M5/simulation results.

The environment was a KVM virtual machine exposing an RTX 4090 (24 GiB), driver 580.126.09, a guest CPU string identifying an i9-13900K, 30 visible logical CPUs and approximately 96 GiB RAM. Processes used guest CPU affinity `0,1`, two Torch threads and nice level 5. Guest affinity does not establish physical core placement or exclusive host/GPU access. Temperature/utilization snapshots do not replace continuous monitoring or a resource reservation.

Python 3.11.9, Torch 2.3.1+cu121, CUDA build 12.1, Transformers 4.41.2 and NumPy 1.26.0 ran in a new user virtual environment. The interpreter binaries came from the previously tested pinned Docker image; **collection ran natively with host system libraries**, including a different OpenSSL from the CPU container. Docker was not usable by the measurement user. No driver or shared serving environment was modified. Hugging Face offline flags and loopback transport were used; there was no network namespace isolation. Full package versions are in [the GPU lock](../requirements-timing-gpu.lock); [provenance](https://github.com/cayoc9/tokenleak/blob/main/lab/results/timing-v2-rtx4090/provenance.json) records the conditions and limits.

The public GPT-2 XL checkpoint revision is `15ea56dee5df4983c59b2538573817e1667135e2`, with safetensors/configuration/tokenizer hashes checked before execution. Runtime records show 1,557,611,200 model parameters, `cuda:0`, float16, eager attention and TF32 disabled. The first marker follows prefill and a host-visible argmax `.item()` barrier. This observable includes transport and scheduling; it is not a GPU-kernel-only duration.

## Completed runs

| Run | Evaluation requests | Warmups | Purpose |
|---|---:|---:|---|
| CUDA smoke | 36 | 9 | Technical integration with two candidates; no significance inference |
| Eight-candidate pilot | 1,680 | 45 | Exploratory single-slot classification with two negative controls |
| Full corpus trace | 5,000 | 5 | Actual response timing for the original reconstructed inputs; no reconstruction predictions |

All runs completed their planned denominators. The original source bundle's twelve unit tests passed on the new runtime, and a tiny CUDA arithmetic check passed before model loading. The current tree has thirteen v2 tests and 31 laboratory tests, with an additional known-distribution check for the independent auditor. Tests and data-integrity checks do not establish a successful attack.

## Exploratory pilot result

The target is one input token at a public slot in a 16-token input, from eight balanced public candidates. Profiling has fifty samples per candidate and condition. Validation uses a new process with known contexts; the primary test uses another process and two held-out instruction families, with eight repetition blocks. The fixed feature/ranking are defined in [the contract](TIMING_PROTOCOL_V2.md). Conditions run sequentially, so temporal drift remains a limitation.

| Primary condition | Top-1 count | Top-1 | Top-5 | Planned bootstrap 95% CI for Top-1 |
|---|---:|---:|---:|---|
| GPT-2 XL | 13/128 | 10.156% | 62.500% | 7.813–13.281% |
| Input-blinded GPT-2 XL | 15/128 | 11.719% | 63.281% | 8.594–14.844% |
| No-inference transport | 14/128 | 10.938% | 57.031% | 7.813–14.844% |
| Uniform reference | expected 16/128 | 12.500% | 62.500% | — |

The planned one-sided within-context/repetition permutation p-value for the model is 0.8581. Paired Top-1 differences are -1.563 percentage points versus input blinding (95% interval -5.469 to +2.344) and -0.781 points versus no inference (-3.125 to +1.563). These are conditional on the fixed candidates, four contexts, process sessions and device. Eight repetition blocks are not eight independent machines or prompt populations.

An independent standard-library auditor re-estimated profiles, reranked all saved validation/test observations and recounted Top-1/Top-5. It also computed the declared conditional permutation null exactly by enumeration/convolution for the eight-candidate groups: p = 0.85392, consistent with the 2,000-draw Monte Carlo result. Its exact empirical block-bootstrap interval is 7.031–13.281%; the original Monte Carlo lower bound differs by one request out of 128. Both analyses are preserved and distinguished. The exact check is an audit of the same conditional test, not new measurement evidence or broader population coverage.

See the original [pilot summary](https://github.com/cayoc9/tokenleak/blob/main/lab/results/timing-v2-rtx4090/pilot_gpu_summary.json) and [audit](https://github.com/cayoc9/tokenleak/blob/main/lab/results/timing-v2-rtx4090/audit_pilot_gpu_strict.json). Failure to demonstrate advantage does not prove equivalence, universal absence of leakage or safety of other observables/deployments. No feature was retuned to obtain a desired accuracy. The earlier CPU M5 result also happened to be 13/128, but that was a different experiment, checkpoint, feature/ranking and dataset; numerical equality is not a controlled CPU/GPU comparison.

## Full 5,000-input trace

The separate collection exercised every frozen `SPC5K-reconstruido-v1` input, covering **8,431 distinct input token IDs and 178,209 input positions**. It recorded eight response markers per request, totaling **40,000 markers**, plus five warmups. The independent audit checked the seeded request order, every input against the frozen corpus, raw-read/marker association and latency aggregates.

Time from request send to first complete marker had mean **9.969 ms**, median **9.919 ms**, and 5th/95th percentiles **9.648/10.379 ms**. This is a request-level latency distribution in this environment. The eight response markers follow generated output steps; they are not times aligned to the input's 8–64 positions. The trace therefore does not supply 178,209 individually observed input-token timings, token-reconstruction predictions, protected-field recovery or prompt exact match. The reconstructed corpus is not identified as the historical SPC-5K.

See [trace summary](https://github.com/cayoc9/tokenleak/blob/main/lab/results/timing-v2-rtx4090/corpus_gpu_summary.json) and [audit](https://github.com/cayoc9/tokenleak/blob/main/lab/results/timing-v2-rtx4090/audit_corpus_gpu.json).

## Full-candidate budget and next evidence

The separate 8,431-candidate design has **1,770,510 evaluation requests**, not 5,000. Scaling the pilot's accumulated last-marker request durations by `8431/8` gives approximately **25.35 hours of request work**, excluding startup, orchestration, analysis and drift. This is an extrapolation, not a guaranteed runtime, and already exceeds the configured 24-hour collection cap. It was not launched. A new explicit resource window/configuration and an appropriate full-candidate audit are needed before that run; the new stdlib auditor is intentionally bounded to eight classification candidates and diagnostic corpus traces.

The real-timing artifacts reduce the measurement/instrumentation gap. Historical dataset identity, per-input-position observability, the physical cache channel, a separate adversarial observer, sequence reconstruction, defenses/overhead and modern-model transfer remain open. Real inference on a GPU does not establish Flush+Reload over GPU memory or validate the historical fusion accuracy.

## Artifact use and privacy

Compact unchanged summaries, independent audit reports and de-identified environment provenance are in [`results/timing-v2-rtx4090/`](https://github.com/cayoc9/tokenleak/tree/main/lab/results/timing-v2-rtx4090). The release evidence bundle includes raw observations, controlled inputs, diagnostics, profiles, predictions and frozen execution source, plus a SHA-256 manifest. Local account paths in execution metadata are replaced by a documented placeholder; scientific records remain byte-identical. Hostnames, network addresses, SSH material, authentication logs, private messages, model weights and dissertation sources are excluded. Original complete run directories remain private.

To audit an extracted evidence run without Torch or NumPy:

```bash
python3 lab/audit_timing_v2.py --run /path/to/evidence/pilot_gpu --report /path/to/new-audit.json
```

The auditor refuses output overwrite and optimized Python (`-O`). It is a bounded audit of the declared artifacts and conditional statistics, not proof of authenticity, independent adversarial access or a successful attack.
