# Instrumental timing protocol v2

Version 2 preserves the completed M5 CPU pilot and the independent simulation. It fixes a new, exploratory question after M5's negative primary result. It is not a rerun of the historical 73.4% fusion experiment and does not select parameters to approach that value. See [operator instructions](RUN_TIMING_V2.md).

## Question and observable

**H_TTFT_v2:** identity of one input token in a public fixed slot changes the time to the first response marker enough for a fixed profiling rule to generalize to a new process and unseen public contexts.

The coordinator submits raw GPT-2 `input_ids` over padded 2,048-byte requests to an instrumented HTTP service on `127.0.0.1`. Its only classification feature is milliseconds from sending the request to receiving the first complete 64-byte response marker. The server runs a real prefill, then autoregressive decoding, emitting a marker after each forward pass and argmax completion. Markers contain no generated-token values. There is no alignment between input-token positions and response intervals. Transport, request parsing and scheduling contribute to the observable; it is not GPU kernel time.

This is a strong laboratory interface: raw input IDs, public candidate vocabulary, known input length and known secret position. The coordinator knows inputs to score the experiment. Excluding them from the feature computation does not establish that an independent passive attacker could obtain the same view. Attacker separation is a subsequent protocol requirement.

## Fixed design

`data/corpus-v1/` is the existing **SPC5K-reconstruido-v1**: 5,000 synthetic public-word prompts, 8,431 distinct GPT-2 token IDs and fixed splits. File SHA-256 identities are checked before planning or running. This establishes repeatability of our new benchmark, not identity with the unavailable historical corpus.

The single-slot experiment uses four public 16-token contexts, obtained from the frozen corpus. Slot index 8 is the secret; other positions are public. Two instruction families supply profiling/validation contexts and two disjoint families supply primary-test contexts. All candidate classes are balanced within each context and repeat. The pilot draws eight IDs from the 8,431-type vocabulary with a fixed seed; the full design uses every type. They are distinct tasks from submitting the 5,000 original prompts.

For each condition, there are three fresh victim processes:

| Partition | Contexts | Repeats per context and candidate | Purpose |
|---|---|---:|---|
| `profile` | two profiling contexts | 25 | Estimate mean and sample standard deviation: 50 samples per candidate |
| `validation` | the same two contexts, new process | 2 | Diagnostic transfer across process restart; no automatic tuning |
| `new_session_new_context` | two held-out contexts, new process | 8 | Primary test; 16 observations per candidate |

The three conditions are real GPT-2 inference (`model`), real inference with the secret slot replaced by fixed ID 2134 (`input_blinded`), and the same transport with no inference (`constant`). Each phase randomizes condition order with a fixed seed; each repetition randomizes context/candidate requests. Conditions run sequentially in separate processes. Temporal drift can therefore confound comparisons; the paired interval does not remove this limitation. Follow-up confirmation needs independent sessions and, where practical, an interleaved design.

Warmup requests use sixteen EOS IDs and never enter calibration or accuracy denominators. Output length, thread count, dtype, checkpoint, seeds, time budget and request budget are declared in each configuration. UUID request IDs encode no class labels. These are execution units, not independent human prompts or independent machines.

## Profiling and scoring

Only `profile` labels and measured first-marker times estimate per-candidate means and standard deviations. The fixed score for candidate `c` is:

```
log_compatibility(c, t) = -0.5 * ((t - mean[c]) / max(sd[c], 0.001 ms)) ** 2
```

This is Gaussian compatibility without the density normalization `1/sd`; it is not a calibrated posterior probability. Computation in log space avoids exponential underflow. Ties select the lowest public token ID. There is no cache term, language-model prior, learned fusion, pruning or search over features. Validation/test labels enter scoring accuracy only after ranking. A unit test changes held-out labels and checks that profiles and saved predictions remain identical.

Top-1 and Top-5 classify one controlled token. Uniform baselines are `1/K` and `min(5,K)/K`; Top-5 is trivial in the two-candidate technical smoke. A complete evaluation needs every planned label and observation. Missing or malformed records fail the run instead of silently reducing the denominator. Saved raw socket reads preserve coalesced markers.

For nontechnical runs, the primary Top-1 has a 95% bootstrap interval across eight repetition blocks, 2,000 within-context/repetition label permutations with fixed predictions, and paired block intervals for `model - input_blinded` and `model - constant`. These are conditional on the specific candidates, four contexts, process sessions and hardware. Adjacent repeats are not proven independent; bootstrap blocks do not supply cross-machine or broad prompt-population uncertainty. There is no automatic declaration that a small p-value validates an attack. Analyze negative controls, effect magnitude, timing drift and independent replication together.

`registered_measurement` means this configuration is frozen in this repository before its intended collection. It does not mean external preregistration, advisor approval or completed execution. Editing a configuration or selecting a new feature creates a new experiment identity and requires fresh held-out measurements.

## Separate corpus trace

`corpus_trace` submits all 5,000 original inputs in randomized order and saves actual response-marker times, inputs and privileged server diagnostics separately. The full trace has 5,000 requests plus five warmups. Its summary is a latency distribution. It does not create 178,209 observed input-token times, infer input-token alignment or report reconstruction accuracy. A three-prompt CPU configuration tests this collection path technically.

## Budget and evidence

| Configuration | Role | Evaluation requests | Warmups |
|---|---|---:|---:|
| `timing_smoke_cpu_v2.json` | technical, GPT-2 small CPU | 36 | 9 |
| `timing_smoke_gpt2xl_cuda_v2.json` | technical, GPT-2 XL CUDA | 36 | 9 |
| `timing_pilot_gpt2xl_v2.json` | exploratory, eight candidates | 1,680 | 45 |
| `timing_corpus_smoke_cpu_v2.json` | technical, three full inputs | 3 | 1 |
| `timing_corpus5000_gpt2xl_v2.json` | diagnostic, 5,000 full inputs | 5,000 | 5 |
| `timing_8431_gpt2xl_v2.json` | frozen full candidate design | 1,770,510 | 45 |

The full candidate experiment is **not 5,000 inference calls**. It contains 421,550 profiling, 33,724 validation and 134,896 primary-test requests **per condition**. Budget guards reject it before model loading unless a sufficient `--max-requests` is explicit. Measure pilot throughput and estimate wall time, thermal drift and output storage before choosing a full execution window. The configured 24-hour collection cap is a failure limit, not a completion-time prediction.

Every run freezes source files, configuration, data, checkpoint hashes, host/runtime inventory, randomized design, raw socket observations, separate truth/diagnostic files, estimated profiles, predictions, counts, logs and final summaries. The source snapshot is what subprocesses execute. A fresh directory is required; failed or partial runs remain preserved. SQLite indexes large raw streams without loading them all into RAM. Analysis and source snapshots may contain sensitive host metadata in a real deployment and must be reviewed before any public release.

No result from this runner measures physical cache access, Flush+Reload, shared pages, a passive adversary, full prompt reconstruction, mitigation overhead, Pearson correlation or the historical fused accuracy. Physical validation needs its own sharing/primitive feasibility check, observer process, leakage model, controls and artifact contract. LLaMA or other architectures also require separate model and tokenizer contracts.
