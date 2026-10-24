# Reproducibility and provenance

The upstream baseline is `071bf8ee26ea0ebf0f4e539cf87c0a07ee68245f`, which adds requirements to the previous `771754fa99f5b661a4480b3c5d173aa328adc9e9` package. Its scientific `src/`, data and paper remain preserved in this fork. The inherited root requirements use a scikit-learn pin unavailable in the original reproduction environment; the laboratory uses tested `1.4.1.post1` and separate complete lock files.

## Three separate claims

1. **Public synthetic replay:** eight external timing-noise settings and ten measurement rounds, 800,000 positions. The published CSV and metadata were reproduced; ten zero-noise rounds also matched the original computation bit for bit under the validated acceleration. Zero external noise means 65.453% Top-1 and 70.810% Top-5. The figure is not 73.4%/91.2%, and the experiment does not implement the textual SPC-5K protocol.
2. **Independent simulation:** new controlled corpus, declared vocabulary/calibration, estimated profiles, training-only fusion, nine ablations/controls and uncertainty under fixed corpus/channel. Historical SPC-5K identity is not established. Strong synthetic cache separation explains the high score; a collision sensitivity probes that assumption.
3. **Measured observation pilot:** GPT-2 small CPU, fixed-size marker streaming and client timing; a public fixed input slot among eight candidates. This is a feasibility measurement with different target/access/model than complete prompt reconstruction or the historical simulation. Physical cache and production observation are not measured.

The original manuscript's results and publication metadata are claims preserved from upstream; software reproduction does not independently validate all of them. The upstream README is archived verbatim in `docs/upstream/README_071bf8e.md`.

## What makes an execution inspectable

Each new run gets a unique directory, source/configuration snapshot, pinned image/runtime versions, IDs and explicit partitions, raw observations, predictions, denominators and hashes. Primary numerical targets do not select parameters. Repeating a reduced pilot establishes that pilot's replay, rather than validating every possible future run.

The portable simulation pilot matched nineteen scientific files byte for byte: corpus/splits/vocabulary plus raw profile measurements, profiles, training pairs, weights, observations, predictions, candidate audit and metrics in two seeds. Changes to asset paths are packaging changes; checkpoint/tokenizer identities remain explicit.

External audits distinguish full counts/retokenization/profile checks from sampled ranking/edit-distance checks. Confidence intervals identify their resampling unit. Zero-width intervals under a fixed deterministic control do not represent certainty about a new population, machine or deployment.

All newly published corpus words are public dictionary material. Only technical experimental evidence is exported. Private conversations, personal records, dissertation chapters and authentication state are not part of the public package.
