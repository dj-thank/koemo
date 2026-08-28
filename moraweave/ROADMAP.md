# Roadmap

## Phase 0 — public executable contracts

- [x] immutable observed transcript
- [x] four-stream fusion and Grammar Honeytrap
- [x] selective re-listening planner
- [x] rights-gated hashed memory
- [x] local teacher probability cache
- [x] privacy-preserving data builders
- [x] Japanese evaluation metrics
- [x] model-free CI and clean wheel install

## Phase 1 — real-model acceptance

- [ ] pin faster-whisper and CTranslate2 revisions
- [ ] run Japanese short/long audio smoke tests
- [ ] validate Qwen3-ASR second-ear adapter against the pinned official package
- [ ] compare Qwen forced alignment with current word/mora timing
- [ ] measure CPU and GTX 1660 SUPER profiles

## Phase 2 — public-data benchmark

- [ ] exact Common Voice Japanese version manifest
- [ ] speaker-disjoint train/dev/test digest
- [ ] JMdict reading-memory ablation
- [ ] rights-cleared Aozora proper-noun/text-memory ablation
- [ ] SaSLaW or another permitted learner-speech evaluation lane
- [ ] ReazonSpeech lane after exact license review

## Phase 3 — Whisper internal modification

- [ ] train mora CTC only with frozen encoder baseline
- [ ] train upper encoder plus mora/phone heads
- [ ] add boundary and F0/accent tasks
- [ ] add preservation labels for fillers, restarts, and learner errors
- [ ] evaluate four-branch gate versus simple shared encoder

## Phase 4 — sparse intelligence

- [ ] calibrate entropy/disagreement thresholds
- [ ] share the first-pass model instance with span re-decoding
- [ ] cache encoder states for re-listening
- [ ] benchmark teacher probability cache hit rate
- [ ] route only unresolved spans to the second ASR
- [ ] implement abstention and human-review queue

## Phase 5 — release criteria

- [ ] observed CER non-regression
- [ ] Kana-CER and MLER improvement
- [ ] learner-error preservation improvement
- [ ] unsupported correction non-regression
- [ ] number/proper-noun non-regression
- [ ] reproducible rights/split/evaluation manifests
- [ ] signed Windows package only after target-machine validation
