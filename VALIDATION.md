# Validation report — bundle 0.3.0

Validation date: 2026-08-28
Target project: `japanese-speaking-assessment-poc`
Bundle: `japanese-speaking-assessment-mora-core`

## What was executed

```text
Node.js       v22.16.0
npm           10.9.2
Python        3.13.5
PyTorch       2.10.0+cpu
NumPy         2.3.5
jsonschema    4.26.0
```

Commands:

```bash
npm run test:all
npm run check
```

Results:

```text
Node test runner        30 passed, 0 failed, 0 skipped
Python unittest         45 passed, 0 failed, 1 skipped
Combined                75 passed, 0 failed, 1 skipped
Node syntax checks      passed
Python byte compilation passed
```

The skipped test is the optional real Hugging Face `WhisperForConditionalGeneration`
compatibility smoke test. `transformers` was not installed in this isolated build
environment. An installation attempt could not reach PyPI because outbound package
networking was unavailable. The same test is retained as a dedicated CI lane in
the complete source archive.

## Behaviors covered by tests

- Shared Python/Node mora segmentation fixtures.
- Small-kana composition, moraic nasal, geminate, and long-vowel handling.
- Character alignment to timed mora aggregation and confidence weighting.
- Canonical `MoraUnit` validation.
- Acoustic-only observed candidate selection.
- Immutable observed transcript, candidate set, mora evidence, and uncertainty
  spans protected by SHA-256.
- Rank-only local-LLM normalization that cannot invent candidate text.
- Loopback-only Ollama endpoint default and cloud-model-name rejection.
- Structured candidate-ID response validation.
- Source-waveform and post-VAD model-input hash separation.
- CTC forward scoring of every closed-set candidate reading and exact-ID score attachment.
- Greedy CTC frame spans converted to canonical timed mora units.
- One-window N-best duration rejection.
- CTranslate2 result score reconstruction and duplicate-text collapse.
- One shared encoder pass for text, mora CTC, phone CTC, and boundary heads.
- Forward loss calculation, backpropagation, CTC collapse, and auxiliary-head
  save/load.
- Rejection of blank IDs inside CTC targets, inconsistent CTC padding, invalid
  boundary labels, and labels extending beyond encoder lengths.
- Draft 2020-12 validation of N-best, transcript, mora-unit, and multitask
  configuration examples.
- Additive installer copy verification, idempotence, dry-run behavior, and
  no-overwrite default.

## Not executed in this environment

- Real faster-whisper/CTranslate2 model inference and model-weight download.
- A real request to a running local Ollama daemon.
- Fine-tuning with a real Whisper checkpoint.
- Learner-speech benchmark evaluation.
- Direct modification of any external `japanese-speaking-assessment-poc` tree.
  This release is intentionally additive and requires explicit target-path and
  integration decisions by the operator.

## Required target-machine acceptance checks

1. Pin the PoC's working `faster-whisper` and CTranslate2 versions.
2. Run `scripts/whisper_nbest.py` against at least one short Japanese recording.
3. Validate the resulting JSON with `schemas/asr-nbest.schema.json`.
4. Run the local Ollama reranker with a locally stored model and confirm no
   candidate text changes.
5. Wire the additive modules according to `integration/INTEGRATION.md`.
6. Run the complete existing PoC test suite plus this bundle's tests.
7. Evaluate CER, kana-CER, mora-label error rate, boundary error,
   learner-error-preservation rate, unsupported-correction rate, and runtime
   factor on held-out learner speech before enabling the new path by default.
