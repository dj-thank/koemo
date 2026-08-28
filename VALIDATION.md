# Validation report — Japanese Complete Transcription v1.0.0

Validation date: 2026-08-28
Branch: `public/japanese-complete-transcription-v1.0.0`

## Executed before publication

```bash
python -m unittest -v tests.test_complete_transcription
python -m compileall -q japanese_transcriber tests
python -m japanese_transcriber --help
```

Observed result:

```text
Complete-transcription unit tests: 8 passed, 0 failed, 0 skipped
Python compileall: passed
CLI parser/help smoke test: passed
```

## Behaviors covered

- Japanese/ASCII segment joining without artificial Japanese spaces.
- Deterministic Unicode, spacing and punctuation normalization.
- Loopback-only Ollama endpoint enforcement.
- Exact segment-ID validation for local-LLM responses.
- Similarity and length guards with deterministic fallback on excessive rewrite.
- RTTM parsing, word/segment overlap assignment and stable `話者1` labels.
- End-to-end fake-engine transcription without model weights.
- Immutable observed transcript SHA-256 verification and tamper detection.
- Separate normalized transcript linked to the observed hash.
- JSON, TXT, observed TXT, Markdown, SRT, VTT, TSV and word JSONL output creation.
- Privacy-preserving source metadata: absolute paths are omitted by default.
- Japanese-oriented defaults: language `ja`, VAD, word timestamps and `large-v3-turbo`.

## Public CI

The branch workflow additionally runs:

```bash
python -m compileall -q japanese_transcriber scripts training tests
python -m unittest -v tests.test_complete_transcription
python -m unittest -v tests.test_python_components
npm test
```

## Not claimed as validated in the model-free build

- Real faster-whisper/CTranslate2 model inference.
- Real requests to a running Ollama daemon.
- External diarization model inference; only RTTM import/assignment is tested.
- Real pyopenjtalk dictionary/model execution.
- Fine-tuning with a real Whisper checkpoint.
- CER, kana-CER, mora-label error rate or learner-error-preservation benchmarks.

These require target hardware, model weights and a held-out Japanese audio corpus. The code reports this boundary explicitly rather than presenting model-free tests as recognition-accuracy evidence.
