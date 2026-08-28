# Validation report — Japanese Complete Transcription v1.0.0

Validation date: 2026-08-28
Branch: `public/japanese-complete-transcription-v1.0.0`
Authoritative green workflow run: `33165841169`
Validated head: `5badbe55f3b3323170b05cbaf804e58545067b8a`

## GitHub Actions result

Both jobs completed successfully:

```text
python-core  success
node-core    success
```

Python lane:

```text
compileall                              passed
complete-transcription tests            10 passed, 0 failed, 0 skipped
Python mora/N-best component tests       6 passed, 0 failed, 1 skipped
pip install . --no-build-isolation       passed
installed jtranscribe --help             passed
installed-package mora smoke test        passed
```

The single skipped component test exercises PyTorch backpropagation and is skipped because the lightweight public Python lane intentionally does not install Torch. It is not a failure.

Node lane:

```text
mora/contract/fusion tests               8 passed, 0 failed, 0 skipped
```

## Behaviors covered

- Japanese/ASCII segment joining without artificial Japanese spaces.
- Deterministic Unicode, spacing and punctuation normalization.
- Loopback-only Ollama endpoint enforcement.
- Rejection of endpoint query strings, remote hosts and cloud-routed models.
- Environment-proxy suppression and HTTP redirect blocking for local LLM traffic.
- Exact segment-ID validation for local-LLM responses.
- Similarity and length guards with deterministic fallback on excessive rewrite.
- RTTM parsing, word/segment overlap assignment and stable `話者1` labels.
- End-to-end fake-engine transcription without model weights.
- Immutable observed transcript SHA-256 verification and tamper detection.
- Separate normalized transcript linked to the observed hash.
- JSON, TXT, observed TXT, Markdown, SRT, VTT, TSV and word JSONL output creation.
- Privacy-preserving source metadata: absolute paths are omitted by default.
- Packaged mora tokenizer available after installation.
- Japanese-oriented defaults: language `ja`, VAD, word timestamps and `large-v3-turbo`.
- Build metadata and installed CLI entry point.

## Commands executed by the public workflow

```bash
python -m compileall -q japanese_transcriber scripts training tests
python -m unittest -v tests.test_complete_transcription
python -m unittest -v tests.test_python_components
python -m pip install --upgrade pip setuptools wheel
python -m pip install . --no-build-isolation
jtranscribe --help
python -c "from japanese_transcriber.reading import reading_and_mora; assert reading_and_mora('がっこう')[1] == ['ガ', 'ッ', 'コ', 'ウ']"
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
