# Contributing

Contributions are welcome. Preserve the central invariant:

```text
observedTranscript != normalizedTranscript
```

An observed transcript must remain derived from acoustic evidence. Readability normalization, including a local LLM, is always a separate derivative linked through `observedSha256`.

Before opening a pull request, run:

```bash
python -m compileall -q japanese_transcriber scripts training tests
python -m unittest -v tests.test_complete_transcription
python -m unittest -v tests.test_python_components
npm test
```

Requirements for changes:

- transcription format changes must update `schemas/complete-transcript.schema.json`;
- new formatters need model-free end-to-end tests;
- changes to speaker assignment need RTTM overlap fixtures;
- local-LLM changes must preserve loopback-only defaults, exact segment IDs and rewrite guards;
- mora changes must retain shared Python/Node fixture coverage;
- model-quality claims require a named held-out corpus, metrics and reproducible configuration.

Do not commit model weights, learner recordings, API keys, local `.env` files, absolute private paths or generated transcripts containing personal data.
