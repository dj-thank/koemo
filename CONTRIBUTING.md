# Contributing

Contributions are welcome. Keep changes narrowly scoped and preserve the central
invariant:

```text
observedTranscript != normalizedTranscript
```

An observed transcript must remain derived from acoustic evidence. A local LLM
may rank existing candidates for normalization, but it must not silently rewrite
or replace the immutable observed record.

Before opening a pull request, run:

```bash
python -m pip install "numpy>=1.24" "torch>=2.4" "jsonschema>=4.20"
npm run test:all
npm run check
```

Changes to mora segmentation must add shared fixture coverage for both Python and
Node.js. Changes to schemas must update and validate all examples in `fixtures/`.
Changes to the Whisper multitask layer must include forward, backward, padding,
and persistence tests where applicable.

Do not commit model weights, learner recordings, API keys, local `.env` files, or
machine-specific absolute paths.
