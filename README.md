# Japanese speaking assessment — mora-aware ASR core v0.3.0

Public source distribution for a mora-aware Japanese ASR and speaking-assessment core.

The implementation separates the acoustically observed transcript from any
LLM-assisted normalization:

```text
observedTranscript != normalizedTranscript
```

## Included release

- `japanese-speaking-assessment-mora-core-v0.3.0-public.zip` — complete 58-file
  source tree, tests, schemas, CI workflow, documentation, and additive PoC
  installer.
- `PUBLICATION_MANIFEST.json` — source commit/tree IDs, artifact metadata, and
  test summary. The archive contains the complete per-file source manifest.
- `SHA256SUMS.txt` — release archive checksum.
- `PROJECT_README.md` — full usage and architecture documentation.
- `SSOT_MORA.md` — points to the canonical SSOT packaged in the release archive.
- `VALIDATION.md` — executed and unexecuted validation scope.

## Verified before publication

```text
Node.js tests       30 passed, 0 failed
Python tests        45 passed, 0 failed, 1 optional compatibility test skipped
Syntax checks       passed
JSON Schema checks  passed
```

The skipped test requires the optional `transformers` dependency. A dedicated
GitHub Actions lane is included in the source archive.

## Quick start

Download and extract the ZIP, then run:

```bash
python -m pip install "numpy>=1.24" "torch>=2.4" "jsonschema>=4.20"
npm run test:all
npm run check
python scripts/japanese_mora.py segment --text "がっこう"
```

Expected mora sequence:

```text
ガ / ッ / コ / ウ
```

## Major components

- canonical Japanese mora segmentation in Python and Node.js;
- character-alignment aggregation into timed `moraUnits`;
- true CTranslate2/faster-whisper N-best extraction for one utterance window;
- immutable observed-transcript contract protected by SHA-256;
- loopback-only Ollama candidate ranking with no free rewrite by default;
- shared Whisper encoder with mora CTC, optional phone CTC, and boundary heads;
- CTC candidate scoring and approximate mora-frame timing;
- additive, non-destructive integration tooling for an existing PoC.

## License

MIT © 2026 SHOTA Moriwaki.
