# Japanese Complete Transcription SSOT — mora and transcript evidence

The mora-aware ASR contract remains an underlying layer of the complete Japanese transcription system.

Central invariant:

```text
observedTranscript != normalizedTranscript
```

The authoritative complete document contract is:

```text
schemas/complete-transcript.schema.json
```

The observed layer contains the ASR text, segment and word timestamps, acoustic confidence evidence, speaker assignments, readings and mora annotations. Its canonical payload is protected by SHA-256 and verified by `japanese_transcriber.pipeline.verify_observed_integrity`.

The normalized layer is a derivative for readability. It stores `observedSha256`, may use deterministic normalization or guarded local Ollama normalization, and never overwrites observed evidence.

Mora rules:

- small-kana compounds such as `キャ`, `ティ`, `ファ` are one mora;
- `ン`, `ッ`, and `ー` are each one independent mora;
- missing acoustic timing remains null;
- kanji readings require explicit pyopenjtalk opt-in;
- pronunciation and learner-error evaluation must use observed evidence, not only normalized text.
