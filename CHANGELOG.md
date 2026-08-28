# Changelog

## 0.3.0 — 2026-08-28

- Added single-utterance CTranslate2 N-best extraction for faster-whisper.
- Added score reconstruction compatible with faster-whisper's average-logprob
  calculation, duplicate-text collapse, and fail-closed duration handling.
- Added local Ollama structured candidate ranking with loopback and cloud-model
  safety checks.
- Added end-to-end observed/normalized transcript pipeline.
- Added Whisper mora/phone CTC and boundary heads with shared encoder execution.
- Added versioned mora vocabulary, auxiliary-head persistence, schemas, and CI
  compatibility lanes.
- Added separate source-waveform and post-VAD model-input hashes with a stable
  float32 little-endian representation.
- Added strict CTC/boundary padding checks, schema example validation, and an
  idempotent SHA-256-verifying installer with PowerShell entry point.
- Added CTC forward scoring for N-best candidate readings and greedy CTC span
  conversion to canonical timed mora units.
- Added exact candidate-ID validation when attaching Python CTC score rows to
  the Node ASR fusion path.

## 0.1.0 — 2026-08-28

- Added canonical mora segmentation and character-alignment aggregation.
- Added immutable transcript contract, deterministic candidate fusion, and
  mora-derived fluency metrics.
