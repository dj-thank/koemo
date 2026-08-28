# Changelog

## 1.0.0 — 2026-08-28

- Expanded the mora-aware ASR core into a complete Japanese transcription application.
- Added long-form faster-whisper transcription with VAD, word timestamps, prompts and hotwords.
- Added immutable observed transcript evidence and a separately stored readable transcript.
- Added guarded loopback-only Ollama normalization with exact segment-ID and similarity checks.
- Added TXT, observed TXT, JSON, Markdown, SRT, WebVTT, segment TSV and word JSONL outputs.
- Added optional RTTM speaker-label import and stable Japanese speaker labels.
- Added segment/word confidence evidence and explicit uncertainty reasons.
- Added optional pyopenjtalk readings and mora annotations for kanji-containing words.
- Added batch file/directory CLI and atomic output writes.
- Added model-free end-to-end tests and public CI.

## 0.3.0 — 2026-08-28

- Added single-utterance CTranslate2 N-best extraction for faster-whisper.
- Added local Ollama structured candidate ranking.
- Added Whisper mora/phone CTC and boundary heads with shared encoder execution.
- Added immutable observed/normalized transcript contracts and mora-aware scoring utilities.
