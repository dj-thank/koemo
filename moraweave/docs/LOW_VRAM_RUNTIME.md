# Low-VRAM runtime

The generic adapter runtime is modular but may instantiate a first-pass faster-whisper model and a separate N-best adapter model. On a 6 GB GPU this can exceed the available memory.

The recommended constrained-GPU entrypoint is:

```bash
python -m moraweave.transcribe_low_vram recording.m4a \
  --device cuda \
  --compute-type int8_float16 \
  --model large-v3-turbo \
  --output-dir transcripts
```

CPU fallback:

```bash
python -m moraweave.transcribe_low_vram recording.m4a \
  --device cpu \
  --compute-type int8 \
  --model small \
  --output-dir transcripts
```

## What is shared

- one `WhisperModel` instance;
- one decoded 16 kHz waveform;
- first-pass long-form segmentation;
- selective CTranslate2 N-best decoding for uncertain spans;
- optional hashed lexical memory;
- optional local teacher probabilities;
- query-selected acoustic cache.

The runtime does **not** promise that `large-v3-turbo` will fit every 6 GB setup. CUDA, CTranslate2 version, compute type, other processes, and driver allocation matter. Begin with `small` or `medium` if allocation fails, then measure peak VRAM on the target machine.

## Acoustic cache

```bash
python -m moraweave.transcribe_low_vram recording.m4a \
  --acoustic-cache data/acoustic_cache.sqlite3
```

A cache key includes the source audio digest, time span, model, beam settings, language, prompt digest, and hotword digest. It stores candidates only, not waveform bytes. Repeating the same request can therefore avoid a second GPU decode while changes in context or decoding settings create a distinct key.

## Why Qwen second ear is disabled here

The low-VRAM runtime keeps a single Whisper family resident and disables Qwen3-ASR by design. A second ASR can still be run in a sequential process after releasing Whisper, or on another device. This avoids pretending that two large model families fit simultaneously on constrained hardware.
