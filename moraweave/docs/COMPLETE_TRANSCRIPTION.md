# Complete Japanese transcription

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[asr]'
```

Optional local teacher and public memory use only the Python standard library at runtime. Qwen3-ASR second-ear support requires:

```bash
python -m pip install -e '.[qwen]'
```

## Transcribe

```bash
python scripts/transcribe_audio.py meeting.m4a \
  --output-dir transcripts \
  --model large-v3-turbo \
  --initial-prompt '生成AIと日本語音声認識の技術会議です。' \
  --hotwords '森脇渉太,Kotodama,Qwen3-ASR,CTranslate2,モーラ'
```

GTX 1660 SUPERの開始候補:

```bash
python scripts/transcribe_audio.py meeting.m4a \
  --device cuda \
  --compute-type int8_float16 \
  --output-dir transcripts
```

CPU fallback:

```bash
python scripts/transcribe_audio.py meeting.m4a \
  --model small \
  --device cpu \
  --compute-type int8 \
  --output-dir transcripts
```

These hardware settings are starting profiles, not measured performance guarantees.

## Selective processing

The first pass uses faster-whisper long-form segmentation with VAD and word timestamps. A segment becomes uncertain when one or more conditions hold:

```text
avg_logprob below threshold
no_speech_prob above threshold
minimum word probability below threshold
candidate entropy high
four-stream winners disagree
```

Only uncertain segments receive high-beam N-best decoding. `--qwen-second-ear` optionally adds Qwen3-ASR to that lane. The local teacher receives only the candidate texts and returns closed-set probabilities.

## Hashed public memory

Build a memory database first, then pass it to transcription:

```bash
python scripts/build_jmdict_memory.py JMdict_e.xml \
  --database data/public_memory.sqlite3 \
  --rights-registry data/rights_registry.json

python scripts/transcribe_audio.py meeting.m4a \
  --memory-database data/public_memory.sqlite3
```

## Local teacher probability cache

```bash
python scripts/transcribe_audio.py meeting.m4a \
  --teacher-model qwen3:4b \
  --teacher-cache data/teacher_cache.sqlite3
```

The client:

- accepts loopback HTTP only;
- disables environment proxies;
- blocks redirects;
- rejects cloud-routed model names;
- requires every candidate ID exactly once;
- accepts probabilities, not rewritten observed text.

## Outputs

```text
<name>.moraweave.json  complete evidence record
<name>.observed.txt    spoken/observed Japanese
<name>.txt             readability derivative
<name>.srt             subtitles
<name>.vtt             WebVTT subtitles
```

Every normalized segment carries its corresponding observed evidence SHA-256. The normalized text may be more readable, but it is never substituted for the observed layer during pronunciation, error, or disfluency evaluation.

## Current limits

- The public CI does not download ASR weights or process real audio.
- The faster-whisper N-best adapter uses internal wrapped CTranslate2 interfaces; pin and retest dependency versions.
- Qwen3-ASR upstream APIs may change; the adapter is isolated and disabled by default.
- A trained mora/F0/preservation checkpoint is not included.
- Speaker diarization is an external evidence lane, not inferred by the core runner yet.
- Long-form first pass and N-best adapter currently instantiate separate faster-whisper model objects; production deployment should share the encoder/model instance to reduce VRAM.
