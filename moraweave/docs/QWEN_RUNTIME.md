# Qwen runtime integration

## Qwen3-ASR second ear

MoraWeave treats Qwen3-ASR as an independent evidence source, not as a replacement for Whisper evidence.

Pinned audit target:

```text
QwenLM/Qwen3-ASR@7c6daf77a2421100f5fb066495372c00129d39ff
qwen-asr API family: 0.0.6
```

Supported adapter behavior:

- `ja`, `jpn`, `jp`, `Japanese`, `日本語` → official `Japanese` language name;
- `auto`/empty → no forced language;
- exact span decoding by loading only the requested range;
- optional timestamp request;
- model/runtime version provenance;
- one transcript per input from the official high-level wrapper.

The high-level Qwen wrapper is not presented as decoder N-best. Independent Qwen output enters the lattice as a second-source candidate. If its text agrees with an existing candidate, source-support diversity is recorded.

Example:

```bash
python -m pip install -e '.[qwen]'

moraweave-transcribe audio.m4a \
  --qwen-second-ear \
  --qwen-model Qwen/Qwen3-ASR-0.6B \
  --qwen-device-map cuda:0 \
  --qwen-dtype float16
```

On low-memory hardware, keep `max_inference_batch_size=1` and do not keep Whisper, Qwen ASR and a large teacher resident simultaneously.

## Qwen3 Forced Aligner

The aligner accepts the audio span, candidate text and official language name. It is used to localize evidence; forced alignment does not prove that every supplied token was actually spoken.

Use both:

```text
free mora/ASR evidence      what appears to have been spoken
forced alignment evidence   where a supplied hypothesis can be placed
```

A deletion or insertion claim requires comparison between these lanes.

## Qwen3.8-Flash-Next local teacher

Pinned research target:

```text
QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526
```

MoraWeave connects to a separately served local model through a loopback OpenAI-compatible endpoint. It does not download the model automatically and public CI does not run the weights.

```bash
moraweave-transcribe audio.m4a \
  --teacher-protocol openai \
  --teacher-model Qwen/Qwen3.8-Flash-Next \
  --teacher-endpoint http://127.0.0.1:8000/v1/chat/completions
```

Teacher contract:

```json
{
  "probabilities": [
    {"id": "candidate-a", "p": 0.55},
    {"id": "candidate-b", "p": 0.45}
  ],
  "abstain": false
}
```

Restrictions:

- candidate IDs must be an exact permutation of the request set;
- probabilities are finite, bounded and normalized;
- the teacher may abstain;
- no free transcript is accepted;
- no chain-of-thought is stored;
- an abstained result remains abstained in cache;
- teacher preference cannot directly author observed text;
- grammar-honeytrap logic penalizes language preference unsupported by acoustic+mora evidence.

## Qwen3.8 idea translation

The following are architecture translations, not direct reproductions:

```text
query-selected state access → ambiguity-selected acoustic/evidence requests
gated branch composition    → four evidence-stream gate
lexical augmentation        → rights-gated hashed n-gram memory
long-context efficiency     → consensus locking + span cache
```

The project must not describe these as QSA, DeltaNet or Qwen3.8 kernels unless those kernels are actually implemented and benchmarked.

## Failure policy

If Qwen is unavailable:

- base Whisper observation remains available;
- missing evidence increases risk/lowers coverage;
- the observation may become provisional;
- no placeholder score is fabricated;
- cache misses never silently become positive evidence.
