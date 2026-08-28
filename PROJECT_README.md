# Japanese speaking assessment — mora-aware ASR core

This repository is a tested development bundle for the existing
`japanese-speaking-assessment-poc`. It now covers three layers:

1. a canonical Japanese mora representation shared by Python and Node.js;
2. true CTranslate2 N-best extraction plus local-LLM rank-only normalization;
3. a PyTorch Whisper extension with mora CTC, optional phone CTC, and boundary
   heads that shares the pretrained Whisper encoder with the text decoder.

The design keeps **what the learner actually said** separate from a readable or
contextually preferred normalization.

```text
observedTranscript != normalizedTranscript
```

## Implemented

### Runtime ASR path

- NFKC and hiragana-to-katakana normalization.
- Correct one-mora handling for `キャ`, `ティ`, `ファ`, and related sequences.
- Independent mora handling for `ン`, `ッ`, and `ー`.
- Character-CTC/forced-alignment aggregation into timed `moraUnits`.
- Duration-weighted mora confidence aggregation.
- CTranslate2 `num_hypotheses` decoding through `scripts/whisper_nbest.py`.
- Preservation of sequence score, reconstructed average log probability, token
  IDs, source-waveform hash, model-input hash, and every distinct decoded
  candidate. Waveform hashes use canonical little-endian float32 bytes.
- Deliberate rejection of long-form N-best approximation: the adapter accepts
  one Whisper window only, normally at most 30 seconds.
- Deterministic acoustic selection that excludes local-LM scores.
- Immutable observed transcript, candidates, mora units, and uncertainty spans,
  protected by SHA-256.

### Local LLM path

- Local Ollama `/api/chat` integration with JSON-schema structured output.
- Candidate-ID ranking only; the model cannot return replacement text.
- Application-side verification that every original candidate ID appears once.
- Loopback-only endpoint policy by default.
- Cloud-routed Ollama model names blocked by default.
- Separate normalized ranking combining acoustic support, local-LM rank, and
  edit similarity.

### Whisper architecture path

- One shared Whisper encoder pass.
- Original autoregressive Whisper text decoder retained.
- Mora CTC head.
- Optional phone CTC head.
- Binary mora-boundary head.
- Weighted multitask loss and strict CTC target validation.
- CTC greedy decoder.
- Candidate-specific CTC forward scoring that produces `moraCtcScore` for
  N-best fusion.
- Exact-ID Node adapter that attaches the CTC score rows without changing ASR
  candidate text.
- Greedy CTC frame spans converted to canonical timed `moraUnits`.
- Versioned mora vocabulary and auxiliary-head save/load.
- Lazy Hugging Face `from_pretrained` integration.

## Test and static validation

```bash
python -m pip install "numpy>=1.24" "torch>=2.4" "jsonschema>=4.20"
npm run test:all
npm run check
```

The core mora and contract suites need no model weights. The N-best tests use a
fake CTranslate2 result; the multitask tests use a small fake Whisper-like
encoder-decoder. A separate optional test instantiates a tiny Hugging Face
Whisper model when `transformers` is installed.

The included examples in `fixtures/` are also validated against the Draft
2020-12 JSON Schemas.

## Mora segmentation

```bash
python scripts/japanese_mora.py segment --text "がっこう"
```

Expected sequence:

```text
ガ / ッ / コ / ウ
```

Character-level alignment can be converted to timed mora units:

```bash
python scripts/japanese_mora.py merge-align \
  --input fixtures/char_alignment_kyou.json
```

The `き` and `ょ` spans become one `キョ` mora. The following `ウ` remains a
separate mora.

## Extract real faster-whisper N-best candidates

Install the optional ASR dependencies in the same Python environment used by
the PoC, then run one assessment utterance:

```bash
python scripts/whisper_nbest.py answer.wav \
  --model small \
  --language ja \
  --beam-size 8 \
  --num-hypotheses 5 \
  --output answer.nbest.json
```

The adapter reuses faster-whisper's feature extractor, tokenizer, prompt
builder, encoder, and underlying CTranslate2 Whisper model. It does not patch
site-packages. `without_timestamps=true` is enforced for this first N-best
contract; word and mora time boundaries are produced by the separate aligner.

## Rank candidates with a local Ollama model

Start Ollama locally and configure a locally stored model. Then:

```bash
LOCAL_LM_MODEL=your-local-model \
node scripts/ollama_rerank.mjs \
  --input answer.nbest.json \
  --context "学校生活について話しています" \
  --output answer.reranked.json
```

The default endpoint is `http://127.0.0.1:11434/api/chat`. Remote endpoints and
cloud-routed model names fail closed unless an application explicitly opts in.

## Add the modules to the existing local PoC

From this bundle:

```powershell
.\integration\apply-to-local-poc.ps1
```

The equivalent explicit command is:

```powershell
python integration\apply_bundle.py `
  "$HOME\Workspace\Active\japanese-speaking-assessment-poc"
```

The installer is additive and non-destructive. Existing files are skipped
unless `--force` is supplied. Then apply the wiring in
`integration/INTEGRATION.md` to the real function names in `server.mjs`,
`src/scoring.mjs`, `src/fluency.mjs`, and `scripts/gop_align.py`.

## Instantiate the mora-aware Whisper model

```python
from training.whisper_mora_multitask import (
    WhisperMoraMultiTaskConfig,
    WhisperMoraMultiTaskModel,
)

model = WhisperMoraMultiTaskModel.from_pretrained(
    "openai/whisper-small",
    multitask_config=WhisperMoraMultiTaskConfig(
        mora_vocab_size=len(mora_vocabulary),
        mora_blank_id=0,
        phone_vocab_size=None,
    ),
)
```

Training batches provide normal Whisper `text_labels` plus `mora_labels` and,
when available, `boundary_labels` and `phone_labels`. See `SSOT_MORA.md` for the
loss contract.

After inference, `training/mora_ctc_runtime.py` connects the auxiliary head to
the runtime pipeline: `score_mora_candidates` scores each candidate reading,
and `greedy_mora_spans` plus `mora_spans_to_units` emit approximate 20 ms-frame
mora boundaries.

## Honest limits of this bundle

- The target `japanese-speaking-assessment-poc` source tree was not part of this
  standalone bundle, so existing application files are never rewritten
  implicitly. Use the additive installer and then wire the documented integration
  points against the target project's actual function names.
- Real faster-whisper model inference was not run in this build environment;
  no faster-whisper package or model weights were installed here. The adapter is
  unit-tested against the current upstream API shape and should be verified once
  against the PoC's pinned environment and learner-audio evaluation set.
- The multitask architecture and gradients were executed with PyTorch against a
  fake Whisper-like model. The optional current-Transformers smoke test is in CI,
  but real checkpoint fine-tuning still requires a labeled Japanese corpus.
- A mora CTC head improves phonological evidence; it does not by itself guarantee
  correct kanji choice or preserve every learner error. Evaluation must report
  CER, kana-CER, mora-label error rate, boundary error, and learner-error
  preservation separately.
