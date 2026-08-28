# Koemo Mora-Lattice Hybrid ASR

**Architecture decision record — 2026-08-28**

This document defines the research and implementation path for a Japanese ASR
system that keeps acoustic evidence, mora structure, linguistic ranking,
normalization, forced alignment, and pitch accent as distinct but composable
layers.

The non-negotiable invariant is:

> `observedTranscript` is an acoustic observation. `normalizedTranscript` is a
> separate linguistic product. A language model must never silently rewrite the
> observation.

## 1. Research basis used in this design

The architecture borrows mechanisms, not model scale, from recent public work:

1. **Qwen3.8-Flash-Next (2026-08-26)** — hybrid `3 × Gated DeltaNet + 1 ×
   Qwen Sparse Attention`, four-branch Gated Residual, N-gram Embedding, and a
   Muon/AdamW training split. This is an LLM architecture, so its transfer to
   speech is explicitly experimental.
2. **Bidirectional Recurrent Attention for Long-Form ASR**
   (arXiv:2506.19761) — speech-specific evidence that bidirectional recurrent
   attention can match attention accuracy while improving long-form throughput;
   also introduces Direction Dropout for one model to support uni- and
   bidirectional operation.
3. **Streaming SummaryMixing Conformer** (arXiv:2409.07165) and the original
   **SummaryMixing** paper (arXiv:2307.07421) — linear-time speech encoders with
   strong streaming/offline results and substantially lower memory use.
4. **Alternate Intermediate Conditioning with Syllable- and Character-level
   Targets for Japanese ASR** (arXiv:2204.00175) plus Self-Conditioned CTC
   (arXiv:2104.02724) — direct evidence for interaction between Japanese
   character and pronunciation-level intermediate targets.
5. **Language-Aware Intermediate Loss** (arXiv:2506.22846) — a frozen LLM can
   regularize intermediate CTC representations during training while retaining
   standard fast CTC inference.
6. **Qwen3-ASR / Qwen3-ForcedAligner** (arXiv:2601.21337) — a non-autoregressive
   forced aligner for arbitrary text units, used here as an independent teacher
   and comparator rather than a single source of truth.
7. **PASQA** (arXiv:2606.20137) — mora-conditioned fusion, ranking loss,
   localized pitch-accent error prediction, and speaker-invariant training for
   Japanese pitch-accent assessment.

Primary references:

- https://github.com/QwenLM/Qwen3.8-Flash-Next
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://arxiv.org/abs/2506.19761
- https://arxiv.org/abs/2409.07165
- https://arxiv.org/abs/2307.07421
- https://arxiv.org/abs/2204.00175
- https://arxiv.org/abs/2104.02724
- https://arxiv.org/abs/2506.22846
- https://arxiv.org/abs/2601.21337
- https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- https://arxiv.org/abs/2606.20137
- https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html

## 2. Final system shape

```text
16 kHz audio
   │
   ├─ VAD / channel policy / AEC
   │
   ├─ Existing faster-whisper decoder ───────────────┐
   │      └─ window N-best + CT2 scores              │
   │                                                  │
   └─ Whisper acoustic encoder hidden states          │
          │                                           │
          ├─ character CTC head                       │
          ├─ mora CTC head                            │
          ├─ boundary/alignment head                  │
          └─ F0 / voicing / accent heads              │
                  │                                   │
                  └────── mora lattice ◀──────────────┘
                              │
                       acoustic fusion
                              │
                  observedTranscript (locked)
                              │
                 LLM candidate-ID rank only
                              │
             normalizedTranscript (separate field)
                              │
           Qwen3 Forced Aligner comparison/teacher
```

The shared internal object is a **mora lattice**, not a string. Strings are views
of the lattice.

## 3. Canonical data contract

Every subsystem exchanges `moraUnits` with the same fields:

```json
{
  "unitId": "m0007",
  "surface": "きゃ",
  "reading": "キャ",
  "mora": "キャ",
  "kind": "mora",
  "source": "char_ctc",
  "textSpan": {"start": 7, "end": 9},
  "audioSpan": {"start": 0.62, "end": 0.76},
  "posterior": 0.91,
  "phonemes": ["ky", "a"],
  "alternatives": ["キヤ"],
  "sourceIndices": [7, 8],
  "accent": {
    "nucleusProbability": null,
    "phraseBoundaryProbability": null
  },
  "metadata": {}
}
```

Important distinctions:

- `surface`: original observed spelling.
- `reading`: canonical katakana reading.
- `mora`: one lexical mora. Empty for boundaries/noise/unknown units.
- `phonemes`: optional realized phones. They are not forced to equal the lexical
  mora because devoicing, gemination, and coarticulation alter realization.
- `audioSpan`: may be absent until alignment.
- `alternatives`: uncertainty is preserved instead of greedily destroyed.

The transcript-level object always has both fields:

```json
{
  "observedTranscript": "音声認識です",
  "normalizedTranscript": "音声認識です。",
  "observedCandidateId": "w0042-h00",
  "llmPreferredCandidateId": "w0042-h01",
  "moraUnits": [],
  "hypotheses": []
}
```

## 4. Encoder architecture: linear-first, selective-attention fallback

### 4.1 Why pure linear attention is not selected

Pure linear/recurrent attention is attractive for long meetings, but Japanese
proper nouns, homophones, and accent-sensitive distinctions need occasional
precise retrieval of distant evidence. Therefore the design uses a hybrid:

```text
Macro block = 3 × Bidirectional Gated Delta/Recurrent block
            + 1 × Audio Selective Sparse Attention block
```

This transfers Qwen3.8-Flash-Next's 3:1 pattern while replacing causal language
assumptions with speech-appropriate bidirectional and streaming modes.

### 4.2 Production and research variants

**Production baseline**

- SummaryMixing or bidirectional recurrent attention inside a Conformer-like
  block.
- Depthwise convolution retained for local acoustic structure.
- Offline mode: forward + backward state fusion.
- Streaming mode: causal state, bounded right context, and Direction Dropout
  during training.

**Research variant**

- Bidirectional Gated DeltaNet (`forward GDN` + `backward GDN`).
- Every fourth layer is **Audio-QSA**:
  - post-subsampling frames grouped into micro-blocks;
  - a lightweight indexer scores blocks;
  - always include a local window and a small set of global anchors;
  - retrieve a fixed budget of remote blocks.
- Compare this against SummaryMixing, bi-RWKV/bi-Mamba, limited-context
  attention, and the untouched Whisper encoder.

### 4.3 Four-stream gated residual

Qwen3.8's four-branch residual idea maps naturally to four ASR streams:

1. acoustic state;
2. character-orthographic state;
3. mora/pronunciation state;
4. prosody/F0/accent state.

Dynamic read gates decide which streams a block consumes; per-stream write gates
control updates. The first implementation should use a narrow bottleneck and
start with two streams (acoustic + mora), then enable all four after ablation.

### 4.4 Mora N-gram embedding

The Qwen N-gram Embedding concept is adapted as a small hashed table for mora
bigrams/trigrams. It is injected into the mora branch, not used as acoustic
truth. This supplies cheap lexical context such as common mora sequences while
remaining compatible with CTC decoding.

## 5. Six implementation stages

### Stage 1 — character CTC to canonical mora units

Input:

```text
collapsed or frame-level character CTC units
(symbol, posterior, start/end, frame span)
```

Processing:

1. Unicode NFKC normalization.
2. Hiragana-to-katakana canonical reading conversion.
3. Standard CTC duplicate/blank collapse when needed.
4. Small-kana composition (`キ` + `ャ` → `キャ`).
5. Preserve independent morae:
   - sokuon `ッ`;
   - moraic nasal `ン`;
   - long-vowel marker `ー`.
6. Preserve punctuation/noise/unknown units with non-mora kinds.
7. Merge timing and posterior without discarding source indices.

A kanji string is **not** converted to reading in this layer. A separate reading
provider may populate readings, but acoustic evidence remains traceable.

Acceptance criteria:

- deterministic unit IDs and JSON output;
- unit tests for contracted sounds, foreign-sound digraphs, long vowels,
  sokuon, moraic nasal, punctuation, and CTC blank separation;
- no source character or time span disappears without provenance.

### Stage 2 — true faster-whisper N-best and observed transcript

Increasing `beam_size` alone does not expose N-best through faster-whisper's
public `Segment` API. The underlying CTranslate2 Whisper generator supports
`num_hypotheses` and `return_scores`, so Koemo uses a thin adapter at the encoded
window level.

For each window, retain:

```text
candidate ID
text and token IDs
CT2 sequence score
length-normalized average log probability
no-speech probability
compression ratio
window/timestamp metadata
```

Candidate scores are not added raw. Each component is calibrated within the
N-best list and fused:

```text
S_acoustic(h) =
    λw z(log P_whisper)
  + λc z(log P_charCTC)
  + λm z(log P_moraCTC)
  + λa z(alignment quality)
  + λp z(prosody quality)
  + penalties(no-speech, repetition/compression, poor coverage)
```

`observedTranscript` is the argmax of **non-LLM acoustic/lattice evidence** and
is then locked.

Required metrics:

- 1-best CER/MER;
- oracle N-best CER/MER;
- reference recall@N;
- score calibration (ECE/Brier);
- real-time factor and peak memory.

### Stage 3 — LLM rank-only; normalization remains separate

The LLM receives candidate IDs and evidence, for example:

```json
{
  "task": "rank-only",
  "candidates": [
    {"id": "h0", "text": "...", "acousticScore": 1.21},
    {"id": "h1", "text": "...", "acousticScore": 1.17}
  ],
  "outputSchema": {
    "ranking": [{"candidateId": "h0", "rank": 1, "confidence": 0.82}]
  }
}
```

The response may contain only known candidate IDs, ranks, and confidence. Free
text generation is rejected. The default policy records
`llmPreferredCandidateId` but does not modify `observedTranscript`.

A separate deterministic or LLM-backed normalizer may write
`normalizedTranscript`; its method/version is logged.

Optional later experiment:

- permit the bounded LLM tie-break signal only when acoustic margins are small;
- maximum contribution 10–15% of the ranking scale;
- require an explicit ablation and a zero-regression gate on named entities,
  numbers, and negation before enabling it.

Training-only option: LAIL may use a frozen local LLM as an auxiliary loss. It
must not introduce an autoregressive LLM requirement at CTC inference time.

### Stage 4 — Qwen3 Forced Aligner as comparator and teacher

Run Qwen3-ForcedAligner on:

1. locked `observedTranscript`;
2. top-K acoustic candidates;
3. reference text during training/evaluation.

Convert all returned units to the same mora schema, then compare:

- boundary absolute error (mean/median/P95);
- coverage;
- monotonicity;
- overlap and gap rate;
- zero-duration lexical span rate;
- disagreement with CTC posterior peaks.

Qwen alignment is accepted into a fused result only when quality gates pass.
It is not the sole ground truth: public reports have shown zero-duration spans
for real lexical items in some current qwen-asr versions.

Consensus policy:

```text
CTC boundary + Qwen boundary agree       -> high-confidence fused boundary
Qwen passes gates, CTC uncertain         -> Qwen-assisted boundary
CTC strong, Qwen zero/overlap violation  -> retain CTC boundary
both uncertain                           -> unresolved + alternatives
```

The aligner is lazy-loaded and can be disabled, since it is much heavier than
Koemo's current final transcription path.

### Stage 5 — Whisper encoder with character/mora multi-task CTC

Attach heads to selected Whisper encoder layers through adapters first. Do not
immediately replace the production encoder.

Recommended losses:

```text
L = λchar L_CTC_character
  + λmora L_CTC_mora
  + λintermediate L_intermediate_CTC
  + λconsistency KL(project(character posterior → mora) || mora posterior)
  + λboundary L_boundary
  + λteacher L_alignment_distillation
  + optional λLAIL L_language-aware-intermediate
```

Training schedule:

1. freeze Whisper; train projections, adapters, and heads;
2. unfreeze the upper encoder third with a low learning rate;
3. enable self-conditioning from intermediate mora predictions;
4. only then evaluate full/partial encoder fine-tuning.

Character and mora heads must share time resolution. The character-to-mora
projection is many-to-one and preserves alternatives so the consistency loss is
computed over a lattice, not a single greedy string.

For the first GPU-constrained profile:

- adapter width: 512;
- 12 adapter blocks: 9 recurrent/linear + 3 selective-attention;
- mixed precision and gradient checkpointing;
- frozen base encoder until the heads have converged.

### Stage 6 — F0/accent heads and lattice fusion

Frame-level heads:

- voiced/unvoiced probability;
- continuous or quantized log-F0;
- energy;
- accent-nucleus probability;
- accent-phrase boundary probability.

Mora-conditioned fusion follows the PASQA direction:

```text
SSL/Whisper acoustic frames
      × cross-attention conditioned by mora sequence
      -> mora prosody embeddings
      -> accent ranking + localized error heads
```

Losses:

```text
L_prosody =
    λf0 masked smooth-L1(log-F0)
  + λv BCE(voicing)
  + λaccent ranking loss
  + λlocalization frame/mora error localization
  + λphrase phrase-boundary loss
  + λspeaker adversarial speaker-invariance loss
```

Synthetic accent corruption can generate paired correct/incorrect candidates,
which supplies ranking supervision without requiring every utterance to have a
manual accent annotation.

Prosody is a reranking signal, not a license to invent words. It should alter
candidate order only when candidate readings are acoustically confusable and
prosody evidence is sufficiently calibrated.

## 6. Training and optimizer policy

For the experimental hybrid adapter:

- Muon for eligible 2-D projection/mixing matrices;
- AdamW for embeddings, normalization parameters, biases, convolution kernels,
  and scalar gates;
- separate learning rates for frozen-base adapters, CTC heads, and prosody heads;
- gradient clipping and gate-value monitoring;
- start with conventional batch warmup until the speech-specific stability of
  the Qwen-style no-batch-warmup recipe is independently validated.

Qwen3.8's optimizer recipe is not copied blindly; it is an ablation target.

## 7. Evaluation matrix

### Recognition

- CER, WER, mora error rate (MER);
- rare-name/number/technical-term subsets;
- insertion/deletion/substitution breakdown;
- long-meeting accuracy by duration bucket.

### N-best and lattice

- oracle CER/MER at N = 2/4/8/16;
- reference recall@N;
- lattice density and pruning loss;
- acoustic-vs-LLM disagreement rate;
- observed-transcript mutation count (must remain zero by default).

### Alignment

- boundary MAE, median, P90, P95;
- unit coverage;
- zero-duration, overlap, and non-monotonic rates;
- CTC/Qwen consensus coverage.

### Prosody/accent

- F0 RMSE and correlation on voiced frames;
- voicing F1;
- accent-nucleus error rate;
- accent-phrase boundary F1;
- ranking accuracy for synthetic and human-rated accent errors.

### Systems

- real-time factor;
- first-result latency;
- peak VRAM/RAM;
- memory and time scaling versus utterance length;
- offline and streaming parity.

## 8. Mandatory ablations

1. untouched Whisper vs adapter-only;
2. full attention vs limited-context vs SummaryMixing vs recurrent attention vs
   bidirectional GDN;
3. pure linear stack vs 3:1 linear/sparse hybrid;
4. single residual stream vs two-stream vs four-stream gated residual;
5. character CTC only vs mora CTC only vs joint/self-conditioned CTC;
6. acoustic ranking vs LLM rank-only annotation vs bounded LLM tie-break;
7. CTC alignment vs Qwen alignment vs gated consensus;
8. no prosody vs F0 only vs mora-conditioned accent fusion.

No architecture is promoted because it is newer. It is promoted only after it
wins these controlled comparisons.

## 9. Repository layout

```text
koemo/asr/
  __init__.py
  schema.py             # moraUnits / hypotheses / transcript state
  mora.py               # CTC collapse and mora aggregation
  scoring.py            # calibrated acoustic fusion; locked rank-only LLM
  whisper_nbest.py      # low-level CT2 N-best window adapter

future:
  lattice.py
  normalizer.py
  qwen_aligner.py
  alignment_consensus.py
  models/hybrid_mora_encoder.py
  models/heads.py
  training/losses.py
  evaluation/metrics.py
```

## 10. Initial implementation status

Implemented in the first foundation change:

- canonical serializable `MoraUnit` and transcript schemas;
- standard CTC path collapse;
- small-kana/mora aggregation with timing/posterior provenance;
- special handling for `ッ`, `ン`, and `ー`;
- robustly calibrated multi-evidence acoustic ranking;
- stage-2 `observedTranscript` lock;
- rank-only LLM vote schema and bounded logging signal;
- separate `normalizedTranscript` setter;
- low-level CTranslate2 Whisper N-best adapter;
- unit tests covering the contracts above.

Not yet wired into the production `Transcriber`:

- full faster-whisper VAD/window/timestamp orchestration for N-best;
- an actual character CTC inference source;
- Qwen3 Forced Aligner runtime;
- trainable Whisper adapters/heads;
- F0/accent extraction and lattice fusion.

The production 1-best path remains untouched until the N-best and mora outputs
can be compared side-by-side on real Japanese recordings.
