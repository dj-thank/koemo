# MoraWeave frontier architecture v0.2

## Objective

Recover the Japanese that was actually spoken while also producing a readable derivative. The two products are different evidence objects:

```text
observedTranscript   acoustic decision; immutable; may preserve errors and fillers
normalizedTranscript readability derivative; linked to the observed evidence hash
```

## System graph

```text
audio/video
  │
  ├─ VAD / long-form window planner
  │
  ├─ faster-whisper CTranslate2 N-best
  │      ├─ sequence score
  │      ├─ average log probability
  │      ├─ rank / hypothesis count
  │      └─ prompt/hotword provenance
  │
  ├─ mora shadow
  │      ├─ text reading
  │      ├─ mora CTC score
  │      ├─ special-mora evidence
  │      └─ optional forced-alignment timeline
  │
  ├─ rights-gated lexical memory
  ├─ filler/repair/error-preservation evidence
  │
  ▼
calibration layer
  ├─ held-out scalar profiles
  ├─ robust median/MAD fallback
  └─ beam score-rank confidence
  │
  ▼
four-stream evidence gate
  ├─ acoustic
  ├─ mora
  ├─ lexical
  └─ preservation
  │
  ├─ posterior
  ├─ entropy
  ├─ stream disagreement
  ├─ evidence coverage
  ├─ selective risk
  └─ accept / provisional
  │
  ▼
dual evidence lattice
  ├─ locked consensus spine
  └─ contradiction islands
          ├─ number/date/currency
          ├─ proper noun/technical term
          ├─ particle
          └─ special mora
  │
  ▼
evidence acquisition scheduler
  utility = expected information gain / estimated cost
  ├─ Whisper span re-listen
  ├─ Qwen3-ASR second ear
  ├─ Qwen3 Forced Aligner
  └─ local teacher check
  │
  ▼
immutable observed transcript
  │
  └─ delayed, candidate-only local Qwen teacher
          ├─ probabilities over existing IDs
          ├─ abstention
          └─ normalized derivative only
```

## Why four evidence streams

### Acoustic

Contains decoder evidence closest to the waveform. This family has a configured floor and cannot be overruled solely by language-model naturalness.

### Mora

Represents Japanese timing and phonology. It can distinguish a grammatical correction that is not supported by the spoken mora sequence and can collapse surface variants sharing one reading.

### Lexical

Supports domain terms and proper nouns. It is rights-gated and must not be interpreted as acoustic proof.

### Preservation

Rewards retention of fillers, repetitions, repairs and learner errors when the waveform supports them. It prevents “clean Japanese” from being confused with “accurate transcription.”

## Grammar honeytrap

A candidate receives a penalty when a teacher strongly prefers it but acoustic+mora support is lower beyond a deadband. Teacher probability is not a positive acoustic score.

```text
unsupported_preference = max(0, teacher - acoustic_family - deadband)
penalty = strength × unsupported_preference
```

## Uncertainty decomposition

```text
aleatoric proxy  = candidate posterior entropy
epistemic proxy  = evidence-stream disagreement
missing evidence = one minus available gate weight
selection margin = top posterior minus second posterior
```

These quantities are proxies and require held-out calibration. The system therefore stores them with a calibration digest rather than presenting them as universally calibrated probabilities.

## Selective risk

MoraWeave may abstain from a final assertion and emit a provisional observation. This is preferable to silently selecting a fluent but weakly supported sentence.

The acceptance policy is evaluated with a risk-coverage curve. “Accuracy at 100% coverage” and “risk at operational coverage” must both be reported.

## Dual lattice

The lattice uses mora units only when every candidate has a reading/mora shadow. Otherwise it falls back to normalized surface characters and records that fallback.

Locked consensus is never re-decoded unless a later integrity check invalidates it. Compute is concentrated on contradiction islands.

## Runtime cache v2

The SQLite cache stores evidence, not raw audio. Its key includes all settings that can change the result. A cache hit is invalid when context, hotwords, calibration or model identity changes.

Teacher cache entries store the `abstained` flag. An abstention can never be replayed as a positive decision.

## Long-form strategy

- default window: 28 seconds;
- overlap: 1.2 seconds;
- exact overlap stitching for Japanese text;
- independent evidence object per window;
- cached span-level re-listening;
- optional Qwen second ear only on ambiguous windows;
- global evidence hash over ordered window evidence.

Long-form windows are an orchestration device, not a claim that 28 seconds is universally optimal. Window/overlap ablations are required for different domains.

## Security and privacy boundaries

- no raw audio or weights in Git;
- absolute source paths excluded from exported JSON by default;
- loopback-only teachers;
- proxy and redirect bypass blocked;
- candidate ID set checked exactly;
- canonical evidence hashes reject mutation;
- public-data rights state is executable and fail-closed.
