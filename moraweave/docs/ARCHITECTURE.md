# MoraWeave architecture

## 1. Evidence before language preference

MoraWeave does not treat a fluent language-model output as ground truth. A candidate remains a hypothesis until independent evidence supports it.

```text
E_acoustic      decoder/CTC sequence evidence
E_mora          mora and phone compatibility
E_lexical       rights-gated public-memory support
E_preservation  filler/restart/learner-error retention
```

Each evidence stream is normalized across the candidate set. Gate weights are adjusted by stream coverage and separation, then anchored by priors that keep acoustic evidence dominant.

## 2. Four-stream gate

```text
w_s ∝ prior_s × (0.2 + reliability_s)
score(h) = Σ_s w_s × normalized(E_s(h)) - grammar_honeytrap(h)
```

Reliability is higher when a stream covers candidates and meaningfully separates them. Missing evidence becomes zero support, not invented support.

The `Grammar Honeytrap` penalty is activated when teacher/language preference exceeds the mean of acoustic and mora support. It targets the common failure mode where a grammatical candidate wins mainly because it is grammatical.

## 3. Uncertainty as a control signal

MoraWeave computes:

- normalized candidate entropy;
- top-two score margin;
- disagreement among evidence-stream winners;
- local word/mora confidence and disagreement.

These signals are used for routing, not merely displayed in a UI.

```text
low uncertainty  → finalize observed candidate
high uncertainty → select local audio spans → higher-beam re-decode
still ambiguous  → optional second ASR / local teacher probabilities
```

A default budget limits total re-listening duration per segment.

## 4. Mora Shadow

Orthographic text and phonological evidence remain parallel:

```text
今日は会社へ行きます
キョ / ウ / ワ / カ / イ / シャ / エ / イ / キ / マ / ス
```

The mora shadow can rescue orthographic alternatives that sound the same and reject natural text that sounds different. It does not replace the Whisper tokenizer, because mora-only output would lose kanji and segmentation information.

## 5. Shared-encoder training

The optional PyTorch model keeps the existing Whisper text decoder and adds:

```text
shared Whisper encoder
  ├─ four-branch gated residual mixer
  ├─ original autoregressive text decoder
  ├─ mora CTC
  ├─ phone CTC
  ├─ mora-boundary classification
  ├─ F0 regression
  ├─ accent classification
  └─ preservation classification
```

The preservation labels distinguish ordinary speech, filler, restart/self-correction, and learner-error evidence. The goal is to prevent every non-canonical event from being learned as disposable noise.

## 6. Public-memory design

The hashed memory stores:

```text
namespace
BLAKE2b digest of n-gram
n-gram length
frequency count
build and input digests
rights asset ID
```

It does not store source sentences. This reduces accidental redistribution but does not erase upstream license or privacy obligations; the rights registry remains authoritative.

## 7. Immutable transcript contract

`ObservedTranscript.evidence_sha256` covers:

- observed text;
- selected candidate ID;
- complete candidate evidence;
- ranking and gates;
- uncertainty spans;
- source-audio digest when available.

Normalization verifies this hash before attaching its derivative. A normalized result also carries the observed hash.

## 8. Failure behavior

MoraWeave fails closed when:

- candidate IDs are duplicated;
- rank-only output is not an exact permutation;
- an unknown data asset is requested;
- rights state is `deny` or `review` for the requested operation;
- CTC targets contain the blank ID;
- a re-listening span is invalid;
- observed evidence hash changes;
- teacher probabilities are outside `[0,1]` or do not sum to one.

## 9. Research questions

The implementation deliberately leaves the following as benchmark questions:

- Does the four-stream gate improve learner-error preservation at equal CER?
- Which uncertainty signal best predicts beneficial re-listening?
- Does hashed public memory improve proper-noun accuracy without increasing unsupported corrections?
- How much can teacher-probability caching reduce latency?
- Do F0/accent heads improve boundary or homophone selection?
- Does the preservation head reduce deletion of fillers and restarts?
