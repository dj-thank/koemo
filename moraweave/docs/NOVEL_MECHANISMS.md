# MoraWeave original mechanisms

This document describes original MoraWeave proposals. Names here are not claims that no related idea has ever existed; each mechanism must be judged by reproducible ablation and comparison.

## 1. Evidence Weave

Instead of summing fixed ASR and LM scores, MoraWeave maintains four independent streams:

```text
acoustic     What the waveform and decoder support
mora         What Japanese mora/phone timing supports
lexical      What rights-gated public memory supports
preservation What is lost when fillers, restarts, or learner errors are normalized away
```

Gate weights depend on stream coverage and candidate separation. Acoustic evidence retains the largest prior.

Falsifiable hypothesis: dynamic reliability weighting improves learner-error preservation at equal observed CER versus fixed weights.

## 2. Grammar Honeytrap

A language model can strongly prefer a grammatical candidate even when the speaker produced an error. MoraWeave computes a penalty when teacher preference substantially exceeds acoustic and mora support.

```text
penalty ∝ max(0, teacher_preference - mean(acoustic, mora))
```

Counterfactual evaluation inserts a grammatical decoy into every candidate set. Failure means the observed selector chooses the decoy without sufficient acoustic evidence.

## 3. Consensus Spine and Contradiction Islands

N-best candidates are aligned at Unicode-character level.

- **Consensus Spine**: text shared by all candidates; normally does not need more computation.
- **Contradiction Islands**: local insertions, deletions, substitutions, numbers, particles, kanji/proper nouns, or phonetic alternatives.

Only contradiction islands are mapped back to audio and considered for re-listening. This is more precise than re-running a heavy model over an entire meeting or even an entire segment.

Falsifiable hypothesis: contradiction-island routing retains the accuracy gain of full second-pass decoding while reducing re-decoded audio duration.

## 4. Query-Selected Acoustic Memory

A re-listening result is keyed by:

```text
audio SHA-256
start/end milliseconds
adapter and model
beam and hypothesis count
language
prompt digest
hotword digest
```

The cache stores candidate evidence, never waveform bytes. A budget scheduler ranks requests by estimated information gain per second and increases priority for numbers and proper-noun islands.

Falsifiable hypothesis: span cache and scheduling reduce repeated GPU work and improve number/proper-noun accuracy under a fixed latency budget.

## 5. Mora Shadow

Every orthographic candidate may carry a parallel representation:

```text
surface text
reading
mora sequence
phone sequence
mora timing
```

The shadow is not the final transcript. It is evidence used to distinguish:

- same sound, different spelling;
- natural spelling, different sound;
- omitted long vowel, moraic nasal, or geminate;
- contracted sounds such as キャ, ティ, and ファ.

## 6. Error Preservation Head

The optional shared-encoder model includes a framewise preservation head with categories such as:

```text
ordinary speech
filler
restart/self-correction
learner-error evidence
```

The purpose is not to score a person as “wrong” frame by frame. It provides a training signal that non-canonical events must not be deleted merely because a clean transcript has higher language probability.

## 7. Hashed Public Memory

Public text is converted to BLAKE2b n-gram digests and counts after rights approval. Build records include asset ID and source-input digest. Original source sentences are not stored in the SQLite memory.

This design reduces accidental source redistribution, but it does not make restricted or personal data safe to ingest. Rights and privacy gates remain mandatory.

## 8. Teacher Probability Cache

A local teacher returns probabilities over existing candidate IDs, not a new observed sentence. Results are cached by model, context, candidate-set, and audio digests.

Falsifiable hypothesis: probability caching materially reduces latency in repeated domain/context conditions without changing candidate selection.

## 9. Rights Gate as part of model architecture

Data permission is treated as executable state, not README prose:

```text
train
deriveFeatures
redistributeRaw
exportSpeakerId
```

`review` is a hard stop. This allows a model registry and data pipeline to optimize only over assets whose allowed operations are known.

## 10. Two-text truth model

MoraWeave treats both texts as useful but different:

```text
observedTranscript   evidence about the spoken event
normalizedTranscript a readability derivative
```

The observed record is SHA-256 protected. Each normalized segment refers back to its observed evidence hash. Pronunciation, disfluency, learner error, and unsupported-correction metrics use observed evidence.

## Required ablations

A publishable MoraWeave experiment should compare:

1. single-best ASR;
2. N-best acoustic score only;
3. fixed acoustic + LM fusion;
4. four-stream gate without Grammar Honeytrap;
5. four-stream gate with Grammar Honeytrap;
6. full-segment second pass;
7. contradiction-island selective re-listening;
8. selective re-listening plus acoustic cache;
9. optional Qwen3-ASR second ear;
10. auxiliary mora/phone/F0/preservation training heads.

Report observed and normalized metrics separately. A result that improves clean text while erasing spoken errors is not automatically an improvement.
