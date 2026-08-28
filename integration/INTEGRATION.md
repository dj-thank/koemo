# Integration map for `japanese-speaking-assessment-poc`

The modules in this bundle are executable and tested. The existing local PoC
source was not visible in the connected repository or file library, so the
installer copies additive files but does not guess at or rewrite unseen
functions.

## 0. Install the additive modules

From the extracted bundle:

```powershell
.\integration\apply-to-local-poc.ps1
```

Use `-DryRun` to inspect the plan or `-Force` only after reviewing a diff. The
underlying Python installer verifies SHA-256 after every copied file.

The default is non-destructive. Existing files are skipped. Use `--force` only
after reviewing a diff.

## 1. Replace single-best ASR with N-best for short assessment utterances

The primary adapter is `scripts/whisper_nbest.py`. Invoke it from the same Python
environment and model cache currently used by faster-whisper:

```powershell
python scripts\whisper_nbest.py recording.wav `
  --model small `
  --language ja `
  --beam-size 8 `
  --num-hypotheses 5 `
  --output stage-asr-nbest.json
```

Important boundaries:

- one utterance per call;
- normally no more than 30 seconds;
- candidate text has no timestamps;
- retain the existing word-timestamp ASR or forced aligner as a separate stage.

A Node child-process wrapper should reject non-zero exit status and parse only
JSON written to the chosen output file. Do not parse logs from stdout.

The resulting `candidates` already contain `whisperScore`, so they can be passed
to `rankObservedCandidates`.

Persist both waveform hashes from this stage. `audioPcmSha256` identifies the
decoded source waveform before VAD; `modelInputPcmSha256` identifies the exact
post-VAD waveform passed to Whisper. Both use the declared
`sha256-float32le-mono-waveform` representation.

## 2. Convert character alignment to canonical mora units

In `scripts/gop_align.py`, keep the existing character-CTC aligner, then add:

```python
from scripts.japanese_mora import merge_character_alignment

character_units = alignment_result["characters"]

mora_units = merge_character_alignment(
    character_units,
    input_time_unit="auto",
    source="char-merge",
)

alignment_result["moraUnits"] = mora_units
alignment_result["moraCount"] = len(mora_units)
```

Do not delete the character rows until learner-audio migration has been
validated. Store both representations during the transition.

## 3. Create the immutable observed transcript in `server.mjs`

Use the pipeline facade:

```js
import { createObservedRecordFromNBest } from "./src/transcript-pipeline.mjs";

const { record: observedRecord, observedCandidate, observedRanking } =
  createObservedRecordFromNBest(asrNbestResult, {
    id: attemptId,
    moraUnits: alignmentResult.moraUnits ?? [],
    uncertaintySpans,
    metadata: {
      asrModel,
      taskType,
    },
  });
```

Persist `observedRecord` immediately as its own stage artifact. The hash covers
the observed text, candidate set, mora units, and uncertainty spans.

## 4. Run the local LLM only after observed persistence

```js
import { normalizeRecordWithOllama } from "./src/transcript-pipeline.mjs";

const normalization = await normalizeRecordWithOllama(observedRecord, {
  model: process.env.LOCAL_LM_MODEL,
  endpoint: process.env.OLLAMA_ENDPOINT,
  context: previousConversationText,
  question: promptText,
  domainTerms: expectedDomainTerms,
});

const finalTranscriptRecord = normalization.record;
```

The default endpoint policy allows only loopback. The Ollama adapter returns
candidate IDs, not text, and verifies the ID permutation. Persist the local-LM
decision and normalized ranking as derivative evidence.

For an initial experiment where local rank should influence normalization more
strongly, weights can be explicit:

```js
const normalization = await normalizeRecordWithOllama(observedRecord, {
  model: process.env.LOCAL_LM_MODEL,
  normalizedWeights: {
    observedScore: 0.40,
    localLmScore: 0.45,
    editSimilarity: 0.15,
  },
});
```

Do not use these example weights as production defaults without calibration.

## 5. Update `src/fluency.mjs`

Prefer canonical timed units when present:

```js
import { computeFluencyFromMoraUnits } from "./fluency-from-mora.mjs";

export function computeFluency(input) {
  if (Array.isArray(input.moraUnits) && input.moraUnits.length > 0) {
    return {
      ...existingNonMoraMetrics(input),
      ...computeFluencyFromMoraUnits(input.moraUnits, {
        totalDurationMs: input.durationMs,
        pauseThresholdMs: 250,
      }),
    };
  }

  // Compatibility lane for historical records only.
  return existingFluencyImplementation(input);
}
```

New recordings should eventually fail closed when expected `moraUnits` are
missing. The fallback is for historical records.

## 6. Update `src/scoring.mjs`

```js
const observedText = transcriptRecord.observedTranscript.text;
const normalizedText = transcriptRecord.normalizedTranscript?.text ?? observedText;

// Pronunciation, omissions, repetitions, fluency, and learner grammar evidence.
scoreObservedEvidence(observedText, transcriptRecord.moraUnits);

// Meaning/rubric support may inspect both forms but must keep the observed hash.
scoreSemanticEvidence({
  observedText,
  normalizedText,
  observedSha256: transcriptRecord.observedTranscript.sha256,
});
```

Never use only `normalizedText` to grade learner grammar or pronunciation.

## 7. Train the mora-aware Whisper model separately from production inference

Build a fixed vocabulary from training readings:

```python
from training.mora_vocab import MoraVocabulary

vocab = MoraVocabulary.build(reference_kana_values, min_frequency=1)
vocab.save("artifacts/mora_vocab.json")
```

Create the model:

```python
from training.whisper_mora_multitask import (
    WhisperMoraMultiTaskConfig,
    WhisperMoraMultiTaskModel,
)

model = WhisperMoraMultiTaskModel.from_pretrained(
    "openai/whisper-small",
    multitask_config=WhisperMoraMultiTaskConfig(
        mora_vocab_size=len(vocab),
        mora_blank_id=vocab.blank_id,
        phone_vocab_size=None,
    ),
)
```

Each batch may contain:

```text
input_features          required Whisper log-Mel tensor
text_labels             normal Whisper decoder labels
mora_labels             padded mora IDs; -100 outside valid target
mora_label_lengths      optional; inferred from -100 padding
encoder_lengths         preferred post-convolution frame lengths
boundary_labels         optional [batch, encoder_time], -100 ignored
phone_labels            optional when phone head enabled
```

The model rejects non-`-100` CTC padding, blank IDs inside CTC targets, and
non-`-100` boundary labels beyond the declared encoder length. This is
intentional: silently training on padded frames corrupts mora timing.

Start by freezing the text decoder, train the auxiliary heads and upper encoder,
then compare against a fully frozen encoder baseline. Do not replace the current
production ASR until the learner-audio benchmark improves.

## 8. Persistence migration

Historical records without this contract should be wrapped as explicitly
versioned legacy records. Do not silently reinterpret old transcript text as
both observed and normalized evidence.

## 9. Environment flags

```dotenv
ASR_ENGINE=faster-whisper-nbest
ASR_MODEL=small
ASR_BEAM_SIZE=8
ASR_NBEST=5
ASR_NBEST_MAX_UTTERANCE_SECONDS=30

MORA_ENGINE=char-merge
MORA_CTC_ENABLED=false

LOCAL_LM_ENABLED=true
LOCAL_LM_MODE=rank-only
LOCAL_LM_MODEL=your-local-model
OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/chat
LOCAL_LM_ALLOW_FREE_REWRITE=false
LOCAL_LM_ALLOW_REMOTE_ENDPOINT=false
LOCAL_LM_ALLOW_CLOUD_MODEL=false

TRANSCRIPT_OBSERVED_IMMUTABLE=true
```

Turn `MORA_CTC_ENABLED=true` only after a trained auxiliary head is loaded and
its vocabulary hash matches the runtime vocabulary.

## 10. Feed a trained mora head back into N-best fusion

For each ASR candidate, convert its reading to mora IDs with the same saved
`MoraVocabulary`, then score all closed-set targets against one utterance's
`mora_logits`:

```python
from training.mora_ctc_runtime import score_mora_candidates

scores = score_mora_candidates(
    mora_logits,
    {candidate_id: mora_ids for candidate_id, mora_ids in candidate_targets},
    encoder_length=valid_encoder_frames,
    blank_id=vocab.blank_id,
)
```

Attach each finite `moraCtcScore` to its unchanged ASR candidate before calling
`createObservedRecordFromNBest`. The Node side should use the exact-set adapter:

```js
import { attachMoraCtcScores } from "./src/mora-ctc-fusion.mjs";

asrNbestResult.candidates = attachMoraCtcScores(
  asrNbestResult.candidates,
  pythonCtcScoreRows,
);
```

For approximate mora timing, call
`greedy_mora_spans` and `mora_spans_to_units`; keep forced-alignment output as a
separate comparison lane until boundary evaluation is complete.
