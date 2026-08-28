# Calibration and selective-risk protocol

## Dataset split

Use three disjoint sets:

```text
training       model and auxiliary-head optimisation
calibration    temperature/profile fitting and threshold selection
test           final locked evaluation only
```

Speakers, source recordings and near-duplicate utterances must not cross these boundaries.

## Calibration records

The minimal JSONL accepted by `moraweave-calibrate` is:

```json
{"confidence": 0.73, "correct": true}
```

For full experiments, retain additional immutable fields:

```json
{
  "utteranceId": "...",
  "audioSha256": "...",
  "candidateId": "...",
  "confidence": 0.73,
  "correct": true,
  "domain": "meeting",
  "microphone": "...",
  "speakerSplit": "calibration",
  "modelCommit": "...",
  "calibrationDigest": "..."
}
```

## Metrics

Report before and after calibration:

- Expected Calibration Error with declared bins;
- Brier score;
- binary negative log likelihood;
- risk-coverage curve and AURC;
- coverage at fixed risk targets;
- error rate by confidence decile;
- domain, microphone, speaker and utterance-length slices.

ECE alone is insufficient because it depends on binning. Brier, NLL and risk-coverage are mandatory companions.

## Threshold selection

Acceptance, re-listening and second-ear thresholds are fitted only on the calibration set. Do not tune them on the final test set.

Operational decisions:

```text
accepted     posterior and evidence coverage satisfy the calibrated policy
provisional  uncertainty is retained and a review/re-listening action is exposed
```

The system must remain useful when a requested evidence source is unavailable. Missing evidence increases selective risk and may reduce coverage; it must not be silently replaced with a neutral-looking fabricated score.

## Distribution shift

A profile is valid only for its declared model, decoding settings and evaluation domain. Store:

```text
model identifier and revision
runtime/package versions
beam/hypothesis count
language and prompt policy
training/calibration dataset revisions
profile digest
```

Recalibrate after a decoder, model, language policy, prompt, public lexicon or fusion change.

## Paired evaluation

Compare systems on the same utterances. Use paired bootstrap confidence intervals for CER/kana-CER/MLER and paired risk-coverage comparisons. Publish negative results and the percentage of utterances for which no additional evidence was available.

## CLI

```bash
moraweave-calibrate heldout.jsonl --output calibration/profile.json
```

The command refuses fewer than ten records; real releases should use substantially larger, speaker-diverse sets.
