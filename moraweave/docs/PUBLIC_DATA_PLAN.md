# Rights-gated public data plan

MoraWeave uses public data only through an explicit release-level rights record. “Publicly downloadable” is not treated as permission for training, derived-feature publication or raw redistribution.

## Executable states

Each asset/version declares independently:

```text
train              allow / deny / review
derive_features    allow / deny / review
redistribute_raw   allow / deny / review
export_speaker_id  allow / deny / review
```

`review` blocks the operation. It is not interpreted as probable permission.

## Candidate sources

### Common Voice

Use a named release and locale manifest. Speaker/client identifiers are pseudonymized with a secret HMAC key before local indexing. The original identifier is never exported. Record release-specific licence and dataset-card revision.

### ReazonSpeech

Use only an exact named release after confirming the licence and any source-program restrictions for that asset. Preserve provenance from source audio to derived sample. Do not infer that every repository component has one identical redistribution permission.

### SaSLaW

Useful for learner-speech evaluation, but exact download terms and speaker privacy conditions must be captured before training or redistribution. Keep learner evaluation splits speaker-disjoint.

### JMdict

Use readings and lexical metadata under the declared EDRDG terms and preserve attribution. The default lexical memory stores keyed/hash features and aggregate counts, not the original dictionary XML or reconstructable source strings.

### 青空文庫

Rights vary by work. Each work needs a provenance and public-domain/licence record. Never treat the whole collection as one unrestricted corpus.

### Project-generated speech

Collect explicit consent specifying research, model training, derived-feature publication, raw redistribution and withdrawal policy separately. Store consent version and recording lineage.

## Privacy controls

- raw audio stays outside Git;
- source paths are redacted from exported transcripts by default;
- speaker identifiers are HMAC-pseudonymized;
- caches contain evidence JSON, not waveforms;
- deletion requests can be applied by asset/speaker lineage;
- dataset manifests include SHA-256 and split assignment;
- near-duplicate and same-speaker leakage checks run before evaluation.

## Minimum manifest

```json
{
  "assetId": "common-voice-ja-<release>",
  "source": "...",
  "version": "...",
  "license": "...",
  "licenseUrl": "...",
  "train": "review",
  "deriveFeatures": "review",
  "redistributeRaw": "deny",
  "exportSpeakerId": "deny",
  "attribution": "...",
  "reviewedAt": "2026-08-29",
  "notes": "Exact release review required"
}
```

## Evaluation integrity

Report public and non-public subsets separately. A result on a private or rights-restricted set must not be presented as independently reproducible. Publish the evaluation code, configuration, hashes and aggregate metrics even when audio cannot be redistributed.
