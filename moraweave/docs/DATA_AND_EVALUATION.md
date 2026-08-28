# Data and evaluation protocol

## Data governance

Every source must appear in `data/rights_registry.json`. The four permissions are intentionally separate:

| Permission | Meaning |
|---|---|
| `train` | Model parameters may be updated from the asset |
| `deriveFeatures` | Hashed/aggregated features may be created |
| `redistributeRaw` | Raw source files may be included in a release |
| `exportSpeakerId` | Original speaker identity fields may leave the local workspace |

`review` is a hard stop, not a weak form of permission.

Common Voice processing uses local file paths and HMAC speaker pseudonyms. It does not copy audio into the repository. A secret salt is required and must not be committed.

JMdict processing streams XML and stores hashed n-gram/readings statistics. Attribution remains required even when raw entries are not redistributed.

Aozora Bunko requires a per-work manifest. Works with unknown or non-allowed rights states are rejected.

## Split integrity

Speech evaluation must be speaker-disjoint. Recommended split key:

```text
HMAC(secret_salt, upstream_speaker_id)
```

The split manifest should store only pseudonyms, dataset version, source-row digest, and split assignment. Do not publish the salt.

## Required metrics

### Orthographic

- CER
- number/date/money error rate
- proper-noun exact match

### Phonological

- Kana-CER
- mora label error rate (MLER)
- phone error rate
- mora-boundary MAE and F1
- accent/F0 auxiliary performance

### Preservation

- filler preservation
- restart/self-correction preservation
- learner-error preservation
- unsupported correction rate

### Systems

- real-time factor
- peak RAM and VRAM
- re-listening duration ratio
- second-model invocation ratio
- teacher-cache hit rate

## Null semantics

A metric whose reference is unavailable is `null`, not zero. Examples:

- kanji text without a trusted reading → Kana-CER and MLER are `null`;
- no reference numbers but hypothesis inserts a number → number metric is reported as undefined/insertion-sensitive, not perfect;
- no speaker identity → speaker metrics are `null`.

## Counterfactual tests

Each benchmark should include a grammatical decoy candidate that is acoustically wrong. Example:

```text
spoken:  昨日学校を行きました
clean:   昨日学校に行きました
wrong:   昨日会社に行きました
```

A useful observed selector must preserve `spoken` when acoustic and mora evidence support it, even if the local teacher strongly prefers `clean`.

## Acceptance gates

A new model or fusion configuration is not accepted solely for lower normalized CER. It must satisfy:

```text
observed CER does not regress beyond tolerance
learner-error preservation improves or remains stable
unsupported correction does not increase
number/proper-noun performance does not regress
RTF and memory remain within target profile
rights and split manifests validate
```

All aggregate reports must include failure cases and abstentions. Failed jobs cannot disappear from denominators.
