# Security and privacy policy

## Report privately

Use a GitHub private security advisory when available. Do not attach real recordings, API keys, access tokens, original Common Voice client IDs, learner identities, or proprietary transcripts.

Include the affected commit, minimal synthetic reproduction, realistic impact, and mitigation proposal.

## Security boundaries

MoraWeave treats the following as sensitive:

- raw audio and video;
- complete transcripts;
- original dataset speaker IDs;
- HMAC speaker salts;
- local model endpoints and credentials;
- model weights with restricted terms;
- public-data files whose redistribution is not permitted.

## Defaults

- No raw audio or model weights are committed.
- Rights state `review` and `deny` block execution.
- Original speaker IDs are never written by the Common Voice builder.
- Observed transcript evidence is SHA-256 protected.
- Local LLM integration must be loopback-only and closed-set/rank-only by default.
- Candidate IDs must be exact and unique.
- Missing evidence never becomes fabricated confidence.

## Threats considered

### Language-model overwrite

A fluent LLM may erase a real learner error or insert a plausible word. MoraWeave separates observed and normalized records and applies Grammar Honeytrap penalties.

### Data exfiltration

Proxy variables, redirects, remote model names, and telemetry can move transcripts outside the local machine. Adapters must explicitly reject non-loopback local-LLM endpoints and document every network call.

### Corpus leakage

Hashed n-gram memory reduces accidental source-text redistribution but is not a guarantee against all reconstruction attacks. Do not ingest sensitive text merely because it will be hashed.

### Speaker re-identification

Common Voice and learner corpora may contain stable speaker fields. MoraWeave HMAC-pseudonymizes locally, forbids salt publication, and requires speaker-disjoint split reports.

### Dependency/model supply chain

Pin model identifiers, revisions, checksums when available, and license evidence. Do not execute remote model code unless explicitly reviewed. Use `trust_remote_code=False` where supported.

## Unsupported claims

Passing unit tests does not prove recognition accuracy, privacy compliance, or resistance to model inversion. Those require separate evaluation and legal/security review.
