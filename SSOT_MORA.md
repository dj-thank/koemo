# Mora-aware Japanese ASR SSOT

The canonical full SSOT for release 0.3.0 is packaged in
`japanese-speaking-assessment-mora-core-v0.3.0-public.zip` at the repository
root. The source archive is checksum-pinned by `SHA256SUMS.txt`.

Central invariant:

```text
observedTranscript != normalizedTranscript
```

The immutable observed transcript, ASR candidates, mora units, and uncertainty
spans are selected from acoustic evidence before any local-language-model rank
is consulted. See `PROJECT_README.md`, `VALIDATION.md`, and the complete
`SSOT_MORA.md` inside the source archive for the runtime, training, persistence,
and evaluation contracts.
