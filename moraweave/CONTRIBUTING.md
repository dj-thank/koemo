# Contributing

Contributions are welcome when they preserve the central invariant:

```text
observedTranscript != normalizedTranscript
```

Before opening a change:

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_rights_registry.py data/rights_registry.json
python -m build --wheel
```

Requirements:

- candidate selection changes need a grammatical-decoy test;
- mora changes need contracted-sound, geminate, nasal, and long-vowel tests;
- data additions need a rights record, exact version, source digest, and attribution;
- speech experiments need speaker-disjoint splits;
- local-LLM changes must remain loopback-only and closed-set by default;
- accuracy claims need named datasets, revisions, preprocessing, metrics, failures, seeds, hardware, and commits;
- raw recordings, transcripts with personal data, model weights, API keys, salts, and upstream speaker IDs must not be committed.

A lower normalized CER is not sufficient when observed learner errors are erased or unsupported corrections increase.
