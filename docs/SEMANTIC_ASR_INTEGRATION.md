# Optional Semantic ASR authoritative final pass

Koemo keeps its current default behavior. Semantic ASR is an opt-in final-transcription engine for
saved WAV files; Windows Speech and rolling Whisper remain live-preview engines.

## Why this is separate

Koemo and Semantic ASR have different responsibilities:

- Koemo records mic/system audio, performs AEC, manages channels, displays live text, summarizes,
  exports, and provides the desktop UI.
- Semantic ASR owns N-best evidence, mora/semantic ranking, calibration, selective verification,
  accepted/provisional decisions, and observed/normalized separation.

Koemo must consume a pinned Semantic ASR revision rather than carrying another MoraWeave copy.

## Installation for development

From sibling checkouts:

```bat
cd C:\work\semantic-asr
python -m pip install -e ".[asr]"

cd C:\work\koemo
python -m pytest -q tests\test_semantic_asr_bridge.py ^
  tests\test_authoritative_transcription.py ^
  tests\test_semantic_asr_settings.py
```

For a release build, pin an immutable Semantic ASR commit or release and record it in the Koemo
release manifest. Do not depend on a moving branch.

## Settings

The adapter reads these optional keys from Koemo's existing configuration dictionary:

```json
{
  "semantic_asr_enabled": false,
  "semantic_asr_fallback_to_legacy": true,
  "semantic_asr_effort": "cpu-quality",
  "semantic_asr_model": "large-v3-turbo",
  "semantic_asr_device": "auto",
  "semantic_asr_compute_type": "default",
  "semantic_asr_maximum_hypotheses": 12,
  "semantic_asr_evidence_budget_ms": 4000,
  "semantic_asr_maximum_evidence_actions": 4,
  "semantic_asr_window_ms": 28000,
  "semantic_asr_overlap_ms": 1200,
  "semantic_asr_keep_warm": false,
  "semantic_asr_legacy_correction_normalized_only": true
}
```

Default remains disabled. Invalid values fail explicitly rather than silently changing the final
transcript.

## Product integration point

Construct the service once near Koemo's existing final `Transcriber`:

```python
from koemo.authoritative_transcription import AuthoritativeTranscriptionService
from koemo.semantic_asr_bridge import SemanticASRBridge
from koemo.semantic_asr_settings import semantic_asr_settings

settings = semantic_asr_settings(cfg)
service = AuthoritativeTranscriptionService(
    legacy_transcriber,
    policy=settings.policy,
    semantic_bridge=SemanticASRBridge(settings.bridge),
)
```

After Koemo has saved the accepted mic/system WAV, call:

```python
result = service.transcribe_saved_audio(
    wav_path,
    legacy_audio=channel_audio,
    language=cfg.get("language", "ja"),
    context=meeting_title,
    hotwords=tuple(domain_terms),
    on_progress=on_progress,
)
```

Use:

```text
result.observed_text     canonical evidence-preserving text
result.normalized_text   readable derivative
result.decision          accepted / provisional / legacy-unfused
result.engine            semantic-asr / legacy-faster-whisper
result.evidence_sha256   immutable evidence or legacy provenance digest
result.segments          timestamped observed and normalized segments
result.fallback_reason   explicit Semantic ASR failure when fallback occurred
```

Do not pass live-preview text into this service as evidence.

## Regex correction boundary

Koemo's current `native_correction.py` runs inside the legacy transcriber. During migration, legacy
results remain labelled `legacy-unfused`. On the Semantic ASR path, the observed text is not passed
through Koemo regex correction. Any product-specific correction belongs only to a normalized
derivative with a ruleset digest.

## Dual-channel use

Call the service independently for each accepted channel. Preserve:

- original WAV SHA;
- AEC derivative SHA and parameters;
- channel activity decision;
- per-channel Semantic ASR evidence SHA;
- final timestamp/channel merge provenance.

Do not concatenate channel hypotheses into one artificial N-best set.

## Fallback behavior

| Condition | Behavior |
|---|---|
| Semantic ASR disabled | existing faster-whisper path |
| optional dependency missing | explicit error, then legacy only if policy allows |
| model load or OOM failure | explicit fallback reason |
| no legacy audio buffer | fail rather than invent text |
| uncalibrated ranker | reorder-only inside Semantic ASR |
| cache mismatch | recompute; never trust incompatible cache data |

## Validation before default enablement

1. Run legacy and Semantic ASR on the same saved meeting WAVs.
2. Keep speaker/source-disjoint calibration and test sets.
3. Compare CER, numbers, dates, currency, negation, entities, fillers, repairs, unsupported
   insertion, calibration, RTF, RAM, and VRAM.
4. Pass the Semantic ASR deployment gate on every supported Koemo hardware tier.
5. Enable behind a settings flag for beta users.
6. Make it default only after the declared fixed-risk target is met.

The bridge code proves integration contracts. It does not by itself prove real-meeting accuracy
improvement.
