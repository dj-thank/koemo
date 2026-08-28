#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "moraweave"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


CONTRACT_EXPORT = '''from .frontier import (
    CandidateEvidence,
    EvidenceName,
    GateDecision,
    MoraUnit,
    NormalizedTranscript,
    ObservedTranscript,
    RankedCandidate,
    canonical_json,
    sha256_json,
)

__all__ = [
    "CandidateEvidence",
    "EvidenceName",
    "GateDecision",
    "MoraUnit",
    "NormalizedTranscript",
    "ObservedTranscript",
    "RankedCandidate",
    "canonical_json",
    "sha256_json",
]
'''

LONGFORM_SHIM = '''from .longform_frontier import (
    FrontierLongformTranscriber,
    LongformResult,
    LongformSegment,
    RelistenSpan,
    Window,
    merge_candidates,
    plan_windows,
    probe_duration_ms,
    sha256_file,
    stitch_text,
)

__all__ = [
    "FrontierLongformTranscriber",
    "LongformResult",
    "LongformSegment",
    "RelistenSpan",
    "Window",
    "merge_candidates",
    "plan_windows",
    "probe_duration_ms",
    "sha256_file",
    "stitch_text",
]
'''

RELEASE_TEST = '''from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_release_tree_contains_required_public_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "data/rights_registry.json",
        "scripts/validate_rights_registry.py",
        "src/moraweave/__init__.py",
        "src/moraweave/contracts/frontier.py",
        "src/moraweave/runtime_cache.py",
        "src/moraweave/longform_frontier.py",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"missing public release files: {missing}"


def test_pyproject_and_package_version_are_well_formed() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "moraweave"
    assert project["project"]["version"] == "0.2.0"
    scripts = project["project"]["scripts"]
    assert scripts["moraweave-transcribe"] == "moraweave.transcribe_v2:main"
    assert scripts["moraweave-calibrate"] == "moraweave.calibrate_cli:main"


def test_release_does_not_contain_raw_audio_or_weights() -> None:
    forbidden = {".wav", ".mp3", ".m4a", ".flac", ".pt", ".pth", ".safetensors"}
    committed = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden
    ]
    assert committed == []
'''

README_BANNER = '''# MoraWeave v0.2.0 frontier update

MoraWeave v0.2 adds calibrated selective risk, a dual surface/mora lattice, cached ambiguity-only re-listening, official Qwen3-ASR/Forced-Aligner contracts, and a delayed local Qwen3.8 candidate teacher.

```text
observedTranscript != normalizedTranscript
```

```bash
python -m pip install -e '.[asr,dev]'
moraweave-transcribe audio.m4a --output-dir transcripts
moraweave-calibrate heldout.jsonl --output calibration/profile.json
```

Research boundaries and protocols:

- [`docs/RESEARCH_2026-08-29.md`](docs/RESEARCH_2026-08-29.md)
- [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)
- [`docs/CALIBRATION_PROTOCOL.md`](docs/CALIBRATION_PROTOCOL.md)
- [`docs/QWEN_RUNTIME.md`](docs/QWEN_RUNTIME.md)
- [`docs/PUBLIC_DATA_PLAN.md`](docs/PUBLIC_DATA_PLAN.md)

---
'''

CHANGELOG_BANNER = '''## 0.2.0 — 2026-08-29

- Added held-out/robust calibration, posterior risk and provisional abstention.
- Added dual surface/mora contradiction islands and information-gain scheduling.
- Added official Qwen3-ASR/Forced-Aligner and delayed Qwen3.8 teacher contracts.
- Added runtime evidence cache v2 with context/hotword/calibration provenance.
- Added cached long-form re-listening and optional cached Qwen second-ear evidence.
- Added critical-entity, punctuation, calibration and risk-coverage evaluation.
- Added pinned research and rights-gated public-data protocols.
'''


def ensure_frontier_contracts() -> None:
    write(PROJECT_ROOT / "src/moraweave/contracts/__init__.py", CONTRACT_EXPORT)
    (PROJECT_ROOT / "src/moraweave/contracts.py").unlink(missing_ok=True)
    write(PROJECT_ROOT / "src/moraweave/longform_v2.py", LONGFORM_SHIM)
    write(PROJECT_ROOT / "tests/acoustic_memory/test_release.py", RELEASE_TEST)


def patch_adapters() -> None:
    path = PROJECT_ROOT / "src/moraweave/adapters.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace('    language: str = "ja"\n', '    language: str | None = "ja"\n', 1)

    if 'if value.lower() == "auto":' not in text:
        anchor = '    value = str(language).strip()\n    return _QWEN_LANGUAGE.get(value.lower(), value)\n'
        if anchor not in text:
            raise RuntimeError("Qwen language mapping anchor missing")
        text = text.replace(
            anchor,
            '    value = str(language).strip()\n'
            '    if value.lower() == "auto":\n'
            '        return None\n'
            '    return _QWEN_LANGUAGE.get(value.lower(), value)\n',
            1,
        )

    if 'language_policy = "forced"' not in text:
        anchor = '        tokenizer = Tokenizer(\n'
        if anchor not in text:
            raise RuntimeError("faster-whisper tokenizer anchor missing")
        detector = '''        detected_language = request.language
        language_probability = None
        language_policy = "forced"
        if request.language in {None, "", "auto"}:
            language_policy = "auto"
            detect_language = getattr(self.model, "detect_language", None)
            if detect_language is None:
                raise RuntimeError("installed faster-whisper has no public detect_language API")
            detected = detect_language(waveform)
            if isinstance(detected, str):
                detected_language = detected
            elif isinstance(detected, tuple) and detected:
                detected_language = detected[0]
                if len(detected) > 1:
                    try:
                        language_probability = float(detected[1])
                    except (TypeError, ValueError):
                        language_probability = None
            else:
                detected_language = getattr(detected, "language", None)
                probability = getattr(detected, "language_probability", None)
                try:
                    language_probability = None if probability is None else float(probability)
                except (TypeError, ValueError):
                    language_probability = None
            if not detected_language:
                raise RuntimeError("faster-whisper language detection returned no language")

'''
        text = text.replace(anchor, detector + anchor, 1)
        text = text.replace('            language=request.language,\n', '            language=detected_language,\n', 1)

    if 'except TypeError:\n            try:\n                prompt = self.model.get_prompt' not in text:
        pattern = re.compile(
            r'        prompt = self\.model\.get_prompt\(\n'
            r'            tokenizer,\n'
            r'            previous_tokens=initial_tokens,\n'
            r'            without_timestamps=True,\n'
            r'            hotwords=hotwords,\n'
            r'        \)\n'
        )
        replacement = '''        try:
            prompt = self.model.get_prompt(
                tokenizer,
                previous_tokens=initial_tokens,
                without_timestamps=True,
                hotwords=hotwords,
            )
        except TypeError:
            try:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True, None, hotwords)
            except TypeError:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True)
'''
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError("faster-whisper get_prompt anchor missing")

    if '"languagePolicy": language_policy' not in text:
        anchor = '                        "durationSeconds": duration_seconds,\n'
        if anchor not in text:
            raise RuntimeError("adapter metadata anchor missing")
        text = text.replace(
            anchor,
            anchor
            + '                        "language": detected_language,\n'
            + '                        "languageProbability": language_probability,\n'
            + '                        "languagePolicy": language_policy,\n',
            1,
        )
    write(path, text)


def patch_pyproject() -> None:
    path = PROJECT_ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(r'(?m)^version\s*=\s*"[^"]+"', 'version = "0.2.0"', text, count=1)
    if count != 1:
        raise RuntimeError("project version anchor missing")
    if '[project.scripts]' not in text:
        text = text.rstrip() + '\n\n[project.scripts]\n'
    for name, target in {
        "moraweave-transcribe": "moraweave.transcribe_v2:main",
        "moraweave-calibrate": "moraweave.calibrate_cli:main",
    }.items():
        if re.search(rf'(?m)^{re.escape(name)}\s*=', text):
            continue
        match = re.search(r'(?m)^\[project\.scripts\]\s*$', text)
        assert match is not None
        text = text[: match.end()] + f'\n{name} = "{target}"' + text[match.end() :]
    write(path, text)


def patch_version_and_docs() -> None:
    path = PROJECT_ROOT / "src/moraweave/__init__.py"
    text = path.read_text(encoding="utf-8")
    if re.search(r'(?m)^__version__\s*=', text):
        text = re.sub(r'(?m)^__version__\s*=\s*"[^"]+"', '__version__ = "0.2.0"', text, count=1)
    else:
        text = text.rstrip() + '\n\n__version__ = "0.2.0"\n'
    write(path, text)

    readme = PROJECT_ROOT / "README.md"
    existing = readme.read_text(encoding="utf-8")
    if "# MoraWeave v0.2.0 frontier update" not in existing:
        write(readme, README_BANNER + '\n' + existing.lstrip())

    changelog = PROJECT_ROOT / "CHANGELOG.md"
    existing = changelog.read_text(encoding="utf-8") if changelog.exists() else "# Changelog\n"
    if "## 0.2.0 — 2026-08-29" not in existing:
        if existing.startswith("# Changelog"):
            rest = existing[len("# Changelog") :].lstrip()
            existing = "# Changelog\n\n" + CHANGELOG_BANNER + "\n" + rest
        else:
            existing = CHANGELOG_BANNER + "\n" + existing
        write(changelog, existing)


def patch_calibration_cli() -> None:
    path = PROJECT_ROOT / "src/moraweave/calibrate_cli.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace('from .calibration import (', 'from .calibration_runtime import (', 1)
    write(path, text)


def main() -> int:
    ensure_frontier_contracts()
    patch_adapters()
    patch_pyproject()
    patch_version_and_docs()
    patch_calibration_cli()
    for workflow in (
        "moraweave-apply-frontier-v2.yml",
        "moraweave-apply-frontier-v2b.yml",
    ):
        (REPO_ROOT / ".github/workflows" / workflow).unlink(missing_ok=True)
    print("MoraWeave frontier v2 integration repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
