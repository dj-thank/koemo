#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "moraweave"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"migration anchor missing: {label}")
    return text.replace(old, new, 1)


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

This branch implements calibrated evidence fusion, a dual surface/mora shadow lattice, selective evidence acquisition, official Qwen3-ASR/Forced-Aligner contracts, a delayed loopback-only Qwen3.8 teacher, and a versioned long-form runtime cache.

Core safety invariant:

```text
observedTranscript != normalizedTranscript
```

New complete-transcription command:

```bash
python -m pip install -e '.[asr,dev]'
moraweave-transcribe audio.m4a --language ja --output-dir transcripts
```

Ambiguity-only Qwen second ear:

```bash
python -m pip install -e '.[asr,qwen,dev]'
moraweave-transcribe audio.m4a --qwen-second-ear --output-dir transcripts
```

Held-out confidence calibration:

```bash
moraweave-calibrate calibration.jsonl --output calibration/profile.json
```

Research and claim boundaries:

- [`docs/RESEARCH_2026-08-29.md`](docs/RESEARCH_2026-08-29.md)
- [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)
- [`docs/CALIBRATION_PROTOCOL.md`](docs/CALIBRATION_PROTOCOL.md)
- [`docs/QWEN_RUNTIME.md`](docs/QWEN_RUNTIME.md)
- [`docs/PUBLIC_DATA_PLAN.md`](docs/PUBLIC_DATA_PLAN.md)

---
'''

CHANGELOG_BANNER = '''## 0.2.0 — 2026-08-29

- Replaced fragile candidate-set min-max scaling with persisted/robust calibration.
- Added posterior, uncertainty decomposition, evidence coverage, selective risk and provisional abstention.
- Added a dual surface/mora shadow lattice and critical contradiction islands.
- Added expected-information-gain-per-cost evidence scheduling.
- Updated Qwen3-ASR language/span/timestamp contracts and added forced alignment.
- Added a delayed, structured, loopback-only Qwen3.8 teacher with abstention.
- Added runtime evidence cache v2, including context/hotword/calibration keys and preserved teacher abstention.
- Added cached long-form Whisper re-listening and optional cached Qwen second-ear inference.
- Added ECE, Brier, NLL, AURC, punctuation and critical-entity evaluation.
- Corrected the Japanese CER fixture denominator from 1/6 to 1/7.
- Added pinned research, calibration, Qwen runtime and public-data rights documentation.
'''


def patch_adapters() -> None:
    path = PROJECT_ROOT / "src/moraweave/adapters.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '    language: str = "ja"\n',
        '    language: str | None = "ja"\n',
        label="DecodeRequest optional language",
    )
    text = replace_once(
        text,
        '    value = str(language).strip()\n    return _QWEN_LANGUAGE.get(value.lower(), value)\n',
        '    value = str(language).strip()\n    if value.lower() == "auto":\n        return None\n    return _QWEN_LANGUAGE.get(value.lower(), value)\n',
        label="Qwen auto language",
    )
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
    text = replace_once(
        text,
        '        tokenizer = Tokenizer(\n',
        detector + '        tokenizer = Tokenizer(\n',
        label="faster-whisper language detection insertion",
    )
    text = replace_once(
        text,
        '            language=request.language,\n',
        '            language=detected_language,\n',
        label="detected tokenizer language",
    )
    old_prompt = '''        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=initial_tokens,
            without_timestamps=True,
            hotwords=hotwords,
        )
'''
    new_prompt = '''        try:
            prompt = self.model.get_prompt(
                tokenizer,
                previous_tokens=initial_tokens,
                without_timestamps=True,
                hotwords=hotwords,
            )
        except TypeError:
            try:
                prompt = self.model.get_prompt(
                    tokenizer, initial_tokens, True, None, hotwords
                )
            except TypeError:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True)
'''
    text = replace_once(text, old_prompt, new_prompt, label="get_prompt compatibility")
    text = replace_once(
        text,
        '                        "durationSeconds": duration_seconds,\n',
        '                        "durationSeconds": duration_seconds,\n'
        '                        "language": detected_language,\n'
        '                        "languageProbability": language_probability,\n'
        '                        "languagePolicy": language_policy,\n',
        label="language provenance",
    )
    write(path, text)


def patch_pyproject() -> None:
    path = PROJECT_ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(r'(?m)^version\s*=\s*"[^"]+"', 'version = "0.2.0"', text, count=1)
    if count != 1:
        raise RuntimeError("project version anchor missing")
    scripts = {
        "moraweave-transcribe": "moraweave.transcribe_v2:main",
        "moraweave-calibrate": "moraweave.calibrate_cli:main",
    }
    section_match = re.search(r'(?m)^\[project\.scripts\]\s*$', text)
    if section_match is None:
        text = text.rstrip() + "\n\n[project.scripts]\n"
        section_start = len(text)
    for name, target in scripts.items():
        if re.search(rf'(?m)^{re.escape(name)}\s*=', text):
            continue
        match = re.search(r'(?m)^\[project\.scripts\]\s*$', text)
        assert match is not None
        insertion = match.end()
        text = text[:insertion] + f'\n{name} = "{target}"' + text[insertion:]
    write(path, text)


def patch_version() -> None:
    path = PROJECT_ROOT / "src/moraweave/__init__.py"
    text = path.read_text(encoding="utf-8")
    if re.search(r'(?m)^__version__\s*=', text):
        text = re.sub(r'(?m)^__version__\s*=\s*"[^"]+"', '__version__ = "0.2.0"', text, count=1)
    else:
        text = text.rstrip() + '\n\n__version__ = "0.2.0"\n'
    write(path, text)


def prepend_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        write(path, block.rstrip() + "\n\n" + text.lstrip())


def main() -> int:
    write(PROJECT_ROOT / "src/moraweave/contracts/__init__.py", CONTRACT_EXPORT)
    (PROJECT_ROOT / "src/moraweave/contracts.py").unlink(missing_ok=True)
    write(PROJECT_ROOT / "tests/acoustic_memory/test_release.py", RELEASE_TEST)
    patch_adapters()
    patch_pyproject()
    patch_version()
    prepend_once(PROJECT_ROOT / "README.md", "# MoraWeave v0.2.0 frontier update", README_BANNER)
    prepend_once(PROJECT_ROOT / "CHANGELOG.md", "## 0.2.0 — 2026-08-29", CHANGELOG_BANNER)
    (REPO_ROOT / ".github/workflows/moraweave-apply-frontier-v2.yml").unlink(missing_ok=True)
    print("MoraWeave frontier v0.2 migration applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
