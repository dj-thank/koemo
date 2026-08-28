"""Fail-closed compatibility checks for the low-level faster-whisper N-best path."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
import re
import sys
from typing import Any

_TESTED_FASTER_WHISPER_VERSION = "1.2.1"
_TESTED_FASTER_WHISPER = (1, 2, 1)
_REQUIRED_CTRANSLATE2_MAJOR = 4


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    compatible: bool
    faster_whisper_version: str | None
    ctranslate2_version: str | None
    python_version: str
    platform: str
    model_type: str
    capabilities: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "compatible": self.compatible,
            "versions": {
                "fasterWhisper": self.faster_whisper_version,
                "ctranslate2": self.ctranslate2_version,
                "python": self.python_version,
            },
            "platform": self.platform,
            "modelType": self.model_type,
            "capabilities": list(self.capabilities),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _release_tuple(value: str) -> tuple[int, ...]:
    """Extract leading release digits without merging prerelease suffix digits."""

    release = value.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for component in release.split("."):
        match = re.match(r"(\d+)", component)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _has_callable(root: Any, path: str) -> bool:
    current = root
    for name in path.split("."):
        if not hasattr(current, name):
            return False
        current = getattr(current, name)
    return callable(current)


def probe_faster_whisper_runtime(
    model: Any,
    *,
    faster_whisper_version: str | None = None,
    ctranslate2_version: str | None = None,
) -> RuntimeCompatibility:
    """Inspect versions and the exact attributes required by the N-best adapter."""

    fw_version = (
        faster_whisper_version
        if faster_whisper_version is not None
        else _package_version("faster-whisper")
    )
    ct2_version = (
        ctranslate2_version
        if ctranslate2_version is not None
        else _package_version("ctranslate2")
    )
    errors: list[str] = []
    warnings: list[str] = []
    capabilities: list[str] = []

    required_callables = (
        "model.generate",
        "encode",
        "get_prompt",
        "feature_extractor",
    )
    for path in required_callables:
        if _has_callable(model, path):
            capabilities.append(path)
        else:
            errors.append(f"missing_callable:{path}")

    required_attributes = (
        "hf_tokenizer",
        "model.is_multilingual",
        "max_length",
    )
    for path in required_attributes:
        current = model
        found = True
        for name in path.split("."):
            if not hasattr(current, name):
                found = False
                break
            current = getattr(current, name)
        if found:
            capabilities.append(path)
        else:
            errors.append(f"missing_attribute:{path}")

    max_length = getattr(model, "max_length", None)
    if max_length is not None and (
        not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0
    ):
        errors.append("invalid_attribute:max_length")

    optional_callables = (
        "model.detect_language",
        "_split_segments_by_timestamps",
        "add_word_timestamps",
    )
    for path in optional_callables:
        if _has_callable(model, path):
            capabilities.append(path)
        else:
            warnings.append(f"missing_optional_callable:{path}")

    if fw_version is None:
        errors.append("missing_package:faster-whisper")
    else:
        parsed = _release_tuple(fw_version)
        if parsed < _TESTED_FASTER_WHISPER:
            errors.append(
                "unsupported_faster_whisper_version:"
                f"{fw_version}<1.2.1"
            )
        elif fw_version != _TESTED_FASTER_WHISPER_VERSION:
            warnings.append(
                "untested_faster_whisper_version:"
                f"{fw_version};tested={_TESTED_FASTER_WHISPER_VERSION}"
            )

    if ct2_version is None:
        errors.append("missing_package:ctranslate2")
    else:
        parsed_ct2 = _release_tuple(ct2_version)
        if not parsed_ct2 or parsed_ct2[0] != _REQUIRED_CTRANSLATE2_MAJOR:
            errors.append(
                "unsupported_ctranslate2_version:"
                f"{ct2_version};required_major=4"
            )

    return RuntimeCompatibility(
        compatible=not errors,
        faster_whisper_version=fw_version,
        ctranslate2_version=ct2_version,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        model_type=f"{type(model).__module__}.{type(model).__qualname__}",
        capabilities=tuple(sorted(set(capabilities))),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def assert_faster_whisper_runtime_compatible(
    model: Any,
    *,
    faster_whisper_version: str | None = None,
    ctranslate2_version: str | None = None,
) -> RuntimeCompatibility:
    report = probe_faster_whisper_runtime(
        model,
        faster_whisper_version=faster_whisper_version,
        ctranslate2_version=ctranslate2_version,
    )
    if not report.compatible:
        raise RuntimeError(
            "incompatible faster-whisper N-best runtime: " + ", ".join(report.errors)
        )
    return report
