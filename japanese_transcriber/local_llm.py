from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse

from .normalization import deterministic_normalize
from .types import Segment


@dataclass(slots=True)
class NormalizationGuard:
    min_similarity: float = 0.68
    min_length_ratio: float = 0.65
    max_length_ratio: float = 1.35
    max_block_chars: int = 2400
    block_size: int = 16


@dataclass(slots=True)
class NormalizationResult:
    segments: list[dict[str, str]]
    rejected_segment_ids: list[str]
    model: str
    endpoint_origin: str


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname.strip("[]")).is_loopback
    except ValueError:
        try:
            return all(ipaddress.ip_address(item[4][0]).is_loopback for item in socket.getaddrinfo(hostname, None))
        except (socket.gaierror, ValueError):
            return False


def validate_local_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Ollama endpoint must use http or https")
    if not _is_loopback_host(parsed.hostname):
        raise ValueError("remote LLM endpoints are disabled; use a loopback Ollama endpoint")
    path = parsed.path.rstrip("/")
    if path in {"", "/api"}:
        path = "/api/chat"
    elif path != "/api/chat":
        raise ValueError("Ollama endpoint path must be /api/chat")
    if parsed.username or parsed.password:
        raise ValueError("credentials in the Ollama endpoint URL are not supported")
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}{path}"


def _segment_blocks(segments: list[Segment], guard: NormalizationGuard) -> list[list[Segment]]:
    blocks: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0
    for segment in segments:
        length = len(segment.text)
        if current and (len(current) >= guard.block_size or current_chars + length > guard.max_block_chars):
            blocks.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += length
    if current:
        blocks.append(current)
    return blocks


def _validate_candidate(original: str, candidate: str, guard: NormalizationGuard) -> bool:
    original_cmp = deterministic_normalize(original)
    candidate_cmp = deterministic_normalize(candidate)
    if not candidate_cmp:
        return False
    ratio = len(candidate_cmp) / max(1, len(original_cmp))
    if ratio < guard.min_length_ratio or ratio > guard.max_length_ratio:
        return False
    return SequenceMatcher(None, original_cmp, candidate_cmp).ratio() >= guard.min_similarity


def validate_normalized_block(block: list[Segment], payload: object, guard: NormalizationGuard) -> tuple[list[dict[str, str]], list[str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("local LLM response must contain a segments array")
    rows = payload["segments"]
    expected = {segment.id: segment for segment in block}
    actual_ids = [str(row.get("id")) for row in rows if isinstance(row, dict)]
    if len(actual_ids) != len(expected) or set(actual_ids) != set(expected):
        raise ValueError("local LLM must return every segment ID exactly once")

    accepted: list[dict[str, str]] = []
    rejected: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each normalized segment must be an object")
        segment_id = str(row.get("id"))
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"normalized segment {segment_id} has no text")
        original = expected[segment_id].text
        if _validate_candidate(original, text, guard):
            accepted.append({"id": segment_id, "text": deterministic_normalize(text)})
        else:
            accepted.append({"id": segment_id, "text": deterministic_normalize(original)})
            rejected.append(segment_id)
    accepted.sort(key=lambda item: expected[item["id"]].index)
    return accepted, rejected


class OllamaNormalizer:
    def __init__(self, *, model: str, endpoint: str = "http://127.0.0.1:11434/api/chat", timeout_seconds: float = 120.0, guard: NormalizationGuard | None = None) -> None:
        if ":cloud" in model.lower() or model.lower().startswith("cloud/"):
            raise ValueError("cloud-routed model names are disabled")
        self.model = model
        self.endpoint = validate_local_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self.guard = guard or NormalizationGuard()

    def normalize(self, segments: list[Segment], *, context: str = "") -> NormalizationResult:
        output: list[dict[str, str]] = []
        rejected: list[str] = []
        for block in _segment_blocks(segments, self.guard):
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["segments"],
                "properties": {
                    "segments": {
                        "type": "array",
                        "minItems": len(block),
                        "maxItems": len(block),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "text"],
                            "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                        },
                    }
                },
            }
            prompt = (
                "あなたは日本語文字起こしの整形器です。内容・固有名詞・数値・助詞・言い間違いを勝手に追加、削除、訂正せず、"
                "空白、句読点、明らかな表記揺れだけを読みやすくしてください。必ず同じIDを一度ずつ返してください。\n"
                f"文脈: {context}\n"
                f"入力: {json.dumps([{'id': s.id, 'text': s.text} for s in block], ensure_ascii=False)}"
            )
            body = json.dumps(
                {
                    "model": self.model,
                    "stream": False,
                    "format": schema,
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": prompt}],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"local Ollama normalization failed: {exc}") from exc
            content = raw.get("message", {}).get("content")
            try:
                payload = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError as exc:
                raise RuntimeError("local Ollama returned invalid JSON") from exc
            accepted, block_rejected = validate_normalized_block(block, payload, self.guard)
            output.extend(accepted)
            rejected.extend(block_rejected)

        return NormalizationResult(
            segments=output,
            rejected_segment_ids=rejected,
            model=self.model,
            endpoint_origin=self.endpoint.rsplit("/api/chat", 1)[0],
        )
