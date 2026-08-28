from __future__ import annotations

import ipaddress
import json
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from .contracts import CandidateEvidence


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        try:
            addresses = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return False
        return bool(addresses) and all(
            ipaddress.ip_address(row[4][0]).is_loopback for row in addresses
        )


def validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not _is_loopback(parsed.hostname):
        raise ValueError("local teacher endpoint must be loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("credentials, query, and fragment are not allowed in local endpoint")
    path = parsed.path.rstrip("/")
    if path in {"", "/api"}:
        path = "/api/chat"
    if path != "/api/chat":
        raise ValueError("only the Ollama /api/chat path is supported")
    host = parsed.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return f"http://{host}{port}{path}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "redirect blocked", headers, fp)


@dataclass(frozen=True, slots=True)
class TeacherResult:
    probabilities: dict[str, float]
    model: str
    endpoint_origin: str


class LocalTeacherClient:
    def __init__(
        self,
        *,
        model: str,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: float = 90.0,
    ) -> None:
        lowered = model.lower()
        if ":cloud" in lowered or lowered.startswith("cloud/"):
            raise ValueError("cloud-routed model names are disabled")
        self.model = model
        self.endpoint = validate_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )

    def probabilities(
        self,
        candidates: list[CandidateEvidence],
        *,
        context: str = "",
    ) -> TeacherResult:
        if len(candidates) < 2:
            raise ValueError("teacher comparison requires at least two candidates")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["probabilities"],
            "properties": {
                "probabilities": {
                    "type": "array",
                    "minItems": len(ids),
                    "maxItems": len(ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "p"],
                        "properties": {
                            "id": {"type": "string"},
                            "p": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                }
            },
        }
        prompt = (
            "あなたは日本語ASR候補の局所教師です。新しい文章を生成してはいけません。"
            "入力候補IDだけに、文脈上の相対確率を割り当て、合計を1にしてください。"
            "自然さだけで発話誤りを消さず、候補集合の外を想像しないでください。\n"
            f"文脈: {context}\n"
            f"候補: {json.dumps([{'id': c.candidate_id, 'text': c.text} for c in candidates], ensure_ascii=False)}"
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
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local teacher request failed: {exc}") from exc
        content = payload.get("message", {}).get("content")
        parsed = json.loads(content) if isinstance(content, str) else content
        rows = parsed.get("probabilities") if isinstance(parsed, dict) else None
        if not isinstance(rows, list):
            raise ValueError("teacher response has no probabilities array")
        actual_ids = [str(row.get("id")) for row in rows if isinstance(row, dict)]
        if len(actual_ids) != len(ids) or set(actual_ids) != set(ids):
            raise ValueError("teacher response must contain every candidate ID exactly once")
        probabilities = {str(row["id"]): float(row["p"]) for row in rows}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise ValueError("teacher probability is invalid")
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("teacher probabilities sum to zero")
        probabilities = {key: value / total for key, value in probabilities.items()}
        origin = self.endpoint.rsplit("/api/chat", 1)[0]
        return TeacherResult(probabilities=probabilities, model=self.model, endpoint_origin=origin)
