from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .contracts import CandidateEvidence
from .memory import HashedNgramMemory
from .pipeline import MoraWeavePipeline
from .rights import RightsRegistry


def _candidate(row: dict[str, object]) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=str(row["id"]),
        text=str(row["text"]),
        token_ids=tuple(int(value) for value in row.get("tokenIds", [])),
        acoustic=_optional_float(row.get("acoustic")),
        mora=_optional_float(row.get("mora")),
        lexical=_optional_float(row.get("lexical")),
        preservation=_optional_float(row.get("preservation")),
        teacher=_optional_float(row.get("teacher")),
        metadata=dict(row.get("metadata", {})),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _write(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def demo(output: str | Path) -> dict[str, object]:
    candidates = [
        CandidateEvidence(
            "c0",
            "昨日学校を行きました",
            acoustic=0.91,
            mora=0.89,
            lexical=0.38,
            preservation=0.96,
            teacher=0.08,
        ),
        CandidateEvidence(
            "c1",
            "昨日学校に行きました",
            acoustic=0.62,
            mora=0.58,
            lexical=0.95,
            preservation=0.42,
            teacher=0.92,
        ),
        CandidateEvidence(
            "c2",
            "昨日会社に行きました",
            acoustic=0.31,
            mora=0.22,
            lexical=0.83,
            preservation=0.21,
            teacher=0.71,
        ),
    ]
    result = MoraWeavePipeline().run(candidates, source_audio_sha256="a" * 64)
    payload = result.as_dict()
    _write(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moraweave")
    sub = parser.add_subparsers(dest="command", required=True)

    demo_parser = sub.add_parser("demo", help="run a model-free evidence-fusion demonstration")
    demo_parser.add_argument("--output", default="runs/demo.json")

    fuse_parser = sub.add_parser("fuse", help="fuse an N-best JSON candidate list")
    fuse_parser.add_argument("input")
    fuse_parser.add_argument("--output", required=True)

    memory_parser = sub.add_parser("memory-build", help="build a rights-gated hashed n-gram memory")
    memory_parser.add_argument("manifest", help="JSONL rows containing assetId and text")
    memory_parser.add_argument("--database", required=True)
    memory_parser.add_argument("--rights-registry", required=True)
    memory_parser.add_argument("--namespace", default="public-ja")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        payload = demo(args.output)
        print(json.dumps(payload["diagnostics"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "fuse":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        rows = payload["candidates"] if isinstance(payload, dict) else payload
        candidates = [_candidate(row) for row in rows]
        result = MoraWeavePipeline().run(candidates)
        _write(args.output, result.as_dict())
        print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "memory-build":
        registry = RightsRegistry.load(args.rights_registry)
        grouped: dict[str, list[str]] = {}
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(str(row["assetId"]), []).append(str(row["text"]))
        memory = HashedNgramMemory(args.database)
        reports = [
            memory.ingest(
                texts,
                asset_id=asset_id,
                registry=registry,
                namespace=args.namespace,
            )
            for asset_id, texts in sorted(grouped.items())
        ]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0

    raise AssertionError(args.command)
