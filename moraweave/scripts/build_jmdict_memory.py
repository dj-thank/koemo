from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from moraweave.memory import HashedNgramMemory
from moraweave.rights import RightsRegistry


def iter_terms(path: Path):
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "entry":
            continue
        kanji = [node.text.strip() for node in element.findall("k_ele/keb") if node.text]
        readings = [node.text.strip() for node in element.findall("r_ele/reb") if node.text]
        for value in [*kanji, *readings]:
            if value:
                yield value
        for left in kanji:
            for right in readings:
                yield f"{left}\t{right}"
        element.clear()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jmdict_xml", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--rights-registry", type=Path, required=True)
    parser.add_argument("--asset-id", default="jmdict-current")
    parser.add_argument("--namespace", default="jmdict-ja")
    args = parser.parse_args()

    registry = RightsRegistry.load(args.rights_registry)
    memory = HashedNgramMemory(args.database)
    report = memory.ingest(
        iter_terms(args.jmdict_xml),
        asset_id=args.asset_id,
        registry=registry,
        namespace=args.namespace,
    )
    memory.close()
    report.update(
        {
            "source": str(args.jmdict_xml),
            "rawXmlRedistributed": False,
            "attributions": registry.export_attributions(),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
