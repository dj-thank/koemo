from __future__ import annotations

from moraweave.contracts import CandidateEvidence
from moraweave.lattice import build_evidence_lattice, islands_to_token_spans


def candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence("spoken", "昨日学校を行きました", acoustic=.9, mora=.9),
        CandidateEvidence("clean", "昨日学校に行きました", acoustic=.6, mora=.5),
        CandidateEvidence("wrong", "昨日会社に行きました", acoustic=.2, mora=.2),
    ]


def test_consensus_and_contradiction_are_separated() -> None:
    lattice = build_evidence_lattice(candidates(), pivot_candidate_id="spoken")
    assert lattice.pivot_text == "昨日学校を行きました"
    assert lattice.contradiction_islands
    combined_consensus = "".join(span.text for span in lattice.consensus_spine)
    assert "昨日" in combined_consensus
    alternatives = {
        alternative.text
        for island in lattice.contradiction_islands
        for alternative in island.alternatives
    }
    assert any("に" in text or "会社" in text for text in alternatives)


def test_character_timeline_maps_island_to_audio() -> None:
    pivot = "昨日学校を行きました"
    timeline = [
        {"charStart": index, "charEnd": index + 1, "startMs": index * 100, "endMs": (index + 1) * 100}
        for index, _ in enumerate(pivot)
    ]
    lattice = build_evidence_lattice(candidates(), pivot_candidate_id="spoken", char_timeline=timeline)
    mapped = [island for island in lattice.contradiction_islands if island.start_ms is not None]
    assert mapped
    spans = islands_to_token_spans(lattice)
    assert spans
    assert all(span["endMs"] > span["startMs"] for span in spans)


def test_insertions_are_anchored_to_nonzero_pivot_span() -> None:
    lattice = build_evidence_lattice(
        [
            CandidateEvidence("a", "東京です", acoustic=.8),
            CandidateEvidence("b", "東京ではです", acoustic=.7),
        ],
        pivot_candidate_id="a",
    )
    assert lattice.contradiction_islands
    assert all(island.pivot_end > island.pivot_start for island in lattice.contradiction_islands)
