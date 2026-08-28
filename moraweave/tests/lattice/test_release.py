from __future__ import annotations

from moraweave.contracts import CandidateEvidence
from moraweave.lattice import build_evidence_lattice, islands_to_token_spans


def _candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence("spoken", "昨日学校を行きました", acoustic=.9, mora=.9),
        CandidateEvidence("clean", "昨日学校に行きました", acoustic=.6, mora=.5),
        CandidateEvidence("wrong", "昨日会社に行きました", acoustic=.2, mora=.2),
    ]


def test_consensus_spine_and_contradiction_islands() -> None:
    lattice = build_evidence_lattice(_candidates(), pivot_candidate_id="spoken")
    assert lattice.contradiction_islands
    assert "昨日" in "".join(span.text for span in lattice.consensus_spine)
    alternatives = {
        alternative.text
        for island in lattice.contradiction_islands
        for alternative in island.alternatives
    }
    assert any("に" in text or "会社" in text for text in alternatives)


def test_contradiction_island_maps_to_audio_span() -> None:
    pivot = "昨日学校を行きました"
    timeline = [
        {"charStart": index, "charEnd": index + 1, "startMs": index * 100, "endMs": (index + 1) * 100}
        for index, _ in enumerate(pivot)
    ]
    lattice = build_evidence_lattice(
        _candidates(), pivot_candidate_id="spoken", char_timeline=timeline
    )
    spans = islands_to_token_spans(lattice)
    assert spans
    assert all(int(span["endMs"]) > int(span["startMs"]) for span in spans)


def test_insertions_anchor_to_nonzero_pivot_span() -> None:
    lattice = build_evidence_lattice(
        [
            CandidateEvidence("a", "東京です", acoustic=.8),
            CandidateEvidence("b", "東京ではです", acoustic=.7),
        ],
        pivot_candidate_id="a",
    )
    assert lattice.contradiction_islands
    assert all(island.pivot_end > island.pivot_start for island in lattice.contradiction_islands)
