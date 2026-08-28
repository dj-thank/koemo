from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .contracts import RankedCandidate
from .shadow_lattice import DualEvidenceLattice, ShadowIsland

ActionKind = Literal[
    "whisper-relisten",
    "qwen-second-ear",
    "forced-align",
    "local-teacher-check",
]


@dataclass(frozen=True, slots=True)
class EvidenceBudget:
    total_cost_ms: int = 12_000
    max_actions: int = 8
    minimum_utility: float = 0.00004

    def __post_init__(self) -> None:
        if self.total_cost_ms < 0 or self.max_actions < 0:
            raise ValueError("evidence budget values must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidenceAction:
    action_id: str
    kind: ActionKind
    start_ms: int | None
    end_ms: int | None
    estimated_cost_ms: int
    expected_information_gain: float
    criticality: float
    utility: float
    reasons: tuple[str, ...]
    affects_observed_decision: bool


@dataclass(frozen=True, slots=True)
class EvidenceAcquisitionPlan:
    selected: tuple[EvidenceAction, ...]
    rejected: tuple[EvidenceAction, ...]
    budget_ms: int
    used_ms: int
    expected_information_gain: float
    stopping_reason: str


def _duration_ms(island: ShadowIsland) -> int:
    if island.start_ms is None or island.end_ms is None:
        return 2_000
    return max(160, island.end_ms - island.start_ms)


def _action_cost(kind: ActionKind, duration_ms: int) -> int:
    fixed, realtime_factor = {
        "whisper-relisten": (120, 0.18),
        "qwen-second-ear": (750, 0.55),
        "forced-align": (320, 0.22),
        "local-teacher-check": (180, 0.0),
    }[kind]
    return int(round(fixed + duration_ms * realtime_factor))


def _fit(kind: ActionKind, island: ShadowIsland) -> float:
    kinds = set(island.kinds)
    if kind == "whisper-relisten":
        return 0.74 + 0.20 * island.posterior_ambiguity
    if kind == "qwen-second-ear":
        important = bool(
            kinds
            & {
                "number",
                "date-or-time",
                "currency",
                "kanji-or-proper-noun",
                "latin-acronym-or-term",
            }
        )
        return 0.86 if important else 0.58
    if kind == "forced-align":
        phonetic = bool(
            kinds
            & {
                "special-mora",
                "particle-or-functional",
                "phonetic-or-punctuation",
            }
        )
        return 0.82 if phonetic else 0.50
    if kind == "local-teacher-check":
        return 0.52 if "particle-or-functional" in kinds else 0.32
    raise AssertionError(kind)


def _candidate_actions(
    lattice: DualEvidenceLattice,
    *,
    global_uncertainty: float,
) -> list[EvidenceAction]:
    actions: list[EvidenceAction] = []
    for island_index, island in enumerate(lattice.contradiction_islands):
        duration = _duration_ms(island)
        base_gain = island.expected_information_gain * (
            0.55 + 0.45 * global_uncertainty
        )
        for kind in (
            "whisper-relisten",
            "forced-align",
            "qwen-second-ear",
            "local-teacher-check",
        ):
            fit = _fit(kind, island)
            gain = min(1.0, max(0.0, base_gain * fit))
            cost = _action_cost(kind, duration)
            utility = gain / max(1, cost)
            reasons = tuple(
                dict.fromkeys(
                    (
                        "contradiction-island",
                        *island.kinds,
                        "mora-shadow"
                        if lattice.alignment_level == "mora"
                        else "surface-shadow",
                    )
                )
            )
            actions.append(
                EvidenceAction(
                    action_id=f"island-{island_index:04d}:{kind}",
                    kind=kind,
                    start_ms=island.start_ms,
                    end_ms=island.end_ms,
                    estimated_cost_ms=cost,
                    expected_information_gain=gain,
                    criticality=island.criticality,
                    utility=utility,
                    reasons=reasons,
                    affects_observed_decision=kind != "local-teacher-check",
                )
            )
    return actions


def plan_evidence_acquisition(
    ranked: Sequence[RankedCandidate],
    lattice: DualEvidenceLattice,
    *,
    budget: EvidenceBudget | None = None,
    enabled: Sequence[ActionKind] = (
        "whisper-relisten",
        "qwen-second-ear",
        "forced-align",
        "local-teacher-check",
    ),
) -> EvidenceAcquisitionPlan:
    """Select evidence requests by expected information gain per cost."""

    budget = budget or EvidenceBudget()
    if not ranked:
        return EvidenceAcquisitionPlan((), (), budget.total_cost_ms, 0, 0.0, "no-ranked-candidates")
    if not ranked[0].gate.needs_relisten:
        return EvidenceAcquisitionPlan((), (), budget.total_cost_ms, 0, 0.0, "observation-already-confident")
    if not lattice.contradiction_islands:
        return EvidenceAcquisitionPlan((), (), budget.total_cost_ms, 0, 0.0, "no-localized-contradiction")

    enabled_set = set(enabled)
    global_uncertainty = min(
        1.0,
        0.55 * ranked[0].gate.entropy
        + 0.30 * ranked[0].gate.disagreement
        + 0.15 * (1.0 - ranked[0].gate.evidence_coverage),
    )
    actions = [
        action
        for action in _candidate_actions(lattice, global_uncertainty=global_uncertainty)
        if action.kind in enabled_set
    ]
    actions.sort(
        key=lambda action: (
            -action.utility,
            -action.expected_information_gain,
            action.estimated_cost_ms,
            action.action_id,
        )
    )

    selected: list[EvidenceAction] = []
    rejected: list[EvidenceAction] = []
    used = 0
    selected_island_kind: set[tuple[str, ActionKind]] = set()
    for action in actions:
        island_id = action.action_id.split(":", 1)[0]
        key = (island_id, action.kind)
        if key in selected_island_kind:
            rejected.append(action)
            continue
        if action.utility < budget.minimum_utility:
            rejected.append(action)
            continue
        if len(selected) >= budget.max_actions:
            rejected.append(action)
            continue
        if used + action.estimated_cost_ms > budget.total_cost_ms:
            rejected.append(action)
            continue
        selected.append(action)
        selected_island_kind.add(key)
        used += action.estimated_cost_ms

    stopping_reason = (
        "budget-exhausted"
        if selected and used >= budget.total_cost_ms
        else "utility-frontier-reached"
        if selected
        else "no-positive-utility-action"
    )
    return EvidenceAcquisitionPlan(
        selected=tuple(
            sorted(
                selected,
                key=lambda action: (
                    action.start_ms is None,
                    action.start_ms or 0,
                    action.kind,
                ),
            )
        ),
        rejected=tuple(rejected),
        budget_ms=budget.total_cost_ms,
        used_ms=used,
        expected_information_gain=sum(
            action.expected_information_gain for action in selected
        ),
        stopping_reason=stopping_reason,
    )
