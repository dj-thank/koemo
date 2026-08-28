"""MoraWeave: mora-aware evidence-fused Japanese speech transcription."""

from .contracts import CandidateEvidence, MoraUnit, NormalizedTranscript, ObservedTranscript
from .gates import GateConfig, gate_candidates
from .pipeline import MoraWeavePipeline, PipelineResult

__all__ = [
    "CandidateEvidence",
    "MoraUnit",
    "NormalizedTranscript",
    "ObservedTranscript",
    "GateConfig",
    "gate_candidates",
    "MoraWeavePipeline",
    "PipelineResult",
]

__version__ = "0.1.0"
