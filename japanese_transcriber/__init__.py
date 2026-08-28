"""Complete local-first Japanese transcription toolkit."""

from .engine import EngineConfig, FasterWhisperEngine
from .pipeline import PipelineConfig, transcribe_file, verify_observed_integrity

__all__ = [
    "EngineConfig",
    "FasterWhisperEngine",
    "PipelineConfig",
    "transcribe_file",
    "verify_observed_integrity",
]
__version__ = "1.0.0"
