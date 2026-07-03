"""文字起こし（faster-whisper / GPU優先・CPUフォールバック）とチャンネル統合。"""
import gc
import time
import threading

from .gpu import gpu_ok
from .native_correction import normalize_transcript_text
from .readiness import model_load_error


class Transcriber:
    def __init__(self, model_size, cpu_threads=2, idle_sec=300, keep_warm=False):
        self.model_size   = model_size
        self.cpu_threads  = cpu_threads
        self.idle_sec     = idle_sec
        self.keep_warm    = keep_warm
        self._model       = None
        self._last_used   = 0
        self.last_seconds = 0.0
        self._lock        = threading.Lock()

    def _ensure_model(self, on_progress=None):
        if self._model is None:
            if on_progress:
                on_progress("モデル読み込み中...")
            from faster_whisper import WhisperModel
            if gpu_ok():
                try:
                    self._model = WhisperModel(self.model_size, device="cuda",
                                               compute_type="float16")
                    return
                except Exception:
                    pass   # GPU失敗時はCPUへ
            try:
                self._model = WhisperModel(
                    self.model_size, device="cpu", compute_type="int8",
                    cpu_threads=self.cpu_threads, num_workers=1,
                )
            except Exception as e:
                raise RuntimeError(model_load_error("文字起こし", self.model_size, e)) from e

    def warmup(self, on_progress=None):
        with self._lock:
            self._ensure_model(on_progress)
            self._last_used = time.time()
            return True

    def transcribe_segments(self, audio, language="ja", on_progress=None,
                            vad_filter=True, min_silence_duration_ms=400,
                            speech_pad_ms=150):
        """16kHz float32 1ch を文字起こしし [(start, end, text), ...] を返す。"""
        with self._lock:
            self._ensure_model(on_progress)
            t0 = time.time()
            kwargs = dict(
                beam_size=1,
                vad_filter=vad_filter,
                condition_on_previous_text=False,
                language=language if language != "auto" else None,
                word_timestamps=False,
            )
            if vad_filter:
                kwargs["vad_parameters"] = dict(
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms,
                )
            segments, _ = self._model.transcribe(audio, **kwargs)
            out = []
            for s in segments:
                text = normalize_transcript_text(s.text)
                if text:
                    out.append((float(s.start), float(s.end), text))
            self.last_seconds = time.time() - t0
            self._last_used = time.time()
            return out

    def maybe_unload(self):
        with self._lock:
            if self.keep_warm:
                return False
            if self._model is not None and self._last_used > 0:
                if time.time() - self._last_used > self.idle_sec:
                    self._model = None
                    gc.collect()
                    return True
        return False

    def unload(self):
        """明示的にWhisperモデルを解放する（LLM要約前のVRAM退避用）。"""
        with self._lock:
            had_model = self._model is not None
            self._model = None
            if had_model:
                gc.collect()
            return had_model

    def reload(self, model_size, cpu_threads=2, idle_sec=300, keep_warm=False):
        with self._lock:
            self.idle_sec = idle_sec
            self.keep_warm = keep_warm
            if self.model_size != model_size or self.cpu_threads != cpu_threads:
                self.model_size  = model_size
                self.cpu_threads = cpu_threads
                self._model      = None
                gc.collect()


def merge_rows(rows):
    """[(start, label, text)] を時系列順に統合し話者ラベル付き Markdown にする。"""
    return "\n".join(f"**{lab}**: {txt}" for (st, lab, txt) in sorted(rows, key=lambda r: r[0]))
