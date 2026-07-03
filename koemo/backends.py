"""要約生成バックエンド（local CT2 / Ollama / OpenAI互換）。"""
import os
import glob
from pathlib import Path

from .gpu import gpu_ok
from .readiness import model_load_error


def find_summary_model(explicit=""):
    """要約用CT2モデルのフォルダを探す（明示パス → HFキャッシュ内 Qwen2.5 ct2）。"""
    if explicit and (Path(explicit) / "model.bin").is_file():
        return explicit
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    cands = []
    for pat in ("models--*Qwen2.5*Instruct*ct2*", "models--*Qwen*ct2*int8*"):
        for snap in glob.glob(str(hub / pat / "snapshots" / "*")):
            if (Path(snap) / "model.bin").is_file():
                cands.append(snap)
    cands.sort(key=lambda d: os.path.getsize(os.path.join(d, "model.bin")), reverse=True)
    return cands[0] if cands else None


class SummaryBackend:
    def generate(self, system, user, max_tokens=1024):
        raise NotImplementedError

    def unload(self):
        return False


class LocalCT2Backend(SummaryBackend):
    def __init__(self, model_dir=""):
        self.model_dir = model_dir
        self._gen = None
        self._tok = None

    def ensure_model(self, on_progress=None):
        if self._gen is not None:
            return
        path = find_summary_model(self.model_dir)
        if not path:
            raise RuntimeError(
                "要約用ローカルモデルが見つかりません。\n"
                "初回セットアップで setup.bat を実行するか、Qwen2.5 Instruct の CTranslate2(int8) フォルダを\n"
                "設定の「要約モデルのパス」に指定してください。")
        if on_progress:
            on_progress("要約モデル読み込み中...")
        try:
            import ctranslate2
            from transformers import AutoTokenizer
            import transformers
            transformers.logging.set_verbosity_error()
            self._tok = AutoTokenizer.from_pretrained(path)
            if gpu_ok():
                try:
                    self._gen = ctranslate2.Generator(path, device="cuda", compute_type="int8_float16")
                    return
                except Exception:
                    pass   # VRAM不足等はCPUへ
            self._gen = ctranslate2.Generator(path, device="cpu", compute_type="int8")
        except Exception as e:
            self._tok = None
            self._gen = None
            raise RuntimeError(model_load_error("要約", "Qwen2.5-3B-Instruct-ct2-int8", e)) from e

    def generate(self, system, user, max_tokens=1024):
        self.ensure_model()
        prompt = self._tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        tokens = self._tok.convert_ids_to_tokens(self._tok.encode(prompt, add_special_tokens=False))
        res = self._gen.generate_batch(
            [tokens], max_length=max_tokens, sampling_topk=1, repetition_penalty=1.05,
            end_token="<|im_end|>", include_prompt_in_result=False)
        return self._tok.decode(res[0].sequences_ids[0], skip_special_tokens=True).strip()

    def unload(self):
        had_model = self._gen is not None
        self._gen = None
        self._tok = None
        return had_model


class OllamaBackend(SummaryBackend):
    def __init__(self, model="qwen2.5:3b", base_url="http://localhost:11434"):
        self.model = model or "qwen2.5:3b"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")

    def generate(self, system, user, max_tokens=1024):
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("Ollamaバックエンドには httpx が必要です。setup.bat を再実行してください。") from e
        try:
            r = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=180,
            )
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollamaに接続できません。Ollamaを起動し、モデル `{self.model}` を用意してください。") from e
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "").strip()


class OpenAICompatBackend(SummaryBackend):
    def __init__(self, base_url="", api_key="", model=""):
        self.base_url = (base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()

    def generate(self, system, user, max_tokens=1024):
        if not self.base_url or not self.model:
            raise RuntimeError("OpenAI互換バックエンドには base_url と model の設定が必要です。")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("OpenAI互換バックエンドには openai パッケージが必要です。setup.bat を再実行してください。") from e
        client = OpenAI(base_url=self.base_url, api_key=self.api_key or "koemo")
        res = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        return (res.choices[0].message.content or "").strip()
