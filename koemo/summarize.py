"""構造化要約（ローカルLLM: CTranslate2 + Qwen2.5）。APIキー不要・GPU優先。

タイトル / 要旨 / 主要トピック / 決定事項 / アクションアイテム / 未解決の質問。
長尺はチャンク分割して各チャンクの要点抽出→統合（map-reduce）。
"""
import gc
import re
import time
import threading

from .backends import LocalCT2Backend, OllamaBackend, OpenAICompatBackend

SECTIONS = {
    "ja": ["要旨", "主要トピック", "決定事項", "アクションアイテム", "未解決の質問"],
    "en": ["Summary", "Key Topics", "Decisions", "Action Items", "Open Questions"],
}

class Summarizer:
    def __init__(self, model_dir="", idle_sec=300, keep_warm=False, cfg=None):
        self.model_dir    = model_dir
        self.idle_sec     = idle_sec
        self.keep_warm    = keep_warm
        self.cfg          = cfg or {}
        self._backend     = None
        self._backend_key = None
        self._last_used   = 0
        self.last_seconds = 0.0
        self._lock        = threading.Lock()

    def _make_backend(self):
        name = self.cfg.get("summary_backend", "local")
        if name == "ollama":
            return OllamaBackend(self.cfg.get("ollama_model", "qwen2.5:3b"),
                                 self.cfg.get("ollama_base_url", "http://localhost:11434"))
        if name == "openai_compat":
            return OpenAICompatBackend(self.cfg.get("openai_base_url", ""),
                                       self.cfg.get("openai_api_key", ""),
                                       self.cfg.get("openai_model", ""))
        return LocalCT2Backend(self.model_dir)

    def _current_backend_key(self):
        name = self.cfg.get("summary_backend", "local")
        if name == "ollama":
            return (name, self.cfg.get("ollama_model", "qwen2.5:3b"),
                    self.cfg.get("ollama_base_url", "http://localhost:11434"))
        if name == "openai_compat":
            return (name, self.cfg.get("openai_base_url", ""),
                    self.cfg.get("openai_api_key", ""), self.cfg.get("openai_model", ""))
        return ("local", self.model_dir)

    def _ensure_backend(self):
        key = self._current_backend_key()
        if self._backend is None or key != self._backend_key:
            if self._backend:
                self._backend.unload()
            self._backend = self._make_backend()
            self._backend_key = key

    def _ensure_model(self, on_progress=None):
        self._ensure_backend()
        ensure = getattr(self._backend, "ensure_model", None)
        if ensure:
            ensure(on_progress)

    def _generate(self, system, user, max_tokens=1024):
        self._ensure_backend()
        return self._backend.generate(system, user, max_tokens=max_tokens)

    @staticmethod
    def _chunks(text, max_chars=12000):
        if len(text) <= max_chars:
            return [text]
        out, cur, size = [], [], 0
        for line in text.split("\n"):
            if size + len(line) > max_chars and cur:
                out.append("\n".join(cur)); cur = []; size = 0
            cur.append(line); size += len(line) + 1
        if cur:
            out.append("\n".join(cur))
        return out

    @staticmethod
    def _query_terms(question):
        raw_terms = re.findall(r"[A-Za-z0-9_]+|[一-龯ぁ-んァ-ンー]{2,}", (question or "").lower())
        terms = []
        for term in raw_terms:
            if term not in terms:
                terms.append(term)
            if re.fullmatch(r"[一-龯ぁ-んァ-ンー]{5,}", term):
                for size in (4, 3):
                    for i in range(0, len(term) - size + 1):
                        part = term[i:i + size]
                        if part not in terms:
                            terms.append(part)
        return terms

    @classmethod
    def _select_chat_context(cls, question, transcript, chunk_chars=8000, max_context_chars=24000):
        chunks = cls._chunks(transcript, max_chars=chunk_chars)
        if len(chunks) <= 2 or sum(len(c) for c in chunks) <= max_context_chars:
            return "\n".join(chunks)

        terms = cls._query_terms(question)
        scored = []
        for idx, chunk in enumerate(chunks):
            low = chunk.lower()
            score = sum(low.count(term) * max(1, len(term)) for term in terms)
            scored.append((score, idx, chunk))

        selected: dict[int, str] = {0: chunks[0], len(chunks) - 1: chunks[-1]}
        for score, idx, chunk in sorted(scored, key=lambda x: (-x[0], x[1])):
            if score <= 0:
                break
            selected[idx] = chunk
            if sum(len(c) for c in selected.values()) >= max_context_chars:
                break
        if len(selected) <= 2 and len(chunks) > 2:
            selected[len(chunks) // 2] = chunks[len(chunks) // 2]

        parts = []
        total = 0
        per_chunk_limit = max(1200, max_context_chars // max(1, len(selected)))
        for idx in sorted(selected):
            chunk = cls._excerpt_chunk(selected[idx], terms, per_chunk_limit)
            if total + len(chunk) > max_context_chars and parts:
                continue
            parts.append(f"[chunk {idx + 1}/{len(chunks)}]\n{chunk}")
            total += len(chunk)
        return "\n\n...\n\n".join(parts)

    @staticmethod
    def _excerpt_chunk(chunk, terms, limit):
        if len(chunk) <= limit:
            return chunk
        low = chunk.lower()
        positions = [low.find(term) for term in terms if term and low.find(term) >= 0]
        center = min(positions) if positions else 0
        start = max(0, center - limit // 2)
        end = min(len(chunk), start + limit)
        if end - start < limit:
            start = max(0, end - limit)
        prefix = "...\n" if start else ""
        suffix = "\n..." if end < len(chunk) else ""
        return prefix + chunk[start:end] + suffix

    def _section_names(self, language):
        raw = self.cfg.get("summary_sections") or SECTIONS.get(language, SECTIONS["ja"])
        if isinstance(raw, str):
            sections = [s.strip() for s in raw.split(",")]
        else:
            sections = [str(s).strip() for s in raw]
        sections = [s for s in sections if s]
        return sections or SECTIONS.get(language, SECTIONS["ja"])

    def _system_message(self, language):
        lang = "日本語" if language == "ja" else "English"
        msg = (f"あなたは優秀な会議議事録アシスタントです。出力は必ず自然で正確な{lang}のみを使い、"
               "中国語簡体字や誤字を含めないこと。指定された見出しに厳密に従うこと。")
        extra = (self.cfg.get("summary_extra_instructions") or "").strip()
        if extra:
            msg += f"\n追加指示: {extra}"
        return msg, lang

    @staticmethod
    def _plain_lines(transcript):
        lines = []
        for raw in (transcript or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^\*\*[^*]+\*\*:\s*", "", line)
            line = re.sub(r"^[^:：]{1,12}[:：]\s*", "", line)
            line = line.strip(" 　")
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _sentences(lines):
        text = "。".join(line.rstrip("。") for line in lines if line).strip()
        parts = re.split(r"(?<=[。！？!?])\s*|[\r\n]+", text)
        out = []
        for part in parts:
            s = part.strip(" 　。！？!?")
            if not s:
                continue
            if len(s) > 90:
                chunks = [s[i:i + 90] for i in range(0, len(s), 90)]
                out.extend(chunks)
            else:
                out.append(s)
        return out

    @staticmethod
    def _pick_title(sentences, language):
        skip = ("ご視聴ありがとうございました", "ありがとうございました", "thank you", "thanks for watching")
        for sentence in sentences:
            low = sentence.lower()
            if any(s in low for s in skip):
                continue
            cleaned = re.sub(r"[。！？!?].*$", "", sentence).strip(" 　-・")
            if len(cleaned) >= 6:
                return cleaned[:24] if language == "ja" else cleaned[:60]
        return "会議メモ" if language == "ja" else "Meeting Notes"

    @staticmethod
    def _bullets(items, limit=5):
        out = []
        seen = set()
        for item in items:
            cleaned = item.strip(" 　-・")
            if not cleaned:
                continue
            key = cleaned[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append(f"- {cleaned}")
            if len(out) >= limit:
                break
        return "\n".join(out) if out else "- 特になし"

    def fast_summarize(self, transcript, language="ja"):
        """LLMを使わず、停止直後に返せる軽量な議事録を作る。"""
        transcript = (transcript or "").strip()
        if not transcript:
            return "（無題）", "（文字起こしが空のため、要約はありません）"

        lines = self._plain_lines(transcript)
        sentences = self._sentences(lines)
        title = self._pick_title(sentences, language)
        secs = self._section_names(language)

        if not sentences:
            sentences = lines[:]
        overview = "。".join(sentences[:3]).strip()
        if overview and language == "ja" and not overview.endswith(("。", "！", "？")):
            overview += "。"
        if not overview:
            overview = "文字起こしから会話内容を保存しました。"

        decision_re = re.compile(r"(決定|合意|採用|確定|承認|締切|期限|リリース|方針)")
        action_re = re.compile(r"(TODO|対応|担当|確認します|確認する|お願いします|してください|やります|進めます|作成|修正)")
        question_re = re.compile(r"(未解決|確認が必要|質問|課題|懸念|要確認|どう|なぜ|いつ|誰)")
        decisions = [s for s in sentences if decision_re.search(s)]
        actions = [s for s in sentences if action_re.search(s)]
        questions = [s for s in sentences if question_re.search(s)]

        section_body = {
            secs[0]: overview,
        }
        if len(secs) > 1:
            section_body[secs[1]] = self._bullets(sentences[:5])
        if len(secs) > 2:
            section_body[secs[2]] = self._bullets(decisions)
        if len(secs) > 3:
            section_body[secs[3]] = self._bullets(actions)
        if len(secs) > 4:
            section_body[secs[4]] = self._bullets(questions)

        parts = []
        for i, sec in enumerate(secs):
            body = section_body.get(sec)
            if body is None:
                body = self._bullets(sentences[:5]) if i else overview
            parts.append(f"## {sec}\n{body}")
        self.last_seconds = 0.0
        self._last_used = time.time()
        return title, "\n\n".join(parts)

    def summarize(self, transcript, language="ja", on_progress=None):
        """戻り: (title, body_markdown)。"""
        transcript = (transcript or "").strip()
        if not transcript:
            return "（無題）", "（文字起こしが空のため、要約はありません）"
        sysmsg, lang = self._system_message(language)
        secs = self._section_names(language)
        fmt = "\n".join(
            [f"## {secs[0]}\n（会議全体を2〜4文で簡潔に要約。必ず記述する）"] +
            [f"## {s}\n- 箇条書き（該当が無ければ「特になし」）" for s in secs[1:]])

        with self._lock:
            self._ensure_model(on_progress)
            t0 = time.time()
            chunks = self._chunks(transcript)

            if len(chunks) == 1:
                if on_progress:
                    on_progress("AI要約を生成中...")
                source = chunks[0]
            else:
                notes = []
                for i, ch in enumerate(chunks):
                    if on_progress:
                        on_progress(f"要約中... ({i + 1}/{len(chunks)})")
                    notes.append(self._generate(
                        sysmsg,
                        f"次の会議の一部から、重要な事実・決定・依頼を{lang}の箇条書きで抽出してください。\n\n{ch}",
                        max_tokens=512))
                source = "\n".join(notes)
                if on_progress:
                    on_progress("要約を統合中...")

            body = self._generate(
                sysmsg,
                (f"以下の会議内容を{lang}で要約してください。次の見出しをこの順で必ず使ってください。\n\n"
                 f"{fmt}\n\n会議内容:\n{source}"),
                max_tokens=1024)

            if on_progress:
                on_progress("タイトル生成中...")
            title = self._generate(
                sysmsg,
                f"次の会議内容に簡潔な{lang}のタイトルを1つだけ付けてください。"
                f"記号や引用符は不要、20文字以内、タイトルのみ出力。\n\n{transcript[:3000]}",
                max_tokens=40).splitlines()[0].strip(" 　\"'`#・-") or "会議メモ"
            self.last_seconds = time.time() - t0
            self._last_used = time.time()
            return title, body

    def chat(self, question, transcript, history=None, language="ja"):
        """会議文字起こしを根拠にしたQ&A。"""
        question = (question or "").strip()
        transcript = (transcript or "").strip()
        if not question:
            return ""
        if not transcript:
            return "文字起こしが空のため回答できません。"

        sysmsg, lang = self._system_message(language)
        sysmsg += f"\n以下の会議内容のみに基づき、{lang}で簡潔に回答してください。根拠が無い場合は不明と答えてください。"
        context = self._select_chat_context(question, transcript)
        hist = ""
        if history:
            recent = history[-6:]
            hist = "\n".join(f"{role}: {text}" for role, text in recent if text)

        user = f"会議内容:\n{context}\n\n"
        if hist:
            user += f"これまでの会話:\n{hist}\n\n"
        user += f"質問: {question}"

        with self._lock:
            self._ensure_model()
            t0 = time.time()
            answer = self._generate(sysmsg, user, max_tokens=512)
            self.last_seconds = time.time() - t0
            self._last_used = time.time()
            return answer

    def maybe_unload(self):
        with self._lock:
            if self.keep_warm:
                return False
            if self._backend is not None and self._last_used > 0:
                if time.time() - self._last_used > self.idle_sec:
                    unloaded = self._backend.unload()
                    if unloaded:
                        gc.collect()
                        return True
        return False

    def reload(self, model_dir="", idle_sec=300, keep_warm=False, cfg=None):
        with self._lock:
            self.idle_sec = idle_sec
            self.keep_warm = keep_warm
            old_key = self._current_backend_key()
            self.model_dir = model_dir
            if cfg is not None:
                self.cfg = cfg
            if old_key != self._current_backend_key():
                if self._backend:
                    self._backend.unload()
                self._backend = None
                self._backend_key = None
                gc.collect()
