# Koemo / コエモ

**Private-by-default meeting recorder, transcriber & summarizer for Windows.**
会議を録音し、文字起こしして、AIで要約する — 既定ではあなたのPCの中だけで完結します。

Koemo captures both your **microphone** and your PC's **system audio** (the other participants),
transcribes each side on-device, and produces a structured summary with a local LLM.
By default it uses no cloud account or API key. Your audio never leaves your machine.
If you opt into Ollama or OpenAI-compatible summary/chat backends, transcript and chat text is sent to the endpoint you configure.

---

## ✨ Features

- 🎤🔊 **Dual capture** — records your mic *and* system audio simultaneously, with acoustic echo cancellation (AEC).
- 📡 **Live transcription preview** — shows immediate mic activity feedback, then uses either Windows-native captions or rolling Whisper depending on the machine/settings.
- 📝 **High-accuracy final transcription** — saved mic/system WAVs are processed by Whisper `large-v3-turbo`; live captions are速報 only and are not mixed into the final transcript.
- 🗣 **Diarization for system audio** — can split the other side into multiple speakers when local diarization models are available.
- 🤖 **Structured local summary** — instant local summary by default, with full local/remote LLM summary available from settings.
- 🌐 **Long meetings** — automatic chunked (map-reduce) summarization.
- 📂 **Audio import and export** — import common audio/video files and export results as Markdown, PDF, or DOCX.
- 📚 **Meeting library** — saved meetings are indexed in local SQLite with search and reopen.
- 🔌 **Summary backends** — local CTranslate2 by default, with opt-in Ollama or OpenAI-compatible endpoints.
- 🧩 **Summary templates** — edit section headings and add custom summary instructions.
- 💬 **Meeting chat** — ask questions against a completed meeting transcript.
- 📅 **Meeting detection** — detects Zoom/Teams/Webex processes and prompts via tray notification.
- 📆 **Calendar title hints** — optional ICS/Outlook calendar lookup can use the current appointment title as the saved meeting title.
- ⚡ **Fast stop-time output** — warmed Whisper `large-v3-turbo` plus instant local summary is designed to finish within 10 seconds for short clips.
- 🔒 **Local by default** — Windows speech recognition for live captions, Whisper for final transcripts, Qwen2.5 (CTranslate2) for summaries. Optional Ollama/OpenAI-compatible backends send transcript/chat text only to the endpoint you configure.
- 🧊 **Lightweight when idle** — models are lazy-loaded and released after 5 minutes of inactivity.

## 🖥 Requirements

- Windows 10/11, Python 3.10+
- (Optional) NVIDIA GPU for faster local summarization
- Microphone; system-audio capture uses WASAPI loopback (no extra drivers)

## 🚀 Setup

```bat
:: 1) install Python 3.10+ (check "Add Python to PATH")
:: 2) run:
setup.bat
```

`setup.bat` installs dependencies, the final transcription model (`large-v3-turbo`), and the summary model (Qwen2.5-3B, CTranslate2 int8).
The optional public-corpus dictionary builder uses `datasets`; install `requirements-tools.txt` only when running `scripts/build_native_corrections.py --hf-dataset ...`.

## ▶️ Usage

1. Launch **start.bat** (or the desktop *Koemo* shortcut). Koemo lives in the system tray.
2. Press **Ctrl + Shift + R** to start recording.
3. A live transcription window appears while recording.
4. Press **Ctrl + Shift + R** again to stop → final transcription + summary → a results window opens.

Outputs are saved to the folder shown in settings (`~/.koemo/config.json` → `save_dir`):

- `recording_*_mic.wav`, `recording_*_system.wav` — the two channels
- `summary_*.md` — structured summary + speaker-labelled transcript

> 💡 Headphones are recommended. When using speakers, AEC suppresses the echo that leaks into the mic.

### Windows Security / Public release

Public release builds must be signed. Unsigned PyInstaller EXEs can trigger SmartScreen / Windows App
Control warnings or `WinError 4551`, and are only suitable for internal beta validation. Koemo's release
helper therefore requires Azure Artifact Signing metadata by default and only creates unsigned artifacts
when `--unsigned-beta` is explicitly passed. Even signed non-Store apps can show SmartScreen warnings
until reputation builds.

The 10-second target is measured on the fastest validated local path. Portable EXE builds do not bundle
CUDA DLLs, so some machines may fall back to CPU for Whisper final transcription.

### Fast Mode

Koemo defaults to:

- `fast_summary=true`: stop-time summary is generated locally without waiting for an LLM.
- `native_only_transcription=false`: the official saved transcript uses Whisper `large-v3-turbo`.
- `live_backend=auto`: GPUが使える場合は日本語精度重視の Whisper rolling、CPU環境ではWindows純正を使う。
- `live_fallback_backend=whisper_rolling`: if Windows speech is blocked, live preview falls back to rolling Whisper.
- `use_live_transcript_on_stop=false`: live速報 is never reused as the official transcript.
- `preload_final_transcriber=true`: the high-accuracy model is warmed at startup.
- Full LLM summaries release the live/final Whisper models first unless `keep_warm=true`, avoiding three model families resident in 8GB VRAM at once.
- `final_channel_policy=auto_dedupe`: mic/system WAVs are both checked when active; both are kept unless the mic is detected as saturated speaker echo or low-level system leakage after AEC. Use `all_active` in settings when every active channel must always be kept.
- `enable_calendar_title_hint=false`: calendar title hints are opt-in; set an ICS file, or enable Outlook lookup when `pywin32` is installed in the running Python environment.

Tray menu:

- **履歴** opens the local meeting library.
- **設定** controls devices, live transcription, final channel policy, meeting detection, calendar title hints, summary backend, API endpoint fields, and summary sections.
- Cloud/API keys are stored only in `~/.koemo/config.json`; do not put keys in repository files.

## 🧠 How it works

```
mic ── Windows native speech or rolling Whisper ── live preview

mic/sys ── DualRecorder + AEC ── WAV archive ── Whisper large-v3-turbo final pass ── channel select/dedupe ── fast local summary
```

- **Audio:** [`soundcard`](https://pypi.org/project/SoundCard/) for WASAPI loopback + mic.
- **Recording memory:** captured audio is streamed to temporary float32 spool files while recording; only the live-preview tail is kept in a bounded ring buffer.
- **AEC:** two-pass frequency-domain Wiener echo canceller using the system channel as reference. It processes STFT frames in bounded batches so long recordings do not allocate all frames at once.
- **Transcription:** Windows native speech recognition or rolling Whisper is used for live preview. The saved transcript is produced by faster-whisper `large-v3-turbo`.
- **Live latency:** mic activity is shown immediately; Windows Speech text hypotheses are OS-dependent and can arrive later than the activity indicator.
- **Japanese correction:** Koemo loads built-in rules plus `~/.koemo/native_corrections.json`; `scripts/build_native_corrections.py` can generate phrase hints from Koemo logs, and can also use public Hugging Face text datasets after `pip install -r requirements-tools.txt`.
- **Important:** if Windows blocks speech recognition, Koemo reports the exact OS error and falls back to Whisper rolling for live preview when configured. Final transcripts remain Whisper-based.
- **Summary:** Qwen2.5-Instruct via [CTranslate2](https://github.com/OpenNMT/CTranslate2), greedy decoding.
- **UI:** PySide6 (Qt) tray app.

Configuration lives in `~/.koemo/config.json` (editable via the tray → Settings).

## 📦 Packaging

```bat
python -m pip install -r requirements.txt -r requirements-build.txt
python -m PyInstaller koemo.spec --noconfirm --clean
```

The packaged app is created at `dist/Koemo/Koemo.exe`. Models are not bundled; Koemo keeps using
`~/.koemo` and the Hugging Face cache. CUDA DLLs are not bundled either, so the packaged build is
safe to run on CPU and can use GPU only when the required runtime DLLs are available on the machine.
Optional corpus-building dependencies such as `datasets`, `pandas`, and `pyarrow` are excluded from
the packaged app because they are not part of the recording/transcription/summary runtime.

### Public EXE release

Signed public artifacts are built through:

```bat
set KOEMO_SIGNTOOL=C:\path\to\signtool.exe
set KOEMO_SIGNING_DLIB=C:\path\to\Azure.CodeSigning.Dlib.dll
set KOEMO_SIGNING_METADATA=C:\path\outside\repo\metadata.json
python scripts\koemo_release_build.py --install-tools
```

The script rebuilds `dist\Koemo`, signs every `.exe` / `.dll` / `.pyd`, verifies signatures, creates a
portable zip, builds a Japanese Inno Setup installer, signs the installer, writes `SHA256SUMS.txt`, and
copies Japanese release notes into `release\`.

For internal validation without signing:

```bat
python scripts\koemo_release_build.py --unsigned-beta --skip-installer
```

Unsigned artifacts are named `UNSIGNED-BETA` and must not be described as worldwide-ready.

### Release pipeline (verify → sign → publish)

1. **Verify**: reproduce the local-GPU summary path end to end and capture a log. See
   [`docs/REIMPORT_VERIFICATION.md`](docs/REIMPORT_VERIFICATION.md) for the latest run
   (transcribe + `Summarizer.summarize()` via CT2/Qwen2.5-3B, with actual generated text).
   Re-run with `scripts/koemo_model_bench.py` / `scripts/koemo_fast_integration_check.py` before
   each release.
2. **Sign**: see [`docs/SIGNING_RUNBOOK.md`](docs/SIGNING_RUNBOOK.md) for the full Azure Artifact
   Signing account setup (one-time, ~$9.99/month Basic tier, requires explicit approval before any
   paid step) and the `scripts\koemo_release_build.py` invocation above (repeatable, no code
   changes needed once credentials exist).
3. **Publish**: attach the signed `release\Koemo-*-Setup.exe`, `release\Koemo-*-portable.zip`,
   `release\SHA256SUMS.txt`, and `release\Koemo-*-release.json` (check `"signed": true`) to the
   GitHub release. Unsigned `UNSIGNED-BETA` artifacts are for internal validation only and must not
   be published as a public release.

## Meetily Comparison

Checked against Meetily official pages on **2026-05-30**:
[Open Source](https://meetily.ai/open-source) and [Docs](https://meetily.ai/docs/).

| Capability | Meetily official docs | Koemo |
|---|---|---|
| Local/offline processing | Yes | Yes, local CT2 is the default and no network is used unless a remote backend is selected |
| Live transcription | Yes | Yes, rolling preview during recording plus final authoritative pass |
| Audio import/export | Yes | Yes, import plus Markdown/PDF/DOCX export |
| Summary templates/providers | Yes | Yes, custom sections/instructions plus local/Ollama/OpenAI-compatible backends |
| Meeting history/search | Yes | Yes, local SQLite library with FTS5/LIKE search |
| Meeting chat/Q&A | Not listed in the checked docs | Yes, transcript-grounded chat from results/history |
| Channel separation | System audio capture is documented | Mic/system are recorded and transcribed as separate channels with AEC |
| Speaker handling | Not listed in the checked docs | System-audio diarization can label 相手1/相手2… |

## 🛣 Roadmap

- Implemented: import existing audio files
- Implemented: real-time (live) transcription preview
- Implemented: within-channel speaker diarization
- Implemented: export to PDF / DOCX / Markdown
- Implemented: meeting history / library + search
- Implemented: summary backends: local / Ollama / OpenAI-compatible
- Implemented: user-defined summary templates
- Implemented: meeting transcript Q&A chat
- Implemented: meeting auto-detection
- Implemented: calendar title hints from ICS / Outlook
- Implemented: single-exe packaging
- Implemented: dedicated icon / logo

## 📄 License

MIT — see [LICENSE](LICENSE). Built with open-source models; runs entirely offline.
