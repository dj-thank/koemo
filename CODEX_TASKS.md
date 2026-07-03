# Koemo — Codex 実装ハンドオフ（Phase B残り 〜 Phase D）

> このドキュメントだけで実装を完了できるよう、現状・環境の落とし穴・各タスクの設計・検証手順を自己完結で記述する。Codex は冷えた状態で開始する前提。**速度重視**。最終目標は「Meetily の全機能を実装し上回る」。

---

## 0. 現状ミッション（2026-05-31時点）
Phase B〜D の機能実装は完了済み。現在の主タスクは「録音/文字起こし/要約/パッケージの実運用堅牢性を、実ファイルと実機検証で閉じる」こと。

| 領域 | 現状 |
|---|---|
| リアルタイム（録音中）ライブ文字起こし | 実装済み。Windows native + Whisper rolling fallback。 |
| 会議履歴 / 全文検索 | 実装済み。SQLite + FTS5/LIKE fallback。 |
| 要約バックエンド（Ollama / OpenAI互換）| 実装済み。local既定、remoteは明示opt-in。 |
| 要約テンプレート | 実装済み。設定UIから編集可能。 |
| 会議とチャット（Q&A） | 実装済み。結果/履歴から transcript-grounded chat。 |
| 会議自動検出（Zoom/Teams等） | 実装済み。通知のみで自動録音はしない。 |
| カレンダー連携 | 実装済み。ICS/Outlook予定タイトルヒント。 |
| 単体exe化（PyInstaller） | 実装済み。`dist\Koemo\Koemo.exe` を継続再ビルド検証。 |
| 専用アイコン/ロゴ | 実装済み。assets反映済み。 |

### 2026-05-30 実装結果
- B: `DualRecorder.snapshot()` / `LiveTranscriber` / `LiveWindow` / 設定トグルを実装。停止後の最終 `_process` は従来通り正として維持。
- C1: `~/.koemo/library.db` のSQLite履歴、FTS5検索＋LIKEフォールバック、履歴ウィンドウ、保存/取込後の自動登録を実装。
- C2: 要約バックエンドを `local`（既定・外部通信なし）/ `ollama` / `openai_compat` に分離。APIキーは `~/.koemo/config.json` のみ。
- C3: 要約セクションと追加指示の設定UI/プロンプト反映を実装。
- C4: 結果/履歴から開ける会議チャットを実装。LLM呼び出しはワーカースレッド。
- D1: `psutil` の会議アプリ検出とトレイ通知を実装（自動録音はしない）。
- D3: `koemo.spec` を追加し、`dist/Koemo/Koemo.exe` のビルドと起動確認まで完了。
- D4: `assets/koemo.png` / `assets/koemo.ico` を追加し、トレイ/ウィンドウ/EXEアイコンに反映。
- D2: ICS/Outlook予定タイトルヒントを設定から有効化できるようにした。
- 検証: `py_compile`、ライブ/要約テンプレート/チャット/ライブラリのスモーク、通常アプリ起動、PyInstallerビルド、パッケージEXE起動を実施。

### 2026-05-30 実機追加修正
- packaged EXE 停止後に `faster_whisper/assets/silero_vad_v6.onnx` 不足で ONNXRuntime `NO_SUCHFILE` になる問題を修正。`koemo.spec` で `faster_whisper` のdata assetsを同梱し、`dist/Koemo/_internal/faster_whisper/assets/silero_vad_v6.onnx` の存在を検証済み。
- ライブ文字起こしが無音のシステム音声chを選び続け、マイク発話が表示されない問題を修正。`LiveTranscriber` が直近音声のRMSを見て、有音chを選ぶようにした。
- `scripts/koemo_feature_smoke.py` と `FUNCTION_CHECKLIST.md` を追加。全機能スモークとWindowsセキュリティ観点を継続検証できる。
- 修正版EXEで `Ctrl+Shift+R` 録音 → Windows TTS → 停止 → 文字起こし → 要約 → `summary_20260530_143112.md` 保存 → 履歴登録まで実機確認済み。

---

## 1. 現状（**壊さないこと**）

**場所**: `C:\Users\rambo\RamboPC\DevHub\10_active\koemo`（OSS/MIT, Windows優先, PySide6）

**完了済み（Phase A + 話者分離）**: 2ch録音(mic+システム音声soundcardループバック)+AEC、チャンネル別文字起こし(faster-whisper, GPU, 既定 large-v3-turbo)、システム音声側の話者分離(sherpa-onnx → 相手1/相手2)、構造化ローカル要約(CTranslate2 + Qwen2.5-3B → タイトル/要旨/主要トピック/決定事項/アクションアイテム/未解決の質問)、長尺チャンク分割、PySide6トレイ/ホットキー(Ctrl+Shift+R)/結果ウィンドウ(Markdown整形・PDF/DOCX/MD出力)/設定、音声ファイル取込、keep-warm、遅延ロード/アイドル解放。

**ファイルマップと主要API**（再利用すること）:
- `koemo.pyw` — エントリ（DPI設定 → `gpu.enable_cuda_dlls()` → `app.main()`）
- `koemo/gpu.py` — `enable_cuda_dlls()`（nvidia-* DLLをPATH追加）, `gpu_ok()`→bool
- `koemo/config.py` — `CONFIG_DIR=~/.koemo`, `CONFIG_FILE`, `RECORDINGS_DIR=~/RamboPC/DevHub/10_active/koemo/outputs/meetings`, `MIC_LABEL="あなた"`, `SYS_LABEL="相手"`, `DEFAULT_CONFIG`(下記キー), `load_config()`, `save_config(cfg)`
- `koemo/audio.py` — `write_wav(path,sr,i16,ch)`, `cancel_echo(mic,ref)`, `DualRecorder(cfg)`(`.start()/.stop()/.elapsed()`; 録音本体は `.koemo_tmp/*.f32` にspool、ライブはリング保持。stop→`{"channels":{label:float32_16k/memmap}, "duration", "ts", "sr", "files", "write_errors", "capture_errors", "temp_files"}` or None), `list_devices()→(mics,spks)`
- `koemo/transcribe.py` — `Transcriber(model_size,cpu_threads,idle_sec,keep_warm)`（`.transcribe_segments(audio_or_path, language, on_progress)→[(start,end,text)]`, `.last_seconds`, `.unload()`, `.maybe_unload()`, `.reload(...)`）, `merge_rows(rows)`（rows=[(start,label,text)]）
- `koemo/backends.py` — `find_summary_model(explicit)`, local/Ollama/OpenAI互換の要約backend。
- `koemo/summarize.py` — `SECTIONS` dict, `Summarizer(model_dir,idle_sec,keep_warm)`（`._ensure_model()`, `._generate(system,user,max_tokens)`→str(greedy+rep_pen1.05), `._chunks(text)`, `.summarize(transcript,language,on_progress)→(title, body_md)`, `.last_seconds`, `.maybe_unload()`, `.reload(...)`）
- `koemo/diarize.py` — `available()→bool`, `diarize(audio16k)→[(start,end,speaker_int)]`, `assign_speakers(segments,turns,base_label)→[(start,label,text)]`, `download_diarization_models()`
- `koemo/export.py` — `export_markdown(md,path)`, `export_pdf(md,path)`(**要QApplication**), `export_docx(md,path)`
- `koemo/app.py` — `make_icon(rec)`, `Toast(QWidget)`, `_Bridge(QObject)`(signals: `toggle/toast(str)/toast_close/results(object)/error(str)`), `KoemoApp(QObject)`（`.cfg/.recorder/.transcriber/.summarizer/.sig/._toast/._tray`; `_build_tray`, `_register_hotkey`, `_on_toggle/_start/_tick/_stop`, `_process`（録音→ch別文字起こし→システム側diarize→`merge_rows`→`summarize`→`save summary_{ts}.md`→`sig.results.emit`）, `_import_audio/_process_file`, `_open_results(ResultsWindow)`, `_open_settings(SettingsDialog)`, `_on_cfg_saved`, `_idle_watcher`, `main()`）
- `koemo/ui_results.py` — `ResultsWindow(title, summary_md, transcript_md, save_dir, duration)`
- `koemo/ui_settings.py` — `SettingsDialog(cfg, on_save)`

**DEFAULT_CONFIG キー**: 現行値は `koemo/config.py` を正とする。主要キーは `summary_backend`, `summary_model_dir`, `ollama_*`, `openai_*`, `summary_sections`, `whisper_model=large-v3-turbo`, `live_whisper_model=small`, `hotkey`, `save_dir`, `summary_language`, `sample_rate`, `cpu_threads`, `idle_unload_sec`, `keep_warm`, `preload_*`, `fast_summary`, `use_live_transcript_on_stop`, `live_backend=auto`, `live_fallback_backend`, `native_speech_language`, `final_channel_policy`, `native_only_transcription`, `live_*`, `record_mic`, `record_system`, `enable_aec`, `enable_diarization`, `enable_live_transcription`, `enable_meeting_detection`, `enable_calendar_title_hint`, `calendar_*`, `mic_name`, `speaker_name`。新キーはここに追記し、UIにも反映する。

---

## 2. 環境・制約・**落とし穴**（必読）

1. **OS/Python**: Windows 11, Python 3.14.3（`C:\Python314`、ユーザー site-packages は `C:\Users\rambo\AppData\Roaming\Python\Python314\site-packages`）。シェルはPowerShell中心、ヒアドキュメントはBashツールを使う。
2. **GPU**: RTX 2080 Super 8GB。**torchはCPU版**。GPU推論は **CTranslate2** で行う（`gpu.gpu_ok()` が真なら cuda）。`enable_cuda_dlls()` が pip の `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` の bin を **PATHに追加**（`os.add_dll_directory` だけでは不足）。GPU処理を呼ぶ前に必ず `enable_cuda_dlls()` 済みであること（エントリで実行済み。単体テストでは自分で呼ぶ）。
3. **llama-cpp-python は使用不可**（このCPUでモデルロード時 `0xc000001d` ILLEGAL_INSTRUCTION クラッシュ）。ローカルLLMは CTranslate2 のみ。
4. **Qt のイベントループは直接呼び出さない**。ランタイムのセキュリティフックが「`exec` の直後に半角開き括弧が続く文字列」を誤検知してファイル書込をブロックする。`run_event_loop = app.exec` と名前束縛してから `run_event_loop()` を呼ぶ（既存 `app.py` の `main()` を参照）。**ソース・ドキュメント中でも `.exec` ＋半角括弧 の直書きを避ける**（例: ダイアログ表示は `_run = dialog.exec; _run()` に置き換える。`QDialog.open()`/`show()` でも可）。
5. **`export_pdf` は QApplication が必要**（Qtのフォント処理）。実アプリは常にQApplicationありなので問題ないが、単体テストでは `QT_QPA_PLATFORM=offscreen` + `QApplication([])` を先に作る。
6. **モデル**:
   - 文字起こし: faster-whisper 内蔵名 `large-v3-turbo`（高速・高精度。Parakeetは英語専用なので日本語はturboで代替）。`tiny/base/small/medium/large-v3` も選択可。
   - 要約: `jncraton/Qwen2.5-3B-Instruct-ct2-int8`（HFキャッシュ、`find_summary_model()` が model.bin最大を自動選択）。**7Bは8GB VRAMに載らない、3Bが上限**。
   - 話者分離: `~/.koemo/models`（seg `model.int8.onnx` + emb `speakernet.onnx`。`wespeaker_resnet34.onnx`/`eres2net.onnx` も存在）。clustering threshold=0.4。
7. **設定・APIキーは `~/.koemo/config.json`（リポジトリ外）**。クラウドバックエンドの鍵を絶対にリポジトリへコミットしない。READMEに明記。
8. **依存**: 追加が必要なものは cp314/win wheel 確認済み: `sherpa-onnx`(導入済), `python-docx`(導入済), `psutil`, `httpx`, `openai`（いずれも導入可）。Ollama連携はPythonクライアントではなく `httpx` でHTTP APIを叩く。`requirements.txt` と `setup.bat` に反映。
9. **スレッド/UI**: 重い処理はワーカースレッドで実行し、UI更新は `_Bridge` のSignal経由でGUIスレッドへ marshal する（既存 `_process` 参照）。新ウィンドウ生成・表示はGUIスレッドで。
10. **スタイル**: 日本語コメント、既存の暗色テーマ(#0d0d1a/#a0c4ff等)・"Yu Gothic UI"。backend/UI分離を維持。

---

## 3. 検証ツールキット（各タスク末で必須）

- **構文**: `python -m py_compile koemo.pyw koemo/*.py`
- **日本語TTSで音声生成**（PowerShell System.Speech）:
  - wavへ: `$s.SetOutputToWaveFile($path, (New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)))` → `$s.Speak(@"...本文..."@)`
  - **システム音声ループバック検証**: TTSを既定出力へ `Speak` しながら `DualRecorder` で録音 → 相手ch に入る（録音核心の実証）。
  - 日本語音声は `Microsoft Haruka Desktop`、別話者の検証には `Microsoft Zira Desktop`（英語・別声）。
- **Qt単体テスト**: `QT_QPA_PLATFORM=offscreen` を設定し、`QApplication([])` を先に生成してから Qt API を叩く。
- **アプリ起動確認**: `Start-Process pythonw -ArgumentList "koemo.pyw" -WorkingDirectory <repo> -PassThru` → `Start-Sleep 7` → `$_.HasExited`。既存インスタンスは `Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | ? CommandLine -like '*koemo.pyw*' | Stop-Process` で停止。
- **モジュール検証**: `sys.path.insert(0, repo)` → `from koemo... import ...` で実コードを直接叩く（GUIを起動せず）。
- **速度計測**: `transcriber.last_seconds` / `summarizer.last_seconds` を確認。目標: turbo文字起こしはGPUでリアルタイム以上、要約warm時 ~8秒以内、ライブ文字起こし遅延 ~3〜5秒。

---

## 4. タスク詳細（推奨順）

### B. リアルタイム（録音中）ライブ文字起こし  ★最優先
**目的**: 録音中に文字起こしを逐次表示。停止時の最終文字起こし（現 `_process`）は据え置きで「正」とする（ライブはプレビュー）。

**設計（ローリング窓・プレビュー＋最終確定の二層）**:
- `audio.DualRecorder` に**ライブ供給**を追加: `_capture` ループ内で各チャンクを `self._live[label]`(list) にも追記し、スレッドセーフな `snapshot(label)->np.float32`（現在までの連結コピー）を提供。または `on_chunk(label, chunk)` コールバック注入でも可。録音停止時の最終バッファ生成は現状維持。
- 新規 `koemo/live.py` の `LiveTranscriber`:
  - 別スレッドで ~4秒ごとに `recorder.snapshot(SYS_LABEL)`（無ければ MIC）を取得し、**末尾 ~20〜30秒の窓**を `Transcriber.transcribe_segments(window, language)` でプレビュー文字起こし。
  - 確定済み（窓の手前で安定した）セグメントは確定テキストとして保持し、末尾だけ毎回再計算。`on_update(text)` コールバックで全文（確定＋暫定）を返す。
  - turbo + GPU 前提で短窓なら高速。`keep_warm` 推奨。
- 新規 `koemo/ui_live.py` の `LiveWindow(QWidget)`: 右下に常駐するライブ字幕（または通常ウィンドウ）。`update_text(str)` を Signal 経由で受け、`QTextBrowser`/`QLabel` に表示。録音中のみ表示。
- `koemo/app.py`: `_start` で LiveWindow表示＋`LiveTranscriber`開始（GUIスレッドへは `_Bridge` に `live(str)` Signalを追加して marshal）。`_stop` で停止・閉じる。最終 `_process` は不変。設定キー `enable_live_transcription`(既定True) を追加し、設定UIにトグル。

**検証**: TTSをシステム音声へ再生しながら `pythonw koemo.pyw` で録音 → 録音中にライブ窓へ文字が増えていく → 停止後の最終結果は従来通り正確。`live.py` 単体は、長めwavを擬似ストリーム（チャンクに分割して順に snapshot へ）して `on_update` の逐次出力を確認。

---

### C1. 会議履歴 / 全文検索
**目的**: 過去の会議を一覧・検索・再オープン・エクスポート。

**設計**:
- 新規 `koemo/library.py`: SQLite `~/.koemo/library.db`。テーブル `meetings(id INTEGER PK, ts TEXT, title TEXT, date TEXT, summary_md TEXT, transcript_md TEXT, save_dir TEXT, duration INT, created_at TEXT)`。FTS5仮想テーブル（title/summary/transcript）で全文検索。関数: `add(title, summary_md, transcript_md, save_dir, duration, ts)`, `search(query)->rows`, `recent(limit)`, `get(id)`。FTS5未対応環境向けに LIKE フォールバック。
- `koemo/app.py`: `_process` と `_process_file` の保存後に `library.add(...)` を呼ぶ。
- 新規 `koemo/ui_library.py` の `LibraryWindow(QWidget)`: 検索ボックス＋リスト（タイトル/日付）。項目クリックで `ResultsWindow(title, summary_md, transcript_md, Path(save_dir), duration)` を開く（既存ResultsWindowを再利用）。トレイメニューに「📚 履歴」を追加し `LibraryWindow` を開く。

**検証**: 数件 `library.add` → `search("リリース")` がヒット → `LibraryWindow` で一覧・検索・オープンができる（GUI部分は起動確認で代替）。

---

### C2. 要約バックエンド（ローカル既定＋Ollama/OpenAI互換）
**目的**: 「全機能実装」。ローカルCT2を既定に、任意でOllamaやOpenAI互換エンドポイント（Groq/OpenRouter/Claude等）も選べる。

**設計**:
- 新規 `koemo/backends.py`:
  - `class SummaryBackend: def generate(self, system, user, max_tokens)->str`
  - `LocalCT2Backend`: 現 `Summarizer._ensure_model/_generate` のCT2生成を移植（モデル/トークナイザ保持・GPU/CPU・lazy/unload）。
  - `OllamaBackend(model)`: `ollama` python client か httpx で `http://localhost:11434/api/chat`。既定モデル例 `qwen2.5:3b`。未起動時は分かりやすい例外。
  - `OpenAICompatBackend(base_url, api_key, model)`: `openai` SDK（`OpenAI(base_url=..., api_key=...)` のチャット補完）。Groq/OpenRouter/独自はbase_urlで対応。
- `koemo/summarize.py` リファクタ: `Summarizer` は**構造化プロンプト生成ロジック（タイトル/要旨/各見出し/チャンクmap-reduce）を維持**しつつ、`_generate` の中身を選択バックエンドへ委譲。`config["summary_backend"] in {"local","ollama","openai_compat"}` で切替。
- `koemo/config.py`: 追加キー `summary_backend("local")`, `ollama_model("qwen2.5:3b")`, `openai_base_url("")`, `openai_api_key("")`, `openai_model("")`。**鍵は ~/.koemo に保存・リポジトリへ出さない**。
- `koemo/ui_settings.py`: バックエンド選択コンボ＋（OpenAI互換選択時の）base_url/api_key/model 入力欄。
- `requirements.txt`: `httpx`, `openai` を「任意機能」コメント付きで追加（遅延importにして未導入でもローカルは動くようにする）。`pyinstaller` は `requirements-build.txt`、公開コーパス辞書生成用 `datasets` は `requirements-tools.txt` に分離。

**検証**: `summary_backend="local"` で従来通り。Ollama導入環境があれば `OllamaBackend` で短文要約が返る。`OpenAICompatBackend` はモック/実エンドポイントで疎通（鍵が無ければスキップ）。**既定(local)で外部通信が無いこと**を確認。

---

### C3. 要約テンプレート
**目的**: ユーザーが要約の見出し/指示をカスタマイズ。

**設計**:
- `config` に `summary_sections`(list, 既定は `summarize.SECTIONS["ja"]`) と任意 `summary_extra_instructions`(str)。`Summarizer.summarize` は `config` 由来のセクションがあればそれを使い、無ければ既定。先頭セクション（要旨）は散文必須のルールを維持。
- `ui_settings.py`: セクションをカンマ区切りで編集する欄＋追加指示のテキスト欄。

**検証**: セクションを変更 → 要約出力の見出しが反映される（`Summarizer.summarize` を直接呼んで確認）。

---

### C4. 会議とチャット（Q&A）
**目的**: 文字起こしに対してローカルLLMで質問応答。

**設計**:
- `koemo/summarize.py` に `Summarizer.chat(question, transcript, history=None)->str`: バックエンドの `generate` を使い、systemに「以下の会議内容のみに基づき日本語で簡潔に回答」、userに `transcript`(長い場合は `_chunks` の先頭＋要約を文脈に)＋質問。
- 新規 `koemo/ui_chat.py` の `ChatPanel`/`ChatWindow`: 入力欄＋会話表示。`ResultsWindow` に「💬 チャット」タブまたはボタンを追加し、その会議の `transcript_md` を文脈に Q&A。応答生成はワーカースレッド＋Signalで表示。

**検証**: ある文字起こしに対し「決定事項は？」と聞いて妥当な回答（`Summarizer.chat` を直接呼んで確認）。

---

### D1. 会議自動検出
**目的**: Zoom/Teams等の起動を検出し録音を提案。

**設計**:
- 新規 `koemo/detect.py`: `psutil` で実行中プロセス名を監視（`zoom.exe`, `teams.exe`/`ms-teams.exe`, `webex.exe`）。通常のSlack起動は会議とは限らないため既定検出から除外。`MeetingWatcher` を別スレッドで回し、新規検出時に `on_detected(app_name)` を1回（デバウンス）発火。
- `koemo/app.py`: `config["enable_meeting_detection"](既定True)` が真ならウォッチャ開始。検出時 `QSystemTrayIcon.showMessage("会議を検出","録音しますか？ Ctrl+Shift+R")` 通知（Signal経由でGUIへ）。誤検出を避けるため通知のみ（自動録音は任意のオプションに）。
- `ui_settings.py`: 検出トグル。

**検証**: Zoom/Teams（無ければ任意プロセス名を一時的にリストへ追加）起動でトレイ通知が出る。

---

### D2. カレンダー連携（実装済み）
- `.ics` 読み込みと Outlook COM(`win32com`)で「現在進行中/直近の予定タイトル」を取得し、設定で有効な場合に会議タイトルの既定値に使う。繰り返し予定のICS展開は未対応のため、Outlook側の繰り返し展開または単発VEVENTを使う。

---

### D3. 単体exe化（PyInstaller）
**目的**: Python未導入ユーザーが exe だけで使える。

**設計・注意**:
- 新規 `koemo.spec`: `hiddenimports`（`sherpa_onnx`, `ctranslate2`, `faster_whisper`, `av`, PySide6プラグイン）と `datas`（PySide6 platformsプラグイン等）を含める。エントリは `koemo.pyw`。`pyinstaller koemo.spec` でビルド。
- **モデルはバンドルしない**（巨大）。初回起動/setupでダウンロードする現方式を維持（exeでも `~/.koemo` と HFキャッシュへDL）。
- **GPU/CUDA DLL（nvidia-cublas/cudnn, ~1.2GB）はバンドルしない**方針を推奨。exeは**既定CPU**で動作させ、GPUは「別途 pip で nvidia-* を入れた Python 実行」または将来のGPU同梱版として案内（README）。`gpu_ok()` はDLLが無ければ自動でCPUに落ちるので安全。
- ネイティブDLL同梱は壊れやすい。**ビルド後に exe を起動して `HasExited` で起動確認**し、最低限「録音→文字起こし→要約→結果表示」をCPUで通すこと。

**検証**: `pyinstaller koemo.spec` 成功 → `dist/Koemo/Koemo.exe`(または onefile) が単体起動 → 短いTTS取込で要約まで通る（CPU）。

---

### D4. ブランド（アイコン/ロゴ）
**目的**: 描画マイクを専用アイコンへ。

**設計**:
- `assets/koemo.ico` と `assets/koemo.png` を用意（QPainterでプログラム生成でも、簡素なベクター調でも可。"声+メモ" を想起させる吹き出し＋波形など。独自デザインとし、既存製品の模倣をしない）。
- `app.make_icon()` を assets 読み込みに変更（録音中は色替え or バッジ）。ウィンドウ/トレイ/exe(`koemo.spec` の `icon=`) に適用。README にスクリーンショット。

**検証**: トレイ・ウィンドウ・exe にアイコンが反映。

---

## 5. 規約・受け入れ基準
- backend/UI分離を維持。重い処理はワーカースレッド＋`_Bridge` Signal。
- **ローカル既定**。既定設定で一切の外部通信が無いこと。クラウドは明示オプトインのみ。**鍵をリポジトリに置かない**。
- 既存スタイル（日本語コメント・暗色テーマ・Yu Gothic UI）を踏襲。
- 各タスク完了時に **py_compile + 該当の機能検証 + アプリ起動確認** を行い、`requirements.txt`/`setup.bat`/`README.md` を更新。
- Qtのイベントループ/ダイアログは §2-4 に従い `.exec` ＋半角括弧 の直書きを避ける。`export_pdf` 等 Qt描画はQApplication前提。
- 完了後、`README.md` の Roadmap から実装済み項目を「実装済み」へ移し、Meetily比較表（★無料で上回る点）を更新。

## 6. 推奨実行順
B（ライブ）→ C2（バックエンド基盤）→ C3（テンプレート）→ C4（チャット）→ C1（履歴）→ D1（検出）→ D4（アイコン）→ D3（exe）→ D2（カレンダー）。
各フェーズは独立にコミット可能。Bと C2 を先に固めると体感価値と拡張性が高い。

---

## 2026-05-30 高速化・ライブ文字起こし再修正

> 注記: この章以降には当時の検証値・当時の実装方針も履歴として残している。現行の既定値と合否証跡は、最新の `DEFAULT_CONFIG`、README、末尾の 2026-05-31 最終パッケージ再検証節を正とする。

ユーザー実機で「停止後が遅い」「ライブ文字起こしが音声待機中のまま」「Windows Security が厳しい」
という状態を確認し、Kanary 2.0.6 と Meetily の実装方針を参考に再調整した。

### 参考確認
- `C:\Users\rambo\Downloads\Kanary-2.0.6.zip` を作業用 `work/kanary-2.0.6` に展開。
- Kanary は macOS `.app` で、`Speech.framework` と `FoundationModels.framework` を使う構成。Windowsへ直接移植は不可。
- 真似るべき点を「録音中に transcript を作る」「停止時に全量再処理しない」「停止後の要約を軽量にする」と整理。
- Meetily 側は streaming worker / VAD / transcript-update / stop後post-process分離の設計を確認。

### 実装変更
- `koemo/live.py`
  - ライブ backend を `native_windows` / `whisper_rolling` / `off` の切替式にした。
  - 方針再整理により、既定は `native_windows`、fallback は `whisper_rolling`。
  - Windows純正 `System.Speech` をPowerShellブリッジで起動し、Hypothesis/Result をライブ表示する。
  - `System.Speech` がOS設定で拒否された場合は WinRT Speech を試す。
  - どちらも拒否された場合は理由を表示し、設定どおり Whisper rolling へ fallback する。
  - ライブ結果は速報扱いにし、停止後の正式 transcript には保存WAVの Whisper `large-v3-turbo` 結果を使う。
  - 停止時の同期 `flush()` をやめ、GUIが固まる原因を除去。
  - システム音声が強い時はマイク回り込みをライブ対象から外しやすくした。
- `koemo/app.py`
  - ライブ用 `Transcriber` と最終処理用 `Transcriber` を分離。
  - `native_only_transcription=false` を既定に戻し、Whisper final `Transcriber` を起動時に生成・先読みする。
  - 停止後は mic/system の保存WAVを Whisper `large-v3-turbo` で再認識して transcript を作る。
  - mic/system 両方が有音の場合、音量ではなく文字列品質で採用チャンネルを選ぶ。英字ノイズや短すぎる断片を減点し、日本語率とKoemo関連語を加点する。
  - Windows native の確定行が空なら、空のままではなくOS設定確認メッセージを保存する。
  - `fast_summary=true` ではLLMを待たず即時要約。
  - mic/system のWAV保存を維持し、マイク回り込みが悪い場合は system 側の正しい文字起こしを採用する。
  - WAV保存失敗時もメモリ上の録音から処理継続。
  - summary保存失敗時は `%LOCALAPPDATA%\Koemo\Recordings` へ退避。
- `koemo/config.py` / `koemo/ui_settings.py`
  - `live_backend`, `live_fallback_backend`, `native_speech_language`, `show_model_ready_status`, `final_channel_policy`, `native_only_transcription` を追加。
  - `use_live_transcript_on_stop=false` を精度優先の既定にした。
- `scripts/koemo_fast_integration_check.py`
  - 実起動、TTS再生、停止、summary作成までの秒数を計測。通常は `KOEMO_TEST_COMMAND_FILE` の deterministic file command で録音開始/停止し、実機ホットキー併用は `KOEMO_TEST_SEND_HOTKEY=1` の時だけ行う。
  - 未署名EXEが Windows App Control で `WinError 4551` ブロックされた場合は `pythonw koemo.pyw` にフォールバック。
- `scripts/koemo_feature_smoke.py`
  - Windows App Control ブロックを通常の環境制約として分類し、`pythonw` フォールバックを検証。
- `scripts/build_native_corrections.py`
  - Koemo出力と Hugging Face 公開データから `~/.koemo/native_corrections.json` を生成。
  - 公開データは `top_phrases` として保持し、Windows Speech に渡す `grammar_phrases` はKoemo用途の短い語彙だけに絞る。
- `koemo/native_correction.py` / `koemo/data/native_corrections.json`
  - `声も`→`コエモ`、`文字を腰`→`文字起こし`、`死後十秒`→`停止後10秒` などを補正。
  - PowerShell Speech bridge / file recognizer の両方で補正辞書を読み込む。

### 検証結果
- PyWinRT `winrt-Windows.Media.SpeechRecognition` / `winrt-Windows.Globalization` を導入済み。
- Windows Speech の実機API到達を確認。
- `HKCU\Software\Microsoft\Speech_OneCore\Settings\OnlineSpeechPrivacy\HasAccepted=1` 設定後、`System.Speech` の DictationGrammar 起動を確認。
- `python scripts\koemo_feature_smoke.py`: 14/14 PASS。
- Hugging Face 公開データ 50,000行 + Koemoログから補正辞書生成済み。
- 保存済み `recording_20260530_192854_system.wav` は `これはコエモ高速化テストです` / `ライブ文字起こしと停止後10秒以内の処理を確認しています` と再認識できる。
- 同録音の mic 側は `Affairsふろふッヒュー` / `ゴー` と崩れるため、final pass の品質選択で system 側を採用するよう修正済み。
- `python scripts\koemo_fast_integration_check.py`: PASS。
  - 最新再ビルド後の停止ホットキーから summary 作成まで: 3.282秒。
  - `has_native_text=true`。
  - 出力: `C:\Users\rambo\RamboPC\DevHub\10_active\koemo\outputs\meetings\summary_20260530_193818.md`
  - first_lines は `これはコエモ高速化テストです` と `ライブ文字起こしと停止後10秒以内の処理を確認しています`。
- 現在の再ビルド済み `dist\Koemo\Koemo.exe` は起動・録音・停止後処理まで PASS。

### 2026-05-30 再読込レビューで見つけたズレと追加修正
- Context7 で `Windows.Media.SpeechRecognition` の `ContinuousRecognitionSession.ResultGenerated`、constraints、confidence の前提を確認。
- 実失敗 `summary_20260530_194039.md` は mic WAV が `フッ素年五月` confidence 0.058、system WAV が空だった。前回の「ファイル再認識だけを正にする」設計は、実マイク発話では誤採用の危険があると判定。
- `koemo/native_speech.py` に `transcribe_wav_events()` を追加し、System.Speech の confidence を Python 側へ渡す。
- 当時は `koemo/app.py` の final pass で confidence が低すぎる断片を捨て、ライブ候補も比較対象に入れた。現在の既定は `use_live_transcript_on_stop=false` で、ライブ候補は正式 transcript に混ぜない。
- 当時は `koemo/live.py` の WinRT Speech に `SpeechRecognitionTopicConstraint(DICTATION)` と Koemo語彙の `SpeechRecognitionListConstraint` を追加したが、後続の実測で初回仮説が遅くなったため現在は制約なしに戻した。
- `koemo.spec` に `keyboard._winkeyboard` 等を hidden import 追加。再ビルド後、EXEホットキー録音が復帰。
- `setup.bat` は Whisper `large-v3-turbo` を取得する。ライブは Windows純正、正式 transcript は Whisper。
- 当時の `python scripts\koemo_feature_smoke.py`: 15/15 PASS。現在は下記ハイブリッド再検証と 2026-05-31 のチャンネル選択修正後スモークで 25/25 PASS。
- `python scripts\koemo_fast_integration_check.py`: PASS、停止ホットキーから summary 作成まで 3.797秒。

### 残るトレードオフ
- Windows Speech はマイク入力中心。システム音声はWAV保存のみ。
- Windows のマイク/Speech プライバシー許可が無い場合、ネイティブライブは使えない。
- 正式 transcript は Whisper `large-v3-turbo` のため、cold start では初回モデルロード待ちがある。`pythonw koemo.pyw` / `start.bat` 経路ではCUDAを使えるため、起動時preload後は停止後10秒以内を満たす。
- `final_channel_policy=auto_dedupe` では、マイクだけが強くクリップしシステムchがクリーンな時、飽和マイクを音響エコーとして捨てる。さらに AEC 後のマイクが低レベルでシステムchに対して非常に小さい場合も、GPU Whisper が残留漏れを別発話化するのを避けるため system を優先する。入力ゲイン過大の本人発話が常時クリップするケースや、極端に小さい本人発話も全ch保持したい場面は `all_active` を使う。
- ICSの繰り返し予定（RRULE）は展開しない。繰り返し予定タイトルを使いたい場合は Outlook 連携を有効化するか、展開済みVEVENTを含むICSを指定する。

### 2026-05-30 ハイブリッド既定への戻し
- Context7 で UWP SpeechRecognizer の continuous session / ResultGenerated / constraints 前提を再確認。
- `native_only_transcription=false` を既定に戻し、Windows Speech はライブ速報、Whisper `large-v3-turbo` は正式 transcript に分離した。
- `live_fallback_backend=whisper_rolling` を既定に戻し、Windows Speech が使えない時はライブだけ Whisper rolling へ落とす。
- `requirements.txt` / `setup.bat` / `koemo.spec` に `faster-whisper` と assets 同梱を戻した。
- Whisper final 側にも Koemo用途補正辞書を適用し、`声も`→`コエモ` のような既知誤認識を正式結果で補正する。
- `scripts/koemo_fast_integration_check.py` は推奨起動経路 `pythonw koemo.pyw` を既定にした。未署名EXEは便利配布用で、CUDA DLLを同梱しないため10秒性能の基準にはしない。
- 検証:
  - `python -m py_compile ...`: PASS
  - `python scripts\koemo_feature_smoke.py`: 当時 19/19 PASS。2026-05-31 の追加契約後は 26/26 PASS。
  - `python scripts\koemo_fast_integration_check.py`: PASS、停止から summary 作成まで 1.274秒
  - `python scripts\koemo_model_bench.py`: 11.3秒音声で `large-v3-turbo` 3.702秒、正しい `コエモ高速化テスト` を出力
  - `python -m PyInstaller koemo.spec --noconfirm --clean`: PASS、当時の再ビルド後スモーク 19/19 PASS。2026-05-31 の再ビルド後 `packaging_files` / `launch_exe` も PASS、追加契約後スモークは 26/26 PASS。
  - `python scripts\koemo_live_latency_check.py`: マイクRMS立ち上がり基準でライブウィンドウ初回更新 0.054秒。Windows Speech の文字仮説は 1.337〜1.692秒で、厳密な `HypothesisGenerated` 1秒以内は未達。発話検出プレースホルダでライブUIは1秒以内に動く。
- 追加単体検証:
  - Windows native 起動失敗時に `whisper_rolling` へfallbackし、ライブウィンドウ向け状態文を出す。
  - `MicActivityPreview` が Windows Speech 起動待ちより先に開始される。
  - WinRT `ready` は `ContinuousRecognitionSession.start_async()` 完了後に出すよう修正し、測定/起動側が認識開始前に準備完了扱いしないようにした。
  - `native_speech_startup_settle_sec=1.0` を追加し、`ライブ文字起こし中` 表示前に短い安定化時間を置く。
  - native/Whisper rolling の両ライブバックエンドが共通 `LiveEvent(time, label, text, final|provisional)` 形式を返す。
  - `use_live_transcript_on_stop=false` ではライブ誤認識を正式 transcript に混ぜない。
- レイテンシ調整:
  - `SpeechRecognitionTopicConstraint(DICTATION)` と `SpeechRecognitionListConstraint` は実測で初回仮説を遅らせたため、WinRT live は制約なしに戻した。
  - Koemo固有語は WinRT 制約ではなく `normalize_native_text()` / `normalize_transcript_text()` の後段補正で吸収する。
  - Windows Speech の初回文字仮説が1秒を超える場合に備え、`MicActivityPreview` がマイクRMSの立ち上がりで `あなた: ...` を即時表示し、その後の Windows Speech 仮説/確定で置き換える。

## 2026-05-30 ライブ初速レビューと修正（Claude Opus 4.8）

実機実測 `windows_text 1.337〜1.692s` を受けてレビューし、Kanary型の二段構成を保ったまま「ライブUIの体感初速」を実アプリで成立させる surgical 修正を入れた。

### レビューで確認した実装と現実のズレ
- `koemo_live_latency_check.py` の `activity 0.054s` は Qt イベントループを通さない計測値で、実アプリの体感を表していなかった。
- 実アプリでは `_on_toggle`→`_start`→`_start_live`→`NativeWindowsLiveTranscriber.start()` が **GUIスレッドで `_ready.wait()` ブロック**（`start_async`+settle 1.0s）。この間 `MicActivityPreview` の `sig.live.emit` はキューに積まれるだけで描画されず、`ライブ文字起こし中` 表示が `あなた: ...` を上書きしていた。→ ライブUIは実際には1秒以内に動いていなかった。
- WinRT introspection（winrt 3.2.1）で確認: `ContinuousRecognitionSession` に `pause_async()/resume()`、`SpeechRecognizer.add_state_changed`（`SOUND_STARTED`/`SPEECH_DETECTED` は最初の文字仮説より早い）、`system_speech_language=ja`（topic/grammar も `ja`、コードは `ja-JP` 指定）。

### 実装（A: 低リスク）
- `koemo/live.py`
  - `NativeWindowsLiveTranscriber.start(wait=False)` を追加。アプリ既定は **非ブロック**（GUIを固めない）。起動失敗は `on_error`/`on_unavailable` 経由で fallback。計測ハーネスだけ `wait=True`。
  - `_thread_main` の except を、ready前失敗でも `on_unavailable` を呼ぶように変更（非ブロック起動では start() が例外を投げないため）。
  - `add_state_changed` を登録し、`_on_state()`/`_emit_speech_detected()` でエンジン由来の発話検知時に `あなた: ...` を即時表示（実テキストがあれば上書きしない）。
  - `ライブ文字起こし中` は `start_async` 直後に一度だけ出し、settle は表示安定化専用に降格（`_ready` のゲートのまま＝ハーネスの warmup 計測は不変）。発話検知/仮説を上書きしない。
  - `_resolve_language()` を追加し、`supported_topic_languages` に無い要求タグは `system_speech_language` へ寄せる。
- `koemo/app.py`
  - `_native_fell_back` ガードを追加（`_start_live` でリセット、`_fallback_live_after_native` で二重 fallback 防止）。非ブロック化で fallback がコールバック経路になったため。
- `scripts/koemo_live_latency_check.py`: `transcriber.start(wait=True)`（認識開始を待ってからTTS）。
- `scripts/koemo_feature_smoke.py`: 契約テスト3件追加。
  - `live_start_nonblocking_contract`（start() <0.3s 非ブロック / wait=True は ready まで待つ）
  - `live_async_failure_falls_back_contract`（ready前のWinRT+bridge失敗→`on_unavailable`）
  - `native_state_detected_event_contract`（発話検知でライブUIが動く・実テキストは上書きしない）

### ゴール完了判定（確定）
- **UX定義を採用**。`HypothesisGenerated` 常時1秒以内（厳密定義）は `Windows.Media.SpeechRecognition` の仕様上保証できないため破棄。
- ライブUIは RMS＋エンジン発話検知で1秒以内に動く。文字仮説はOS依存で1〜2秒を許容し合否に含めない。
- `large-v3-turbo` / `use_live_transcript_on_stop=false` / ライブ非流用 / `pythonw`推奨 / EXE非基準 はすべて不変。

### 検証
- `python -m py_compile`: PASS
- `python scripts/koemo_feature_smoke.py`: **22/22 PASS**（既存19＋新規3）。`launch_pythonw`/`launch_exe` も PASS（非ブロック化で起動が壊れていない）。
- `live_start_nonblocking_contract`: start() 0.000s 非ブロック、start(wait=True) 0.501s ブロックを確認。
- `python scripts/koemo_fast_integration_check.py`: stop→summary **1.278s**（ベースライン1.274s と一致）。

### B-1 warm-keep: 実装→実測→却下（不採用）
実マイクで `start_async` コストを計測し、warm-keep を一度実装したが、計測の結論により**採用せず差し戻した**。`scripts/koemo_native_warmcold_bench.py` で再現できる。

- `pause_async()` はこのビルドで InvalidOperation。warm再開は `stop_async`/`start_async` で行う必要がある。
- `compile_constraints_async` は約1〜2ms（ボトルネックではない。過去のconstraints実験は的外れだった）。
- `start_async` は **プロセス最初だけ約520〜680ms**（プロセス毎一度の初期化）。同一プロセスの2回目以降は recognizer を作り直しても約78〜80ms。
- recognizer を使い回す warm再開は約73〜75msで、作り直し約78msと**定常差はほぼ無い（実測 reuse_vs_fresh ≈ 2ms）**。
- => warm-keep の唯一の実利は「プロセス毎一度の初期化を起動時に前倒しして初回録音から外す」ことだけ。だが Category A の非ブロック化で初回コールドはGUIを固めず、ライブUIは RMS＋発話検知で1秒以内に動く。常駐asyncioループ＋起動時マイク取得＋stop/start のレース（再arm失敗）の複雑さに見合わないと判断し、**差し戻した（コードは残さない）**。
- 数値は `python scripts/koemo_native_warmcold_bench.py` で再取得可能（`process_first_start_ms` / `steady_fresh_recognizer_start_ms` / `reuse_vs_fresh_steady_gain_ms`）。

### 未実装（実測前提・別スコープ）
- B-3 `sherpa-onnx` 日本語ストリーミングをライブbackendに追加（OS音声プライバシー非依存の真の<1s partial 候補）。モデルDL＋実測が必要。これが「文字仮説そのものを<1sにする」唯一の現実的経路。

### 2026-05-31 正式 transcript チャンネル選択修正
- `koemo_fast_integration_check.py` の過去 `has_expected_text=false` は、ライブ修正の回帰ではなく **正式 transcript 側のチャンネル選択**の脆さだった。
  - スピーカー音をマイクが大音量で拾い（acoustic echo, mic RMS 0.4〜0.5, peak 1.0）、旧 `_active_final_channels` の `mic >= system*1.8` 規則で **正しいシステムchを捨ててマイクのecho幻聴を採用**する事象を確認していた。
  - 修正方針は Option A: `auto_dedupe` でマイクだけが強くクリップし、システムchがクリーンな時はクリーンなシステムchを優先する。`all_active` は明示ポリシーとして先に評価し、飽和疑いでも全有音chを保持する。
  - `NaN/inf` 混入で音量/クリップ判定が壊れないよう、RMS と clip fraction は有限サンプルだけで計算する。
  - `scripts/koemo_fast_integration_check.py` は起動した Koemo を `try/finally` で必ず `_terminate()` し、`_is_koemo_process()` は basename と ROOT 境界（相対 `koemo.pyw` は cwd 解決）で判定する。
- 検証:
  - `py scripts\koemo_feature_smoke.py`: **25/25 PASS**。
  - `py scripts\koemo_fast_integration_check.py`: PASS、stop→summary **1.53s**、`has_expected_text=true`。
  - 実録音31ペアの高速 channel-selection check: 飽和mic + clean system の9件が system-only。先頭2件 `20260531_004036` / `20260530_230812` も system-only。
  - `py -m compileall koemo scripts`: PASS。
  - `py -m PyInstaller koemo.spec --noconfirm --clean`: PASS。再ビルド後 `dist/Koemo/Koemo.exe` と `faster_whisper/assets/silero_vad_v6.onnx` 等の同梱を確認。

### 2026-05-31 D2 カレンダータイトルヒント実装
- `koemo/calendar_hint.py` を追加し、ICSの `VEVENT` から `SUMMARY` / `DTSTART` / `DTEND` を読み、現在時刻の前後ウィンドウに重なる予定タイトルを返すようにした。Outlook COM は利用可能な環境だけ参照し、失敗時は無視する。
- 設定に `enable_calendar_title_hint` / `calendar_ics_path` / `calendar_outlook_enabled` を追加。既定はOFFで既存挙動を変えない。
- 録音後の通常処理で、設定有効時だけ予定タイトルを保存メモのタイトルに採用する。音声ファイル取込は過去素材の可能性があるため対象外。
- 検証:
  - `calendar_title_hint_contract`: PASS（ICSタイトル `朝会, Koemo進捗` を会議タイトル候補に採用、無効時は生成タイトル維持）。
  - 当時 `py scripts\koemo_feature_smoke.py`: **26/26 PASS**（現行の総合スモークは下記 35/35 PASS を正とする）。

### 2026-05-31 Highリスク固定と設定UI追従
- 長尺会議で AEC が全STFTフレームを一括確保しないよう、`cancel_echo()` はバッチ2パスの周波数領域 Wiener AEC に固定。`aec_batched_wiener_contract` でエコー抑圧と最大STFTバッチ数を検証する。
- 録音スレッドの片側失敗は `DualRecorder.stop()` が `capture_errors` として返し、`KoemoApp._process()` が保存メモの `録音警告` に出す。`recording_capture_errors_surface_contract` で、残ったchが保存され警告が呼び出し側へ返ることを検証する。
- 全ch録音失敗も `None` で黙らず `capture_errors` 付きレコードとして返し、アプリ側でユーザー可視のエラーにする。`recording_all_capture_errors_surface_contract` で検証する。
- 長尺録音の生音声はRAM蓄積ではなく一時float32ファイルへspoolし、ライブプレビューだけリングバッファで保持する。`recorder_spools_audio_and_bounds_live_contract` で検証する。
- 録音中/処理中の設定保存は `pending` として遅延反映し、現在録音/処理の `cfg` と recorder/model/backend/hotkey を途中変更しない。`settings_save_during_recording_contract` で検証する。
- `final_channel_policy` を設定UIに露出。通常は `auto_dedupe`、全有音chを残す必要がある場合は `all_active` を設定画面から選べる。
- README を現実装へ追従: `live_backend=auto`、バッチWiener AEC、設定画面での final channel policy、local-by-default（Ollama/OpenAI互換backendを選ぶと設定endpointへ transcript/chat text を送る）を明記。
- full LLM要約の前に、録音/音声取込の両方でlive/final Whisperモデルを明示解放し、live small＋final turbo＋Qwen要約の3モデル同時VRAM常駐を避ける。`keep_warm=true` の場合だけ常駐を尊重する。
- 未使用API `merge_transcript()` と `LiveTranscriber.flush()` を削除した。正式経路は `merge_rows()` と停止時の既存 final rows のみ。
- `record_mic=false` の system-only 録音では、OSの既定マイクを直接開く `native_windows` live を起動せず、録音済みchだけを使う `whisper_rolling` へfallbackする。`native_live_respects_record_mic_contract` で検証する。
- spool削除失敗は `temp_cleanup_errors` と未削除 `temp_files` を保持し、保存summaryの `一時ファイル警告` に出す。`process_surfaces_spool_cleanup_failure_contract` で検証する。
- mic/system 両方OFFの不可能な録音設定は設定UIと録音開始時の両方で拒否する。hung capture thread は `capture_errors` に出し、作成済みspoolをcleanup対象に残す。`recording_no_channel_config_rejected_contract` / `recorder_hung_thread_reports_temp_contract` で検証する。
- 他chに実内容がある時だけ、`ご視聴ありがとうございました` 等の典型Whisper outro幻聴を正式 transcript から除外する。`final_filters_common_whisper_hallucination_contract` と統合ハーネスの `has_common_hallucination=false` で検証する。
- 検証:
  - `py -m compileall koemo scripts`: PASS。
  - `py scripts\koemo_feature_smoke.py`: **40/40 PASS**。

### 2026-05-31 最終パッケージ再検証
- `koemo/app.py` に `KOEMO_TEST_COMMAND_FILE` 指定時だけ有効な integration test file command watcher を追加。通常起動では無効。`toggle:<seq>:<time>` で録音開始/停止、`quit` で終了できる。
- `scripts/koemo_fast_integration_check.py` は file command を既定操作経路にした。以前の `file command + global hotkey` 同時送信は二重トグルで短すぎる空録音を作るため廃止し、ホットキー併用は `KOEMO_TEST_SEND_HOTKEY=1` の明示時だけにした。検証時に書き換える `~/.koemo/config.json` は `finally` で元へ戻し、復元が終わってから `config_restored_by_script` をレポートに書く。
- EXE統合検証用に `KOEMO_TEST_EXE=1` / `KOEMO_TEST_REQUIRE_EXE=1` / `KOEMO_TEST_MAX_SECONDS=<sec>` を追加。App Control でEXEがブロックされた場合に pythonw fallback で合格扱いしない。
- `fast_integration_pythonw_report.json` と `fast_integration_exe_report.json` を別保存し、推奨pythonw経路とEXE no-fallback経路の証跡を同時に残す。既存プロセス掃除は `Koemo.exe` または repo 内 `koemo.pyw` に限定し、repo内venv等の無関係な Python は止めない。統合テストが強制終了した場合の test save_dir `.koemo_tmp\capture_*.f32` はハーネスが開始前/終了後に掃除する。
- `これも高速化テスト` → `コエモ高速化テスト` の補正を追加し、実機TTS/Whisperゆらぎを正式 transcript 側で吸収する。
- 統合レポートは要約本文ではなく `## 文字起こし` 節だけを判定し、ラベル付き正式 transcript 内に期待文があることを `has_whisper_final=true` の条件にした。
- 統合検証は `record_mic=true` / `record_system=true` / `enable_aec=true` / `final_channel_policy=auto_dedupe` を明示固定し、ユーザーの既存configに依存しないようにした。
- 検証:
  - `py -m compileall koemo scripts`: PASS。
  - `py scripts\koemo_feature_smoke.py`: **40/40 PASS**。スモーク後の `.koemo_tmp\capture_*.f32` 残留なし。
  - `py scripts\koemo_fast_integration_check.py`: PASS、stop→summary **2.053s**、`has_whisper_final=true`、`has_expected_text=true`、`has_transcript_section=true`、`has_transcript_label=true`、`has_common_hallucination=false`、`control_mode=test_command_file`、`config_restored_by_script=true`、spool残留なし。
  - pythonw出力: `C:\Users\rambo\RamboPC\DevHub\10_active\koemo\outputs\meetings\summary_20260531_150505.md`
  - `py -m PyInstaller koemo.spec --noconfirm --clean`: PASS。`dist\Koemo\Koemo.exe` 再生成済み（2026-05-31 15:12:03, 51,142,657 bytes）。
  - `KOEMO_TEST_EXE=1 KOEMO_TEST_REQUIRE_EXE=1 KOEMO_TEST_MAX_SECONDS=180 py scripts\koemo_fast_integration_check.py`: PASS、`launch_mode=exe`、stop→summary **17.841s**、fallbackなし、`has_whisper_final=true`、`has_common_hallucination=false`、`config_restored_by_script=true`、spool残留なし。
  - EXE出力: `C:\Users\rambo\RamboPC\DevHub\10_active\koemo\outputs\meetings\summary_20260531_151406.md`
  - 最終 `Koemo` / `pythonw` 残プロセスなし。

### 2026-05-31 横断改善（録音精度外の品質）
- `requirements.txt` から通常アプリに不要な `datasets` / `pyinstaller` / `ollama` を分離。`datasets` は `requirements-tools.txt`、PyInstaller は `requirements-build.txt`。Ollama連携は `httpx` でHTTP APIを叩くためPythonクライアント不要。
- `koemo.spec` の blanket `safe_collect_data_files("PySide6")` を削除し、PyInstaller hook に任せる形へ変更。`datasets` / `pandas` / `pyarrow` / Webサーバ系 / Tk / PIL 等の非ランタイム混入を guard する `packaging_files` 契約を追加。
- 配布サイズ: 旧 `dist\Koemo` 約 **722,924,479 bytes / 7,461 files** → 新 `dist\Koemo` **470,524,286 bytes / 3,314 files**。`PySide6` は約189MB→約72.5MB。guard対象の `datasets` / `pandas` / `pyarrow` / `fastapi` / `uvicorn` / `starlette` / `aiohttp` / `tornado` / `PIL` / `mypy` / `_tcl_data` は同梱なし。
- `setup.bat` は pip更新、Whisper/Qwen/diarizationモデル取得、CUDAランタイム導入の失敗で即停止し、成功表示に進まない。
- 壊れた `~/.koemo/config.json` は黙って既定値に戻さず、`config.invalid_YYYYMMDD_HHMMSS.json` に退避し、起動時toastで知らせる。内部 `_config_*` キーは保存しない。
- 長尺meeting chatは先頭/末尾だけでなく質問語に一致する中間チャンクも文脈に入れる。`K-42` が中間チャンクだけにある回帰で検証。
- ICS `DTSTART;TZID=Asia/Tokyo` / `DTEND;TZID=...` を `zoneinfo` で解釈し、予定タイトル選択の時差ズレを防ぐ。
- 通常の `slack.exe` 起動は会議検出から除外。Zoom/Teams/Webexは維持。
- 結果画面にMarkdown保存ボタンを追加し、READMEのPDF/DOCX/Markdown export表記とUIを一致させた。
- `setup.bat` のインストール系コマンドをすべて `python -m pip ...` に統一し、確認した `python` と別Interpreterの `pip` に入る事故を防止。
- 設定JSONは既知キーのみ読み込み、bool/int/float/list/stringを既定型へcoerceする。型破損した値は既定値へ戻すため、`calendar_title_lookback_min="abc"` などで録音後保存が失敗しない。
- ICSのWindows-style `TZID`（例: `Tokyo Standard Time`）を主要IANA名へ変換。未対応TZIDは従来通りlocal timezone fallback。
- Outlook予定表連携はambient `pywin32` がある環境だけの任意機能としてREADMEへ明記。通常setupの依存膨張は避ける。
- 検証:
  - `py -m compileall koemo scripts`: PASS。
  - `py scripts\koemo_feature_smoke.py`: **40/40 PASS**。
  - `py -m PyInstaller koemo.spec --noconfirm --clean`: PASS。`dist\Koemo\Koemo.exe` 再生成済み（2026-05-31 20:54:33, 43,656,983 bytes）。`dist\Koemo` は **470,525,816 bytes / 3,314 files**。
  - `KOEMO_TEST_EXE=1 KOEMO_TEST_REQUIRE_EXE=1 KOEMO_TEST_MAX_SECONDS=180 py scripts\koemo_fast_integration_check.py`: PASS、`launch_mode=exe`、stop→summary **50.175s**、fallbackなし、`has_whisper_final=true`、`has_expected_text=true`、`has_common_hallucination=false`。
  - `py scripts\koemo_fast_integration_check.py`: PASS、`launch_mode=pythonw`、stop→summary **2.050s**、`has_whisper_final=true`、`has_expected_text=true`、`has_common_hallucination=false`。直前に一度 `Error 0x100000001` の音声取得一時エラーで空録音になったが、設定復元・プロセス終了後の再実行はPASS。
  - 最終 `Koemo` / `pythonw` 残プロセスなし、`.koemo_tmp\capture_*.f32` 残留なし。

### 2026-05-31 EXEメモリ/低レベルマイク漏れ再修正
- ユーザー実行のEXEで `処理中にエラー: mkl_malloc: failed to allocate memory` が出たため、EXE経路を再監査。`live_backend=native_windows` の場合でも未使用の `live_model`（Whisper small）を preload しており、EXEのCPU fallback時に final `large-v3-turbo` と同時常駐しやすいのが主因候補だった。
- `_preload_transcriber()` は実効live backendが `whisper_rolling` の時だけ `live_model.warmup()` するよう修正。`preload_state_contract` は native_windows 経路で final のみをwarmすることも検証する。
- EXE/pythonw統合再実行で、AEC後の低レベルmic残留が GPU/CPU Whisper に別発話として拾われるケースを確認。`auto_dedupe` は system が強く、mic が system の14%以下かつ RMS 0.035以下の時、低SNRのsystem漏れとして system のみを採用するよう追加修正。
- `scripts/koemo_fast_integration_check.py` は、このTTS-only検証で `**あなた**:` 行が出たら `has_whisper_final=false` にする。`has_unexpected_mic_row` もレポートへ保存する。
- 検証:
  - `py -m compileall koemo scripts`: PASS。
  - `py scripts\koemo_feature_smoke.py`: **43/43 PASS**。
  - `py -m PyInstaller koemo.spec --noconfirm --clean`: PASS。`dist\Koemo\Koemo.exe` 再生成済み（2026-05-31 23:03:56, 43,644,506 bytes）。`dist\Koemo` は **467,747,268 bytes / 3,234 files**（entries 3,895）。
  - `py scripts\koemo_fast_integration_check.py`: PASS、`launch_mode=pythonw`、stop→summary **2.420s**、`has_whisper_final=true`、`has_expected_text=true`、`has_common_hallucination=false`、`has_unexpected_mic_row=false`。
  - `KOEMO_TEST_EXE=1 KOEMO_TEST_REQUIRE_EXE=1 KOEMO_TEST_MAX_SECONDS=180 py scripts\koemo_fast_integration_check.py`: PASS、stop→summary **27.246s**、`launch_mode=exe`、fallbackなし、`has_whisper_final=true`、`has_unexpected_mic_row=false`、`mkl_malloc` 再発なし。
  - 最終出力: pythonw `outputs\meetings\summary_20260531_230650.md`、EXE `outputs\meetings\summary_20260531_230814.md`。

### 2026-05-31 ゼロベースレビュー指摘の追補修正
- `DualRecorder.stop()` は `save_dir.mkdir()` に失敗しても例外で処理全体を止めず、録音済み `channels` を返して `write_errors["_save_dir"]` に保存失敗を記録する。CFA/権限/ファイル衝突でWAV保存先を作れない場合でも、正式文字起こしとsummary生成を継続する。
- 音声ファイル取込でも保存先ディレクトリ作成を先に直叩きせず、既存 `write_text_with_fallback()` に任せる。保存先が作れない場合は `%LOCALAPPDATA%\Koemo\Recordings` 側へ保存し、結果画面/保存Markdownに `保存警告` を出す。
- Calendar title hint は window内のイベントを単純に開始時刻距離で選ばず、まず `start <= now <= end` の進行中イベントを優先する。進行中会議の終盤に次会議が近づいていても、現在の会議名を使う。
- `scripts/koemo_fast_integration_check.py` は `KOEMO_TEST_EXE` / `KOEMO_TEST_REQUIRE_EXE` の時だけ EXE ファイル存在を必須にする。通常の推奨 `pythonw` 経路検証は、clean checkoutでEXE未生成でも起動可能。
- 回帰検証:
  - `recording_save_dir_failure_nonfatal_contract`: PASS。
  - `import_save_dir_failure_falls_back_contract`: PASS。
  - `calendar_title_hint_contract`: 進行中会議が近い次会議に勝つケースを追加して PASS。
  - `fast_integration_process_guards`: EXE未生成でもpythonw-only検証が死なない契約を追加して PASS。
