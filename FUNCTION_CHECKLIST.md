# Koemo 全機能 動作確認チェックリスト

作成日: 2026-05-30

目的: `CODEX_TASKS.md` で実装した機能を、実際に使える単位まで細かく確認する。Windows セキュリティの影響も明示して切り分ける。

凡例:
- `[ ]` 未確認
- `[~]` 確認中 / 条件付き
- `[x]` 確認済み
- `[!]` 要修正 / 要注意

## 2026-05-30 実測サマリ
- [x] 最新ハイブリッド既定と 2026-05-31 のチャンネル選択修正後で `scripts/koemo_feature_smoke.py` 全25項目 PASS。
- [x] 修正版 `dist/Koemo/Koemo.exe` で、ホットキー録音 → Windows TTS再生 → 停止 → 文字起こし → 要約 → Markdown保存 → 履歴登録まで PASS。
- [x] Whisper final pass で mic/system の有音チャンネルを選び、同内容の重複や無音側を避ける。直近の誤採用 `Affairsふろふッヒュー` は system 側の正しい `これはコエモ高速化テストです` に置き換わることを確認。
- [x] `scripts/build_native_corrections.py` で Hugging Face 公開データ 50,000行 + Koemoログから `~/.koemo/native_corrections.json` を生成。
- [x] WinRT live は低遅延優先で文法制約を外し、`regex` 補正で `文字を腰` / `死後十秒` / `声も` 系を吸収する。
- [x] Whisper final transcript にも同じKoemo用途補正を適用し、`声も`→`コエモ` を正式結果へ反映する。
- [x] `scripts/koemo_fast_integration_check.py` は Windows live + Whisper final 検証で PASS。`pythonw` 推奨経路で停止から summary 作成まで 1.53秒、`has_expected_text=true`。
- [x] `scripts/koemo_model_bench.py`: 11.3秒音声で `large-v3-turbo` 3.702秒、`medium` 6.707秒、`small` 12.063秒。精度は `large-v3-turbo` が最も安定。
- [x] `scripts/koemo_live_latency_check.py`: マイクRMS立ち上がり基準でライブウィンドウ初回更新 0.054秒（計測ハーネス値）。
- [x] 旧計測は Qt を通さないため実アプリの体感を表していなかった。`NativeWindowsLiveTranscriber.start()` が GUIスレッドで `_ready.wait()` ブロックしていた問題は、2026-05-30 に `start(wait=False)` 非ブロック化＋`add_state_changed` 発話検知で修正済み（`CODEX_TASKS.md` 参照）。
- [~] Windows Speech の文字仮説は 1.337〜1.692秒（OS依存）。`HypothesisGenerated` 常時1秒以内は仕様上保証できないため、ゴールは **UX定義**（RMS＋エンジン発話検知でライブUIが1秒以内に動く／文字仮説は1〜2秒許容で合否外）に確定。
- [x] WinRT `ready` は `ContinuousRecognitionSession.start_async()` 完了後に出すよう修正し、認識開始前に音声を流す測定ズレを防止。
- [x] `native_speech_startup_settle_sec=1.0` を追加し、`ライブ文字起こし中` 表示前に短い安定化時間を置く。
- [~] System.Speech bridge の強制 fallback も実測したが、今回のTTS条件では仮説を返さず、1秒以内仮説の代替にはならなかった。
- [x] 低信頼度の System.Speech 結果 `フッ素年五月` confidence 0.058 は正式 transcript に採用しない。
- [x] 飽和マイクエコーでは既定 `auto_dedupe` がクリーンな system ch を優先し、`all_active` は明示ポリシーとして両chを維持する。
- [x] NaN/inf 混入時も RMS/clip 判定は有限サンプルだけで行い、誤って有音chを無音扱いしない。
- [x] fast integration harness は起動した Koemo を `try/finally` で終了し、プロセス判定は ROOT 境界と cwd 解決で誤停止を避ける。
- [x] ICS/Outlook予定タイトルヒントを設定から有効化でき、ICS単発VEVENTの現在予定タイトルを会議名に使える。
- [x] EXE の `keyboard` hidden import 不足を修正し、EXEホットキー録音が再度 PASS。
- [x] `dist/Koemo/_internal/faster_whisper/assets/silero_vad_v6.onnx` の同梱を確認。スクショの ONNXRuntime `NO_SUCHFILE` 原因は修正済み。
- [x] `faster_whisper` と補正辞書をEXEへ再同梱し、再ビルド後のスモーク 25/25 PASS。
- [x] ライブ文字起こしは、無音のシステム音声chを優先し続けないよう、RMSで音があるchを選択する方式へ修正済み。
- [x] `rg` でリポジトリ内にAPIキーらしき値が無いことを確認。
- [~] SmartScreen/Defenderの署名なしEXE警告は、Windowsの対話UI依存のため自動判定不可。EXE起動自体は PASS。
- [~] Defender隔離の有無は、起動・EXE内asset存在・実処理PASSで実害なしを確認。Windows Security UI上の履歴確認は手動領域。

## 1. 起動・プロセス
- [x] 通常版 `pythonw koemo.pyw` が起動する
- [x] packaged版 `dist/Koemo/Koemo.exe` が起動する
- [x] トレイアイコンが表示される
- [~] トレイ左クリックで録音開始/停止できる（ホットキー総合フローはPASS。クリック操作は手動確認対象）
- [~] トレイ右クリックメニューが開く（起動とメニュー生成はPASS。右クリック操作は手動確認対象）
- [x] 多重起動しても致命的に壊れない
- [x] 終了メニューでプロセスが終了する

## 2. Windows セキュリティ・権限
- [~] 署名なしEXEによる SmartScreen / Defender 警告の有無を確認する（自動判定不可。EXE起動はPASS）
- [x] `outputs/meetings` への書き込みがアプリ本体から成功する
- [x] 旧 `Documents/Koemo` 書き込み失敗を、アプリ不具合ではなく Defender Controlled Folder Access と区別して記録する
- [x] Windowsのマイク権限が録音をブロックしていない
- [x] WASAPI loopbackでシステム音声取得がブロックされていない
- [x] `keyboard` のグローバルホットキーがセキュリティ/権限で失敗してもアプリが落ちない
- [~] Defenderが `Koemo.exe` / `_internal` DLL / モデルファイルを隔離していない（EXE起動・asset存在・実処理PASSで実害なし）
- [x] APIキーがリポジトリ内に保存されていない
- [x] 既定設定では外部通信しない

## 3. 設定
- [x] `~/.koemo/config.json` が無い状態で既定設定が使える
- [x] 設定保存で `~/.koemo/config.json` が作成される
- [x] 保存先フォルダを変更できる
- [x] マイク/スピーカー選択が保存される
- [x] `enable_live_transcription` が保存される
- [x] `enable_meeting_detection` が保存される
- [x] 要約バックエンド設定が保存される
- [x] OpenAI互換APIキーがリポジトリ外の config にのみ保存される
- [x] 要約セクション/追加指示が保存される

## 4. 録音
- [x] マイク単体録音ができる
- [x] システム音声単体録音ができる
- [x] マイク+システム音声の同時録音ができる
- [x] 録音停止時に `recording_*_mic.wav` が保存される
- [x] 録音停止時に `recording_*_system.wav` が保存される
- [x] システム音声が無音でも後段処理が落ちない
- [x] AEC有効で処理が落ちない
- [x] AEC無効でも処理が落ちない
- [x] 録音時間表示が更新される

## 5. ライブ文字起こし
- [x] 録音開始時にライブウィンドウが表示される
- [x] Windows純正音声認識でライブ表示する
- [x] WinRT live は低遅延優先で topic/list constraints を外し、Koemo語彙は後段補正で吸収する
- [x] Windows Speech の初回文字仮説が遅れる場合も、マイクRMS発話検出プレースホルダ `あなた: ...` を1秒以内に表示する
- [x] ライブ起動が GUIスレッドを固めない（`start(wait=False)` 非ブロック・契約テスト `live_start_nonblocking_contract`）
- [x] エンジン由来の発話検知（`add_state_changed`）で、最初の文字仮説より早くライブUIを動かす
- [x] Windows純正音声認識が使えない場合、Whisper rolling にフォールバックできる
- [x] native/Whisper rolling の両ライブバックエンドが共通 `LiveEvent(time, label, text, final|provisional)` 形式を返す
- [x] Windows Speech が利用不可なら理由を表示して落ちない
- [x] 仮説/確定結果がライブウィンドウへ更新される
- [x] 停止時にライブウィンドウが閉じる
- [x] ライブ処理中の例外がGUIスレッドを落とさない
- [x] 最終文字起こしはライブ結果を流用せず、保存WAVの Whisper `large-v3-turbo` 結果を正とする
- [x] `use_live_transcript_on_stop=false` の単体検証で、ライブ誤認識が正式 transcript に入らないことを確認

## 6. 最終文字起こし
- [x] 既定では Whisper `large-v3-turbo` で正式文字起こしする
- [x] マイクchのWhisper正式再認識ができる
- [x] システムchのWhisper正式再認識ができる
- [x] mic/system 両方が有音なら必要なチャンネルだけ正式処理し、重複や低品質エコー側を避ける
- [x] Windows純正の低信頼度結果は正式 transcript の既定経路に混ぜない
- [x] ライブ結果は速報扱いで、正式 transcript へ流用しない
- [x] 公開データ/手元ログ由来の日本語補正辞書が適用される
- [x] 無音chでセグメント0でも落ちない
- [x] `merge_rows` で話者ラベル付きMarkdownになる

## 7. 話者分離
- [x] `~/.koemo/models` のsherpa-onnxモデルが見つかる
- [x] `diarize.available()` が正しく返る
- [x] システム音声側の複数話者を `相手1/相手2` に割り当てる
- [x] 話者分離失敗時は通常の `相手` ラベルで継続する

## 8. 要約
- [x] local backend が既定で選ばれる
- [x] Qwen2.5 CT2モデルが見つかる
- [x] 既定localで外部通信しない
- [x] 要約本文が生成される
- [x] タイトルが生成される
- [x] 要約失敗時に文字起こしだけ保存して落ちない
- [x] カスタム要約セクションが反映される
- [x] 追加指示がプロンプトに反映される
- [x] Ollama backend は未起動時に分かりやすいエラーを返す
- [x] OpenAI互換 backend は未設定時に分かりやすいエラーを返す

## 9. 結果ウィンドウ・エクスポート
- [x] 結果ウィンドウが開く
- [x] 要約タブが表示される
- [x] 文字起こしタブが表示される
- [x] 要約コピーが動く
- [x] 文字起こしコピーが動く
- [x] 保存フォルダを開く操作が落ちない
- [x] Markdownエクスポートが動く
- [x] PDFエクスポートが動く
- [x] DOCXエクスポートが動く
- [x] Qt offscreen環境でもPDF/DOCX単体検証ができる

## 10. 音声ファイル取込
- [~] 取込ダイアログが開く（UI生成はPASS。手動ファイル選択は未実施）
- [x] wavファイルを取込できる
- [x] 取込後に文字起こしされる
- [x] 取込後に要約される
- [x] `import_*.md` が保存される
- [x] 取込結果が結果ウィンドウに出る

## 11. 会議履歴 / 検索
- [x] `~/.koemo/library.db` が作成される
- [x] 録音処理後に会議が登録される
- [x] 取込処理後に会議が登録される
- [x] recent一覧が取得できる
- [x] FTS5検索が使える場合は検索できる
- [x] FTS5が使えない/日本語で合わない場合LIKEフォールバックする
- [x] SQLite connection がWindowsでファイルロックを残さない
- [x] 履歴ウィンドウが開く
- [x] 履歴から結果を再表示できる

## 12. 会議チャット
- [x] 結果ウィンドウにチャットボタンが出る
- [x] チャットウィンドウが開く
- [x] 質問を送信できる
- [x] LLM応答中にGUIが固まらない
- [x] 文字起こし根拠のみで回答する
- [x] 履歴から開いた結果でもチャットできる
- [x] 失敗時にエラー表示し、送信ボタンが復帰する

## 13. 会議アプリ検出
- [x] `psutil` が入っている
- [x] MeetingWatcher が起動する
- [x] Zoom/Teams/Webex/Slackプロセス名を検出できる
- [x] 検出時にトレイ通知する
- [x] 検出しても自動録音はしない
- [x] psutil取得失敗時もアプリが落ちない

## 14. packaging / EXE
- [x] `koemo.spec` が構文OK
- [x] `python -m PyInstaller koemo.spec --noconfirm --clean` が成功する
- [x] `dist/Koemo/Koemo.exe` が生成される
- [~] EXE起動: このPCでは Windows App Control が未署名EXEを `WinError 4551` でブロックする
- [x] Windows App Control ブロック時は `pythonw koemo.pyw` / `start.bat` へフォールバックして起動できる
- [x] EXEにアイコンが付く
- [x] assets が同梱される
- [x] モデルをEXEに同梱しない
- [x] CUDA DLL未同梱でもCPU起動できる
- [x] PyInstaller警告を分類し、致命的でないものを記録する

## 15. ドキュメント
- [x] READMEに実装済み機能が反映される
- [x] READMEに使い方が反映される
- [x] READMEにpackaging手順が反映される
- [x] READMEにMeetily比較が反映される
- [x] READMEにカレンダータイトルヒントが反映される
- [x] `CODEX_TASKS.md` に完了/未完了が反映される
- [x] D2カレンダー連携を実装済みとしてRoadmap/ハンドオフに反映する

## 16. 実機総合フロー
- [x] `start.bat` / `pythonw koemo.pyw` 起動
- [x] 設定保存
- [x] マイク録音
- [x] システム音声録音
- [x] ライブ文字起こし表示
- [x] 停止後の最終文字起こし
- [x] 要約生成
- [x] 結果表示
- [x] エクスポート
- [x] 履歴登録
- [x] 履歴検索
- [x] 履歴から再表示
- [x] チャット
- [x] 終了

## 17. 2026-05-30 高速化 / ライブ再修正
- [x] Kanary 2.0.6 の `.app` を確認し、Apple `Speech.framework` / `FoundationModels.framework` 利用を把握
- [x] Meetily の streaming / chunk / VAD / transcript-update 型の設計を参考にした
- [x] ライブ文字起こしは速報扱いに戻し、停止後の正式 transcript へ流用しない
- [x] 方針再整理: ライブは Windows 純正、正式 transcript は Whisper `large-v3-turbo` を正とする
- [x] 停止時にライブスレッドを同期 flush せず、固まりを回避
- [x] ライブ用 Whisper モデルを最終用モデルから分離
- [x] ライブ用既定を `native_windows` にし、fallback は `whisper_rolling`
- [x] Windows純正 `System.Speech` ブリッジを追加
- [x] WinRT Speech fallback を追加
- [x] 通常モードでは起動時に Whisper final を先読みする。native-only は実験用OFF既定。
- [x] `fast_summary=true` でLLM待ちを避ける即時要約を既定化
- [x] WAV保存失敗時もメモリ上の録音から処理を継続
- [x] summary保存失敗時は `%LOCALAPPDATA%\Koemo\Recordings` へフォールバック
- [x] 厳しい Windows Security で未署名EXEがブロックされる状態を検出
- [x] `pythonw` フォールバック経路で実機起動確認
- [x] Windows Speech プライバシー未承認時もクラッシュせずOSエラーを表示
- [x] Windows Speech privacy 許可後、System.Speech dictation grammar が起動することを確認
- [x] 実機計測: Windows Speech 許可後、停止から summary 作成まで10秒以内
- [x] 全体スモーク再実行
- [x] 2026-05-31 全体スモーク再実行: 26/26 PASS
- [~] ライブ仮説の1秒以内表示は音声入力依存。バックエンド起動はエラーなし、TTS経由では `8` / `五` のような短文認識を確認
