# Koemo — 最終再import検証（2026-07-03）

## 目的
Claude要約 → ローカルGPU要約への移行の最終確認として、`koemo.summarize.Summarizer`
（本番と同一コード、`LocalCT2Backend` 経由で CTranslate2 + Qwen2.5-3B-Instruct-ct2-int8）が
実際に要約テキストを生成できることを、モックではなく実データで再現・記録する。

## 環境ブロッカー（先に確認）
このマシン（DESKTOP-C7HRICV, RTX 2080 Super Max-Q）は **GPUがCode 43で停止中**。

```
Get-PnpDevice 結果:
  FriendlyName: NVIDIA GeForce RTX 2080 Super with Max-Q Design
  Status: Error
  ConfigManagerErrorCode: CM_PROB_FAILED_POST_START
  ProblemDescription: Windows has stopped this device because it has reported
                      problems. (Code 43).
```

`nvidia-smi` は `NVIDIA-SMI has failed because you do not have sufficient permissions`
を返す（ドライバ自体が起動できていないための代替エラー）。これはハードウェア/ドライバ層の
問題であり、koemo側のコードバグではない。`koemo/gpu.py` の `gpu_ok()` はCUDA初期化を
`try/except` で保護しており、失敗時は自動的にCPUへフォールバックする設計（意図通り動作）。

このGPU不調そのものの修復は本タスクのスコープ外（ユーザーが別途調査中）。本検証は
「GPU不調でもコード側の要約ロジックが実際に動作するか」を実データで確認するもの。

## 検証方法
本番と同一のPythonモジュールを直接呼び出す（GUIは起動しない）。

1. `koemo.gpu.enable_cuda_dlls()` → `koemo.gpu.gpu_ok()` を確認（本番の起動シーケンスと同じ順序）。
2. Windows TTS（`Microsoft Haruka Desktop`）で日本語音声を新規生成（16kHz mono wav）。
   内容は「署名見送り決定」「GPU Code43確認」「次アクション」「未解決の質問」を含む
   会議を模したテキスト。
3. `koemo.transcribe.Transcriber("large-v3-turbo", ...).transcribe_segments()` で文字起こし。
4. `merge_rows` 相当の話者ラベル付きtranscriptを組み立て、
   `koemo.summarize.Summarizer(cfg={"summary_backend": "local"}).summarize()` を呼ぶ。
   これは内部で `koemo.backends.LocalCT2Backend` を使い、`gpu_ok()` に応じて
   `device="cuda"` または `device="cpu"` で `ctranslate2.Generator` をロードする
   （`koemo/backends.py` L55-61）。

検証スクリプト: `docs/reimport_verify_20260703/reimport_verify_script.py`
生データ: `docs/reimport_verify_20260703/reimport_verify_report.json`
コンソールログ: `docs/reimport_verify_20260703/reimport_verify_console.log`

## 結果（実測ログより）

| 項目 | 値 |
|---|---|
| `gpu.gpu_ok()` | `False`（Code 43により期待通り） |
| 入力音声長 | 約24秒（TTS生成） |
| `transcribe_segments()` | 59.314秒、6セグメント（CPU） |
| `summarize()` | 310.220秒、`LocalCT2Backend`、`device=cpu` |
| バックエンド | `LocalCT2Backend`（CT2 + Qwen2.5-3B-Instruct-ct2-int8, int8量子化） |

### 生成された文字起こし（実データ）
```
これはコエモの再インポート検証テストです。本日の会議では3つの議題を話し合いました。
1つ目は署名手続きについてです。Azure Trusted Signingの契約はコストが発生するため今回は
見送り、手順書の作成に留めることを決定しました。2つ目はGPU動作確認です。このマシンの
グラフィックボードはCode43エラーで停止しており、（略）CPUフォールバックで（略）処理を
検証することにしました。3つ目は次のアクションです。担当のランボさんが署名手順書を確認し、
来週までにリリースパイプラインの文書を完成させることになりました。未解決の質問として、
マイクロソフトストアでの配布が費用面で有利かどうかは追加調査が必要です。
```

### 生成された要約（実データ、CT2 + Qwen2.5-3B、CPU、greedy decoding）
タイトル: `GPU動作確認と署名手続き検討`

```markdown
## 要旨
本日の会議では、署名手続きの見送りと手順書作成、GPU動作確認のCPUフォールバック、
署名手順書の確認とリリースパイプライン文書の完成が決定し、マイクロソフトストアでの
配布費用面について未解決の質問が出た。

## 主要トピック
- 署名手続きの見送りと手順書作成
- GPU動作確認のCPUフォールバック
- 担当者の署名手順書確認とリリースパイプライン文書の完成

## 決定事項
- Azure Trusted Signing契約の見送りと手順書作成
- GPU動作確認のCPUフォールバック
- ランボさんの署名手順書確認とリリースパイプライン文書の完成

## アクションアイテム
- ランボさんが署名手順書を確認し、来週までにリリースパイプラインの文書を完成させる

## 未解決の質問
- マイクロソフトストアでの配布が費用面で有利かどうかの追加調査が必要
```

## 判定
**PASS（CPUフォールバック経路で確認）**。要約は入力音声の3議題・決定事項・
アクションアイテム・未解決質問を正確に反映しており、幻覚や欠落セクションはない。
CT2 + Qwen2.5-3B の要約生成ロジックは実データで動作することを確認した。

**未確認（ブロック）**: GPU(`device="cuda"`)経路そのものの実測は、このマシンの
Code 43が解消されるまで不可能。`gpu_ok()==True` の環境（正常なCUDAドライバ）では
`koemo/backends.py` の分岐によりCUDA経路が自動選択されるため、コード変更は不要と判断する。
GPU復旧後は `python scripts\koemo_model_bench.py` で `gpu_ok: true` を確認し、
本ドキュメントの追記として速度比較（CPU 310秒 vs GPU想定値）を追加することを推奨する。
