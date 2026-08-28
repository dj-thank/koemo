# 日本語完全文字起こし — Japanese Complete Transcription v1.0.0

音声・動画を入力すると、**読みやすい日本語本文、音響証拠を保った観測文字列、セグメント／単語時刻、字幕、品質情報、話者ラベル、モーラ情報**を一括生成するローカル優先の文字起こしシステムです。

従来の「モーラ対応ASRコア」から、実際に長時間音声を処理して成果物を書き出せるアプリケーションへ拡張しました。

## 重要な設計

```text
音声・動画
  ↓
faster-whisper（長音声、VAD、単語時刻、文脈、ホットワード）
  ↓
observedTranscript ── SHA-256で固定し上書き禁止
  ├─ セグメント時刻
  ├─ 単語時刻・確率
  ├─ 不確実区間
  ├─ 任意の話者ラベル
  └─ 任意の読み・モーラ
  ↓
normalizedTranscript ── 読みやすさ用の別レイヤー
  ├─ 既定: 決定論的な空白・Unicode・句読点整形
  └─ 任意: ローカルOllama（ID完全一致＋編集量ガード）
  ↓
JSON / TXT / Markdown / SRT / VTT / TSV / JSONL
```

**実際に聞こえた内容と、読みやすく整えた内容は同じフィールドに保存しません。**

```text
observedTranscript != normalizedTranscript
```

## 生成物

入力が `meeting.m4a` の場合、既定では以下を生成します。

```text
meeting.transcript.json   全情報を持つ正本
meeting.txt               読みやすい完成文字起こし
meeting.observed.txt      音響認識を保存した観測文字列
meeting.md                メタデータ＋本文＋時刻付きタイムライン
meeting.srt               字幕
meeting.vtt               Web字幕
meeting.segments.tsv      セグメント表
meeting.words.jsonl       単語時刻・確率・話者・読み・モーラ
```

## セットアップ

Python 3.11以上を使用します。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install '.[asr]'
```

漢字を含む各単語へ読みとモーラを付与する場合:

```bash
python -m pip install '.[mora]'
```

## 基本実行

```bash
jtranscribe meeting.m4a --output-dir transcripts
```

または:

```bash
python -m japanese_transcriber meeting.m4a --output-dir transcripts
```

既定値は日本語、`large-v3-turbo`、VAD有効、単語時刻有効です。

## 精度を上げる文脈と固有名詞

```bash
jtranscribe meeting.wav \
  --initial-prompt "生成AIと音声認識の技術会議です。" \
  --hotwords "森脇渉太、Kotodama、CTranslate2、モーラ、Qwen" \
  --output-dir transcripts
```

大量の固有名詞はUTF-8テキストから渡せます。

```bash
jtranscribe meeting.wav --hotwords-file glossary.txt
```

## 長時間・一括処理

faster-whisperの長音声セグメント処理とVADを利用します。ディレクトリも処理できます。

```bash
jtranscribe recordings/ --recursive --output-dir transcripts
```

複数入力では、入力ファイル名ごとに出力フォルダーを分離します。既存ファイルは既定で上書きしません。

## ローカルLLMで読みやすくする

Ollamaをローカル起動した状態で、任意のローカルモデルを指定します。

```bash
jtranscribe interview.wav \
  --ollama-model qwen3:4b \
  --context "日本語の技術インタビュー。発話内容や言い間違いは変更しない。"
```

安全境界:

- 既定ではループバックのOllamaだけを許可
- 各セグメントIDを完全に一致させる
- セグメントの欠落、重複、追加を拒否
- 元文との類似度と文字数比を検査
- ガードを超えたセグメントは決定論的整形へ戻す
- `observedTranscript`には一切書き戻さない

## 話者ラベル

外部ダイアライザーのRTTMを入力すると、セグメントと単語へ話者を割り当てます。

```bash
jtranscribe meeting.wav --rttm meeting.rttm
```

任意の話者IDは、最初に現れた順に `話者1`、`話者2`…へ安定変換します。RTTMを指定しなければ、話者を推測したふりはせず `null` のまま保存します。

## モーラ情報

かなだけの単語は追加依存なしでモーラ分割できます。漢字を含む単語は `pyopenjtalk` を明示的に有効にします。

```bash
jtranscribe lesson.wav --pyopenjtalk
```

例:

```text
がっこう → ガ / ッ / コ / ウ
きゃく   → キャ / ク
スーパー → ス / ー / パ / ー
```

## 品質・不確実性

各セグメントには `avgLogprob`、`noSpeechProb`、`compressionRatio`、単語ごとの`probability`、`uncertaintyReasons`を保存します。

```bash
jtranscribe audio.wav \
  --min-avg-logprob -0.8 \
  --min-word-probability 0.55 \
  --max-no-speech-prob 0.5
```

これらは校正済みの確率ではなく、**見直す区間を見つけるための証拠**として扱います。

## 出力形式を絞る

```bash
jtranscribe audio.wav --formats json,txt,srt,vtt
```

利用可能:

```text
json, txt, observed-txt, md, srt, vtt, tsv, words-jsonl, all
```

## モデル選択

```bash
# 精度優先
jtranscribe audio.wav --model large-v3

# 速度と精度のバランス
jtranscribe audio.wav --model large-v3-turbo

# 軽量
jtranscribe audio.wav --model small
```

GPU／CPU、量子化形式も選択できます。

```bash
jtranscribe audio.wav --device cuda --compute-type int8_float16
jtranscribe audio.wav --device cpu --compute-type int8
```

## 開発テスト

```bash
python -m unittest -v tests.test_complete_transcription
python -m compileall -q japanese_transcriber scripts training tests
npm test
```

## 公開検証の境界

公開CIで検証するもの:

- 日本語モーラ処理
- 観測文字列の不変性
- N-best融合とrank-only制約
- 長音声用パイプラインのモデル非依存テスト
- JSON/TXT/Markdown/SRT/VTT/TSV/JSONL出力
- RTTM話者割当
- ローカルLLMレスポンスの安全検査
- Python／Node.js構文

環境・モデル・データが必要なため、公開CIで成功を主張しないもの:

- 実際のモデル重量を使うfaster-whisper推論
- 実Ollama通信
- 学習者音声でのCER／Kana-CER／MLER評価
- Whisperの微調整
- 外部ダイアライザーモデルの推論

## ライセンス

MIT License — © 2026 SHOTA Moriwaki
