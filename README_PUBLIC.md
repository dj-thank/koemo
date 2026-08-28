# Japanese Speaking Assessment Mora Core v0.3.0

日本語発話評価向けの、モーラ認識・Whisper N-best・観測文字列保護・ローカルLLM再ランキング・Whisper補助学習ヘッドをまとめた公開実装です。

> この成果物は `dj-thank/koemo` の独立した公開配布ブランチに置いています。`main` は変更していません。ルートに残る可能性がある旧ZIPは転送途中で破損したため**使用しないでください**。正本はこのブランチで直接閲覧できるソースです。

## 実装範囲

- `src/mora.mjs` — ひらがな・カタカナ・半角カナの正規化、拗音・外来音の結合、促音・撥音・長音を含むモーラ列、文字CTC時刻のモーラ統合
- `src/transcript-contract.mjs` — `observedTranscript` と音響候補をSHA-256で結び、LLM訂正による上書きを防止
- `src/asr-fusion.mjs` — Whisper、モーラCTC、語彙、ローカルLMの決定論的N-best融合
- `src/local-lm-reranker.mjs` — Ollamaを候補IDの順位付けだけに制限するrank-only実装
- `scripts/japanese_mora.py` — Python版モーラ処理と文字アラインメント統合CLI
- `scripts/whisper_nbest.py` — faster-whisper内部のCTranslate2生成結果から複数仮説とスコアを取得する1発話用アダプター
- `training/mora_multitask.py` — Whisper共有エンコーダーにモーラCTC、任意の音素CTC、境界分類ヘッドを追加
- `schemas/` — ASR候補、モーラ単位、文字起こし記録のJSON Schema
- `tests/` — Node.jsとPythonの単体テスト

## 不変条件

1. `observedTranscript` は音響候補から決め、保存後に上書きしません。
2. `normalizedTranscript` は別フィールドへ保存します。
3. ローカルLLMは既存候補の順位付けだけを行い、新しい観測文字列を生成しません。
4. 長音声は事前に発話単位へ分割し、別ウィンドウの候補を不正に連結しません。
5. 発音評価ではLLM正規化結果だけを根拠にしません。

## 実行

Node.js 20以上:

```bash
npm test
```

Python 3.11以上:

```bash
python -m unittest -v tests.test_python_components
python scripts/japanese_mora.py "きょうはスーパーへ行きます"
```

faster-whisper N-best:

```bash
python -m pip install '.[asr]'
python scripts/whisper_nbest.py sample.wav --model small --nbest 5 --beam-size 8
```

ローカルLLM再ランキングはOllamaのループバックURLだけを既定で許可します。

## 学習モデル

```bash
python -m pip install '.[train]'
```

```python
from training.mora_multitask import MoraMultitaskWhisper

model = MoraMultitaskWhisper.from_pretrained(
    "openai/whisper-small",
    mora_vocab_size=180,
    phone_vocab_size=80,
)
```

総損失は通常のWhisper文字列損失、モーラCTC、音素CTC、境界分類の重み付き和です。最初からWhisperのBPEを置き換えず、既存デコーダーを保ったまま音響エンコーダーへ日本語固有の監督信号を追加します。

## 検証の境界

この公開ブランチのCIは、モーラ分割、アラインメント統合、観測文字列の不変性、候補融合、rank-only制約、Python構文を検証します。実モデル重みを使ったfaster-whisper推論、Ollama実通信、学習者音声ベンチマーク、Whisper微調整は実行環境とデータが必要なため、公開CIでは未実施です。

## ライセンス

MIT License。詳細は `LICENSE` を参照してください。
