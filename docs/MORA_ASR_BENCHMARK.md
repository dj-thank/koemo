# Koemo モーラASR ベンチマーク手順

このベンチマークは、基準システムと改善候補を同じ音声集合で比較し、
CER・Kana-CER・MER・学習者誤り保持・N-best oracle・信頼区間・退行ゲートを
一つのJSONレポートへまとめる。

モデル推論と評価を分離しているため、同じ予測JSONLから何度でも同じ評価を再現できる。

## 1. 実行例

```powershell
python scripts/koemo_mora_asr_bench.py `
  --manifest data/mora-benchmark/manifest.jsonl `
  --baseline data/mora-benchmark/baseline.jsonl `
  --candidate data/mora-benchmark/candidate.jsonl `
  --split test `
  --bootstrap-samples 5000 `
  --require-cer-ci `
  --require-mora-ci `
  --fail-on-regression `
  --output outputs/mora-benchmark-report.json
```

Linux/macOSでは行継続を `\` に置き換える。

固定fixtureによる動作確認:

```powershell
python scripts/koemo_mora_asr_bench.py `
  --manifest tests/fixtures/mora_asr/manifest.jsonl `
  --baseline tests/fixtures/mora_asr/baseline.jsonl `
  --candidate tests/fixtures/mora_asr/candidate.jsonl `
  --bootstrap-samples 200 `
  --require-cer-ci `
  --require-mora-ci `
  --fail-on-regression
```

## 2. manifest JSONL

1行が1発話である。

```json
{
  "utteranceId": "u000001",
  "speakerId": "speaker-001",
  "split": "test",
  "referenceText": "学校へ行きました",
  "referenceReading": "ガッコウヘイキマシタ",
  "targetReading": "ガッコウヘイキマシタ",
  "observedReading": "ガコウヘイキマシタ",
  "audioSha256": "<64-character SHA-256>",
  "groups": {
    "l1": "en",
    "level": "A2",
    "snr": "clean"
  }
}
```

### 必須フィールド

- `utteranceId`: 発話の一意ID
- `speakerId`: 話者の匿名ID
- `split`: `train` / `validation` / `calibration` / `test` / `challenge`
- `referenceText`: 人手確認した実際の発話内容

### 推奨フィールド

- `referenceReading`: `referenceText` の読み
- `targetReading`: 教材上、意図された読み
- `observedReading`: 学習者が実際に発音した読み
- `audioSha256`: 評価音声バイト列のSHA-256
- `groups`: 母語、熟達度、雑音条件などの匿名属性

`referenceReading` は教材の正解ではなく、文字起こし参照文の読みである。
学習者の発音差は `targetReading` と `observedReading` の差として別に保存する。

## 3. prediction JSONL

基準系と候補系を別ファイルにする。1行が1発話である。

```json
{
  "utteranceId": "u000001",
  "systemId": "mora-mbr-v1",
  "text": "学校へ行きました",
  "reading": "ガコウヘイキマシタ",
  "status": "accept",
  "latencyMs": 381.4,
  "candidates": [
    {
      "candidateId": "w0000-h00",
      "text": "学校へ行きました",
      "reading": "ガコウヘイキマシタ",
      "score": -0.21
    },
    {
      "candidateId": "w0000-h01",
      "text": "学校に行きました",
      "reading": "ガコウニイキマシタ",
      "score": -0.27
    }
  ]
}
```

- `text`: システムが最終選択した観測文字列
- `reading`: 最終候補の読み。MERを出す場合は必須
- `status`: `accept` / `review` / `no_speech`
- `latencyMs`: 発話処理時間
- `candidates`: 同じ発話ウィンドウから得たN-best

候補の `score` はログに残すが、ベンチマークのoracle値は正解との距離で計算する。

## 4. 分割漏洩の検査

既定では次をエラーにする。

- 同じ `utteranceId` の重複
- 同じ `speakerId` が複数splitへ出現
- 同じ `audioSha256` が複数splitへ出現

意図的な例外がある場合のみ、次を明示する。

```text
--allow-speaker-overlap
--allow-audio-overlap
```

最終精度報告では、原則としてこれらを使用しない。

## 5. 出力の読み方

### CER / MER

誤り率は発話ごとの単純平均ではなく、全置換・削除・挿入数を合算した
micro error rateである。短い発話だけが過大な重みを持つことを避ける。

### Oracle CER / Oracle MER

N-best候補の中で、参照に最も近い候補を選べたと仮定した上限値である。

```text
selected CER が悪く oracle CER が良い
  -> 候補生成はできている。再ランキング改善を優先

selected CER と oracle CER がともに悪い
  -> N-best生成、音響モデル、VAD、言語・読み処理を優先
```

### 学習者誤り保持

- `meanLearnerPreservation` が高いほど、実際の発音を保持できている
- `normalizedToTargetRate` が高いほど、発音差を教材正解へ戻している可能性がある

発音評価用途では、CERが改善しても `normalizedToTargetRate` が悪化した場合は
本番採用しない。

### 話者単位bootstrap

同一話者の複数発話は独立ではないため、発話ではなく話者を復元抽出する。

レポートの `delta` は次である。

```text
candidate error rate - baseline error rate
```

- 負: 候補系が良い
- 正: 候補系が悪い
- `upperBound < 0`: 指定信頼水準で改善を支持

## 6. 退行ゲート

既定では、候補系が次を悪化させると失敗する。

- CER
- MER
- learner-error preservation
- normalized-to-target rate

許容幅は次で設定する。

```text
--max-cer-regression
--max-mora-regression
--min-preservation-delta
--max-normalization-increase
```

統計的な改善まで必須にする場合:

```text
--require-cer-ci
--require-mora-ci
```

`--fail-on-regression` を付けると、ゲート失敗時の終了コードは `2` になる。
GitHub Actionsやローカルのリリース判定にそのまま使用できる。

## 7. 推奨運用

1. 推論時のモデル名、commit SHA、辞書版、設定hashを予測ファイルと同時保存する。
2. calibration splitで温度・閾値を決定する。
3. test splitは最終比較まで開かない。
4. 全体値だけでなく、母語・熟達度・SNR・デバイス別に確認する。
5. 改善候補はfeature flagで導入し、基準系へ即時rollbackできるようにする。
6. 実音声や個人情報をGitへコミットせず、manifestには匿名IDとhashだけを置く。
