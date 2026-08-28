# Koemo 日本語モーラASR 精度向上スタック

日付: 2026-08-28  
状態: 実装中・本番経路未切替  
対象: `koemo/asr/`

## 1. 目的

Koemoの音声認識を、単に自然な日本語へ書き直すシステムではなく、
**実際に発音された内容を日本語のモーラ単位で忠実に観測できるASR**へ進化させる。

最重要の不変条件は次のとおり。

1. `observedTranscript` は音響候補集合の既存候補からのみ選ぶ。
2. LLMは候補外の文章を生成して `observedTranscript` を置換しない。
3. 自然な文章への補正は `normalizedTranscript` に分離する。
4. 精度改善は、固定評価セット、アブレーション、信頼区間を伴って判定する。
5. 精度が不明な結果は無理に確定せず、`review` または `no_speech` に退避する。

## 2. 現在の判断経路

```text
音声
  │
  ├─ VAD / Whisper encoder
  │
  ├─ Whisper N-best候補
  │     ├─ Whisper score
  │     ├─ token IDs
  │     ├─ no-speech probability
  │     └─ 音声・設定の来歴
  │
  ├─ 補助音響証拠
  │     ├─ character CTC
  │     ├─ mora CTC
  │     ├─ forced alignment
  │     └─ F0 / voicing / accent
  │
  ├─ robust log-linear ranking
  │
  ├─ posterior calibration
  │
  ├─ mora-aware MBR consensus
  │     ├─ 重複候補の抑制
  │     ├─ N-best全体との期待モーラ距離
  │     └─ 既存候補だけを選択
  │
  ├─ confidence gates
  │     ├─ accept
  │     ├─ review
  │     └─ no_speech
  │
  └─ observedTranscript を固定
        └─ 任意でLLM rank-only / normalizedTranscript
```

## 3. 今回追加した精度機構

### 3.1 モーラ単位 Minimum Bayes Risk（MBR）選択

一番高いWhisperスコアだけを機械的に採用せず、各候補について
「N-best全体との期待モーラ編集距離」を計算する。

候補 `h` の概念的なリスクは次のとおり。

```text
Risk(h) = Σ posterior(y) × MoraEditDistance(h, y)
```

最小リスクの候補を選ぶことで、孤立した一位候補よりも、
複数候補が音として合意している中心候補を選べる。

ただし、MBRは新しい文章を合成しない。必ず既存のN-best候補を返す。
したがって `observedTranscript` の音響証拠性を維持できる。

### 3.2 重複ビームの確率水増し防止

同じモーラ列を持つ候補がビーム内に複数存在しても、それだけで確率質量が
不当に増えないよう、既定では重複候補のlogitを `log-mean-exp` で統合する。

表記が異なっても読みが同じ候補は、読み変換器を接続した場合に同じグループへ
まとめられる。

例:

```text
今日   -> キョウ
きょう -> キョウ
```

読み変換器がない場合でも、未知文字を消さず、文字単位のフォールバック証拠として残す。

### 3.3 事後確率校正

音響スコアをそのまま「80%の確率」などと解釈しない。
話者分離された検証セットで温度スケーリングを学習し、次を計測する。

- Negative Log-Likelihood（NLL）
- Brier score
- Expected Calibration Error（ECE）
- N-best entropy

温度は学習データではなく、必ず独立したcalibration splitで決定する。
最終test splitは温度決定に使用しない。

### 3.4 自動棄却・要確認ゲート

次の条件を個別に記録し、曖昧な結果を強制確定しない。

- 選択候補のposteriorが低い
- 一位と二位のconsensus marginが小さい
- N-best entropyが高い
- 選択候補のBayes riskが高い
- no-speech probabilityが高い

結果は次の三状態になる。

```text
accept     十分な根拠がある
review     曖昧であり、人または別モデルの確認が必要
no_speech  発話として扱わない可能性が高い
```

閾値は固定の思いつき値で本番投入せず、開発セット上のprecision-recall曲線と
運用コストから決める。

## 4. 評価指標

### 4.1 文字認識

- CER: 文字単位の置換・削除・挿入
- Kana-CER: 表記差を読みへ変換した後の文字誤り率
- Sentence Exact Match
- N-best Oracle CER: 正解が候補集合に含まれていた場合の上限性能

`1-best CER` と `oracle CER` の差が大きい場合、主な問題は候補生成ではなく
再ランキングにある。差が小さければ、N-best生成または音響モデル自体の改善を優先する。

### 4.2 発音・モーラ

- MER: Mora Error Rate
- モーラ境界F1
- モーラ開始・終了時刻のMAE / P50 / P95
- `ン`、`ッ`、`ー`、拗音、外来音ごとの誤り率
- 母音無声化、促音、長音、撥音の混同行列

### 4.3 学習者発音の保持

発音評価では、意図した正解文へ勝手に直すことが失敗になり得る。
そのため、次の三つを別々に持つ。

```text
targetReading      教材上の意図した読み
observedReading    人手で確認した実際の発音
hypothesisReading  ASRが観測した読み
```

評価する値:

- `target -> observed`: 学習者自身の発音差
- `observed -> hypothesis`: ASRの観測誤差
- `target -> hypothesis`: 正解文への近さ
- learner-error preservation score
- normalized-to-target rate

主指標は `observed -> hypothesis` であり、`target -> hypothesis` だけを最適化しない。

### 4.4 確率・運用

- NLL / Brier / ECE
- accept精度
- review率
- no-speech precision / recall
- hallucination率
- empty / repetition / compression異常率
- Real-Time Factor（RTF）
- P50 / P95 / P99 latency
- VRAM / RAM使用量

## 5. データ分割

同一話者がtrainとtestへ跨ると、話者適応を一般化性能と誤認しやすい。
原則として次の分割を採用する。

```text
train        学習・重み更新
validation   early stopping・モデル選択
calibration  温度・棄却閾値の決定
test         最終報告専用
challenge    雑音・方言・非母語・長音声などの難例
```

分割条件:

- speaker-disjoint
- 録音セッションdisjoint
- 教材文・プロンプトの重複を監査
- 性別、年齢帯、母語、熟達度、収録機器、SNRを可能な範囲で層化
- 個人単位の削除・同意管理ができるmanifest

## 6. 必須アブレーション

精度向上を主張する前に、同じtest setで次を比較する。

| ID | 構成 |
|---|---|
| A0 | 現行faster-whisper one-best |
| A1 | Whisper N-bestのWhisper scoreのみ |
| A2 | A1 + robust score fusion |
| A3 | A2 + mora-aware MBR |
| A4 | A3 + posterior calibration / abstention |
| A5 | A4 + character CTC |
| A6 | A5 + mora CTC |
| A7 | A6 + forced aligner consensus |
| A8 | A7 + rank-only local LLM |
| A9 | A8 + F0 / accent evidence |

各段階で次を確認する。

1. CERだけでなくMERが改善するか。
2. 学習者誤り保持率を悪化させないか。
3. 改善幅のbootstrap confidence intervalがゼロを跨がないか。
4. latencyとメモリ増加が運用上許容できるか。
5. 特定話者・特定母語だけを悪化させていないか。

## 7. 次の実装順序

### Gate 1: 実N-best評価

- faster-whisperの実モデルでN-bestを取得
- 1-best CERとoracle CERを計測
- 同一候補・空文字・timestamp tokenを正規化
- window単位の候補を不正に直積連結しない

合格条件は、候補集合のoracle性能に再ランキング改善余地があること。

### Gate 2: 読み変換とMBR校正

- MeCab / Sudachi / pyopenjtalk等を比較し、辞書版を固定
- 固有名詞・英数字・コードスイッチを保持
- speaker-disjoint calibration setで温度と棄却閾値を学習
- ECEとreview率を報告

### Gate 3: Forced Aligner比較

- Qwen3 Forced Aligner等を遅延ロードの実験経路へ接続
- 現行CTC alignmentと時刻誤差を比較
- 一致しない区間をuncertain spanとして保持
- ライセンス、モデル容量、VRAM、オフライン動作を確認

### Gate 4: モーラCTC学習

共有Whisper encoderへ、少なくとも次を接続する。

```text
Whisper decoder loss
character CTC loss
mora CTC loss
mora boundary loss
```

最初はencoderを凍結した軽量学習、次にLoRA / adapter、最後に必要な層だけを
段階的にunfreezeする。いきなり全層学習へ進まない。

損失重みは固定値だけで決めず、勾配ノルムと各タスクの学習曲線を監視する。
CTC blank比率、collapse後長、無限lossを常時記録する。

### Gate 5: 韻律証拠

- F0
- voicing
- duration
- accent nucleus
- accent phrase boundary

これらを文章の自然さ補正ではなく、候補間の音響的な区別と発音評価へ使用する。

## 8. 本番投入条件

次を満たすまで、現行 `Transcriber` の既定経路を置き換えない。

- 固定test setで主要指標が改善
- learner-error preservationが非劣化
- 校正指標と棄却性能を報告
- Windows / Linux CI通過
- 実モデルsmoke test通過
- 旧経路への即時rollbackが可能
- feature flagで段階的に有効化
- すべての決定にcandidate ID、モデル版、辞書版、設定hashを保存

## 9. 現在の正確な到達点

実装済み:

- モーラ正規化とCTC統合
- Whisper window N-best adapter
- 音響スコア融合
- `observedTranscript` 不変条件
- rank-only LLM契約
- モーラMBR consensus
- 重複候補抑制
- 温度スケーリングと校正指標
- accept / review / no-speech判定
- CER / Kana-CER / MER
- learner-error preservation指標
- Windows / Linuxの単体テストCI

未証明:

- 実音声での精度向上量
- 実モデルN-bestのoracle改善幅
- 最適な読み変換器
- forced alignerの実測優位性
- mora CTC学習後の改善量
- F0 / accent融合の改善量

したがって、現段階で言えるのは
**精度を安全に改善し、改善を正しく測るための判断・校正・評価基盤を実装した**
ということまでである。実精度の改善率は、固定データセットによる比較後にのみ報告する。
