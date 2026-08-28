# MoraWeave

**Mora-aware, evidence-fused Japanese speech transcription with selective re-listening.**

MoraWeaveは、日本語音声から単に「最も自然な文章」を作るのではなく、音響・モーラ・公開語彙記憶・言い淀み保存の証拠を並行して保持し、不一致が大きい箇所だけを再度聴き直す研究用／実用向け基盤です。

## 1. 何を守るか

```text
observedTranscript   実際に聞こえた発話。保存後は不変。
normalizedTranscript 読みやすさの派生物。観測結果を上書きしない。
```

発話者が「昨日、学校を行きました」と言った場合、LLMが自然な「昨日、学校に行きました」を好んでも、観測文字列は前者のまま保存します。

## 2. アーキテクチャ

```text
16 kHz audio
   │
   ├─ ASR N-best lattice ─────────────── acoustic evidence
   ├─ mora/phone/F0/boundary heads ───── phonological evidence
   ├─ hashed public n-gram memory ────── lexical evidence
   └─ filler/restart/error head ───────── preservation evidence
                         │
                         ▼
                 Four-stream gate
                         │
             uncertainty / disagreement
                  ┌──────┴──────┐
                  │             │
              confident     ambiguous span
                  │             │
                  │       selective re-listen
                  │       higher beam / alt ASR
                  │             │
                  └──────┬──────┘
                         ▼
             immutable observed transcript
                         │
          optional local rank-only normalizer
                         ▼
             separate normalized transcript
```

## 3. クイックスタート

Python 3.11以上:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
```

モデルなしのデモ:

```bash
moraweave demo --output runs/demo
```

候補融合:

```bash
moraweave fuse examples/nbest.json --output runs/fused.json
```

ハッシュ化語彙記憶:

```bash
moraweave memory-build examples/public_text_manifest.jsonl \
  --database data/memory.sqlite3 \
  --rights-registry data/rights_registry.json
```

## 4. 実音声アダプター

公開コアはモデル非依存です。実音声では次を差し替え可能にします。

- faster-whisper / CTranslate2
- Qwen3-ASR
- Kotoba-Whisper系
- ReazonSpeech系
- 任意のOpenAI-compatibleローカルASR

重いモデルを全区間へ常時並列実行するのではなく、初回ASRで割れた区間だけへ追加モデルを投入します。

## 5. 公開データ

データ本体はこのリポジトリに同梱しません。各資産を権利台帳へ登録し、許可された処理だけを実行します。

```text
trainAllowed       学習へ使えるか
featureAllowed     ハッシュ化特徴へ変換できるか
redistributeRaw    原音声・原文を再配布できるか
speakerIdExport    元話者IDを成果物へ出せるか
```

Common Voiceでは音声を再配布せず、ローカルmanifestを生成し、話者IDは秘密ソルトによるHMACへ変換します。JMdictはライセンス表記付きで読み語彙へ使用できます。青空文庫は作品単位の権利manifestを要求します。

## 6. 評価

```text
CER                       漢字仮名交じり文字列
Kana-CER                  読み列
MLER                      モーラ列
Number Error Rate         数字・日付・金額
Disfluency Preservation   フィラー・言い直し保存
Unsupported Correction    音響根拠のない訂正
Boundary MAE/F1           モーラ境界
RTF                       実時間係数
```

読みが取得できない漢字列では、Kana-CERやMLERを偽の0にせず`null`として返します。

## 7. 正直な検証境界

モデル非依存テスト、権利ゲート、ハッシュ固定、融合、評価、出力契約はCI対象です。実モデルの精度、GPU速度、学習済み補助ヘッドの改善量は、重みと正解音声を使う別ベンチマークが必要です。

## 8. ライセンス

MoraWeaveコードはMIT。モデル、音声、辞書、コーパスは各提供元の条件に従います。
