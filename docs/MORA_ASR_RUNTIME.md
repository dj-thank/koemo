# モーラASR 実モデル互換性

KoemoのN-best経路は、faster-whisperの公開 `Segment` ではなく、内部の
CTranslate2 Whisper `generate` を直接使用する。そのため、通常のone-best経路より
バージョン互換性を厳格に扱う。

## 既知の検証基準

2026-08-28時点の基準:

```text
faster-whisper == 1.2.1
CTranslate2    >= 4.0, < 5
```

インストール例:

```powershell
pip install -r requirements.txt -c constraints-mora-asr.txt
```

`constraints-mora-asr.txt` は、通常のKoemo依存関係を置き換えるものではなく、
低レベルN-best経路を再現する際の互換性制約である。

## 起動前プローブ

```python
from koemo.asr import assert_faster_whisper_runtime_compatible

report = assert_faster_whisper_runtime_compatible(model)
print(report.to_dict())
```

次を検査する。

- `model.model.generate`
- `model.encode`
- `model.get_prompt`
- `model.feature_extractor`
- `model.hf_tokenizer`
- `model.model.is_multilingual`
- `model.max_length`
- faster-whisperのパッケージ版
- CTranslate2のメジャー版

必要な属性がない場合、N-best推論前に `RuntimeError` で停止する。
one-bestへ黙ってフォールバックして比較結果を混在させない。

新しいfaster-whisper版で構造検査を通過した場合は、直ちに拒否せず
`untested_faster_whisper_version` 警告を記録する。ただし本番採用前に、
固定fixtureと実モデルsmoke testを行い、検証済み範囲を更新する。

## 保存すべき来歴

各ベンチマーク・観測結果へ最低限、次を保存する。

```text
faster-whisper version
CTranslate2 version
Python version
OS / platform
Whisper model ID or local model hash
Koemo commit SHA
N-best settings
suppression token settings
prompt/tokenizer settings
feature/VAD settings
```

`RuntimeCompatibility.to_dict()` は、パッケージ版、Python版、プラットフォーム、
検出した能力、警告、エラーをJSON化できる。

## 更新手順

1. 新しいfaster-whisper版を別環境へインストールする。
2. 互換性プローブを実行する。
3. 単体テストとend-to-end fixtureを実行する。
4. 実モデルで同一音声のtoken IDs、候補数、score、no-speech値を比較する。
5. 固定test splitでCER・MER・oracle・校正値を再計測する。
6. 非劣化を確認してから制約ファイルと検証基準を更新する。

単に「起動した」だけでは互換と判定しない。候補内容とscore意味論まで比較する。
