# Koemo — コード署名セットアップ手順書（Azure Artifact Signing）

**このドキュメントは手順書のみ。課金が発生する操作（Azureサブスクリプション作成・
Artifact Signingアカウント作成）はユーザーの明示承認なしに実行しない。**

## 前提・費用（2026-07時点、公式ソース確認済み）

- サービス名は **Trusted Signing → Artifact Signing** に改称済み（機能差分なし）。
- **料金**: Basic プラン **$9.99/月**（月5,000署名まで、超過は1署名あたり$0.005）。
  Premium プランは$99.99/月（月10万署名まで）。Koemoの署名対象は毎リリース数個の
  PE/installerのみなので **Basicで十分**。
  出典: [Azure Artifact Signing Pricing](https://azure.microsoft.com/en-us/pricing/details/artifact-signing/)、
  [Artifact Signing FAQ](https://learn.microsoft.com/en-us/azure/artifact-signing/faq)。
- **無料/トライアル/スポンサー付きAzureサブスクリプションは不可**。有償(Pay-As-You-Go
  または Enterprise Agreement)サブスクリプションが必須。
- **個人開発者（Individual）として利用可能**（Public Trust証明書）。対象国は
  米国・カナダ・EU・英国の組織、および米国・カナダの個人開発者。日本の個人開発者は
  現時点でPublic Trust個人開発者としての対象国リストに入っていない可能性がある点に注意
  （組織としてのidentity validationか、対象国拡大の確認が必要）。
  出典: [Artifact Signing FAQ - Identity validation](https://learn.microsoft.com/en-us/azure/artifact-signing/faq)。
- 本人確認（identity validation）が必要（政府発行ID＋住所記載の公共料金請求書等）。
  完了まで日数がかかる場合がある。

## 全体の流れ

```
[Azure側 一度だけ]                          [ローカル/CI 毎回]
1. Azureサブスクリプション(有償)を用意        4. SignTool + .NET8 Runtime + dlib を導入
2. Artifact Signingアカウント作成            5. metadata.json を作成（repo外）
3. 証明書プロファイル作成 + ロール割当         6. koemo_release_build.py を実行
   (Identity Verifier → 本人確認 → Signer)
```

## ステップ1〜3: Azureリソース作成（課金発生・要承認）

**この節はユーザー承認後に実施すること。承認が無い限りこの手順書の記載のみに留める。**

1. Azure Portal で **Microsoft.CodeSigning** リソースプロバイダーを有効化
   (サブスクリプション → リソースプロバイダー)。
2. Artifact Signing アカウントを作成（リージョンは **Japan East** を推奨:
   エンドポイント `https://jpe.codesigning.azure.net`。リージョンと証明書プロファイル
   のリージョンが一致しないと signing 時に 403/`SignerSign()` 失敗になるため要注意）。
3. Identity validation（本人確認）を作成し、確認メールのリンクを7日以内に承認。
4. 証明書プロファイル（Certificate Profile）を作成し、Public Trust を選択。
5. 署名を実行するアカウント/managed identityに
   **Artifact Signing Certificate Profile Signer** ロールを割り当てる
   (Access Control (IAM))。

参考: [Artifact Signing tutorials](https://learn.microsoft.com/en-us/azure/artifact-signing/)、
[Assign roles in Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/tutorial-assign-roles)。

## ステップ4: ローカルツール導入（費用なし）

```powershell
# SignTool + dlib + 依存を一括導入（winget経由、推奨）
winget install -e --id Microsoft.Azure.ArtifactSigningClientTools
```

winget が使えない場合の手動導入:

1. **SignTool**: Windows SDK 10.0.2261.755 以降（`Microsoft.Windows.SDK.BuildTools`
   NuGetパッケージ経由でも可）。
2. **.NET 8.0 Runtime**（x64）: https://dotnet.microsoft.com/download/dotnet/8.0
3. **Artifact Signing dlib**: NuGetパッケージ `Microsoft.ArtifactSigning.Client`
   を展開し、`Azure.CodeSigning.Dlib.dll`（x64版）のパスを控える。
4. **Visual C++ Redistributable**（最新版）。

出典: [Set up signing integrations to use Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-signing-integrations)
（2026-05-14付、Microsoft公式）。

## ステップ5: metadata.json 作成（リポジトリ外に保存）

```json
{
  "Endpoint": "https://jpe.codesigning.azure.net",
  "CodeSigningAccountName": "<Artifact Signingアカウント名>",
  "CertificateProfileName": "<証明書プロファイル名>"
}
```

**リポジトリ内に置かないこと。** `koemo_release_build.py` の `ensure_signing()` は
`KOEMO_SIGNING_METADATA` がrepo内パスだと明示的に拒否する安全チェックを持つ
（`scripts/koemo_release_build.py` L71-80 既存実装）。

## ステップ6: koemoの既存署名パイプラインを実行

コード側はすでに完成済み（`scripts/koemo_release_build.py`）。以下の環境変数を
設定するだけで署名済みリリースが作れる。

```bat
set KOEMO_SIGNTOOL=C:\path\to\signtool.exe
set KOEMO_SIGNING_DLIB=C:\path\to\x64\Azure.CodeSigning.Dlib.dll
set KOEMO_SIGNING_METADATA=C:\path\outside\repo\metadata.json
python scripts\koemo_release_build.py --install-tools
```

このスクリプトは以下を自動で行う（既存実装、変更不要）:
1. `dist\Koemo` を PyInstaller で再ビルド。
2. `dist\Koemo` 配下の全 `.exe`/`.dll`/`.pyd` に `signtool sign` を実行
   （`/fd SHA256 /tr http://timestamp.acs.microsoft.com /td SHA256`、
   Artifact Signing推奨のタイムスタンプ局を使用）。
3. `signtool verify` と `Get-AuthenticodeSignature` で署名を検証。
4. portable zip とJapanese Inno Setup installerを作成し、installerにも署名。
5. `secret_scan()` でAPIキー等の混入がないことを確認。
6. `release\Koemo-0.1.0-rc1-release.json` に `signed: true`, `signing: "Azure Artifact Signing"`,
   `signed_pe_count` を記録し、`SHA256SUMS.txt` を生成。

### 検証（署名後に必ず実施）

```powershell
signtool verify /pa /all "dist\Koemo\Koemo.exe"
Get-AuthenticodeSignature -LiteralPath "dist\Koemo\Koemo.exe" | Format-List Status, StatusMessage
```

`Status` が `Valid` であること。署名済みでもSmartScreen reputationが育つまでは
警告が出ることがある（ファイルハッシュのダウンロード実績が蓄積すると解消、
`release-notes-ja.md` に既記載）。

## 既知のエラーと対処（公式FAQより抜粋）

| エラー | 原因/対処 |
|---|---|
| 403 Forbidden | リージョン不一致、ロール未割当、`.NET`/`dlib`バージョン不一致のいずれか。metadata.jsonのEndpointと証明書プロファイルのリージョンを再確認。 |
| "No certificates were found that met all the given criteria." | signtoolがローカル証明書ストアを見ていて、Artifact Signing証明書を使えていない。dlibパス/バージョンを再確認。 |
| SignTool silently fails, no error code | .NET 8 Runtimeが未導入。 |
| ポップアップでログイン画面が出る（CI環境） | `DefaultAzureCredential`の認証順序の問題。managed identityまたは`AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET`の環境変数credentialを使う。 |

出典: [Artifact Signing FAQ - Common error codes](https://learn.microsoft.com/en-us/azure/artifact-signing/faq)。

## 代替: Microsoft Store (MSIX) 経由の自動署名

Azure Artifact Signingの代わりに、Microsoft Store 経由でMSIXパッケージとして配布する
選択肢もある。この場合Store側が自動的に署名を行うためAzureアカウントは不要だが、
Store登録費用（開発者アカウント登録費、個人は少額の一度払い）とStore審査プロセスが
必要になる。現行の `koemo.iss`（Inno Setup EXEインストーラー）とは別パッケージング
（MSIXパッケージング + Partner Center登録）が必要なため、着手する場合は
別タスクとしてスコープを切ることを推奨する。

## このタスクでの結論

- 実際のAzureリソース作成・契約は**実行していない**（課金操作のため、本手順書止まり）。
- コード側の署名ロジック（`scripts/koemo_release_build.py`）は**既存実装で完成済み**、
  署名認証情報を用意すれば追加のコード変更なしで署名済みリリースを生成できることを
  ソースレビューで確認済み。
- 現在の `release/Koemo-0.1.0-rc1-release.json` は `signed: false` / `signing: "UNSIGNED-BETA"`
  のままであり、上記ステップ1〜6をユーザー承認のもとで実施するまで未署名状態が続く。
