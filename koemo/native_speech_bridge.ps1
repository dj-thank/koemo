param(
    [string]$Language = "ja-JP"
)

$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-EventJson {
    param(
        [string]$Type,
        [string]$Text,
        [double]$Confidence = 0.0,
        [string]$Error = ""
    )
    $obj = [ordered]@{
        type = $Type
        text = $Text
        confidence = $Confidence
        error = $Error
        ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    }
    [Console]::Out.WriteLine(($obj | ConvertTo-Json -Compress))
    [Console]::Out.Flush()
}

function Get-KoemoGrammarPhrases {
    $paths = @(
        (Join-Path $PSScriptRoot "data\native_corrections.json"),
        (Join-Path $HOME ".koemo\native_corrections.json")
    )
    $seen = @{}
    $phrases = New-Object System.Collections.Generic.List[string]
    foreach ($jsonPath in $paths) {
        if (!(Test-Path -LiteralPath $jsonPath)) { continue }
        try {
            $data = Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            continue
        }
        $candidates = @()
        if ($data.grammar_phrases) { $candidates += @($data.grammar_phrases) }
        foreach ($candidate in $candidates) {
            $phrase = ([string]$candidate).Trim()
            $phrase = [regex]::Replace($phrase, "\s+", "")
            if ($phrase.Length -lt 3 -or $phrase.Length -gt 28) { continue }
            if ($seen.ContainsKey($phrase)) { continue }
            $seen[$phrase] = $true
            $phrases.Add($phrase) | Out-Null
            if ($phrases.Count -ge 300) { return $phrases }
        }
    }
    return $phrases
}

function Add-KoemoPhraseGrammar {
    param(
        [System.Speech.Recognition.SpeechRecognitionEngine]$Engine,
        [System.Globalization.CultureInfo]$Culture
    )
    try {
        $phrases = Get-KoemoGrammarPhrases
        if ($phrases.Count -le 0) { return }
        $choices = [System.Speech.Recognition.Choices]::new()
        foreach ($phrase in $phrases) {
            $choices.Add($phrase) | Out-Null
        }
        $builder = [System.Speech.Recognition.GrammarBuilder]::new()
        $builder.Culture = $Culture
        $builder.Append($choices)
        $grammar = [System.Speech.Recognition.Grammar]::new($builder)
        $grammar.Name = "Koemo phrase hints"
        $Engine.LoadGrammar($grammar)
    }
    catch {
        # Phrase hints are optional. Dictation remains the primary recognizer.
    }
}

try {
    Add-Type -AssemblyName System.Speech
    $culture = [System.Globalization.CultureInfo]::GetCultureInfo($Language)
    $recognizerInfo = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() |
        Where-Object { $_.Culture.Name -eq $culture.Name } |
        Select-Object -First 1
    if ($null -eq $recognizerInfo) {
        throw "Windows Speech recognizer is not installed for $Language"
    }

    try {
        $engine = [System.Speech.Recognition.SpeechRecognitionEngine]::new($culture)
    }
    catch {
        $engine = [System.Speech.Recognition.SpeechRecognitionEngine]::new()
    }
    Add-KoemoPhraseGrammar -Engine $engine -Culture $culture
    $engine.LoadGrammar([System.Speech.Recognition.DictationGrammar]::new())
    $engine.SetInputToDefaultAudioDevice()

    Register-ObjectEvent -InputObject $engine -EventName SpeechHypothesized -Action {
        $text = $EventArgs.Result.Text
        if (![string]::IsNullOrWhiteSpace($text)) {
            Write-EventJson -Type "hypothesis" -Text $text -Confidence $EventArgs.Result.Confidence
        }
    } | Out-Null

    Register-ObjectEvent -InputObject $engine -EventName SpeechRecognized -Action {
        $text = $EventArgs.Result.Text
        if (![string]::IsNullOrWhiteSpace($text)) {
            Write-EventJson -Type "result" -Text $text -Confidence $EventArgs.Result.Confidence
        }
    } | Out-Null

    Register-ObjectEvent -InputObject $engine -EventName SpeechRecognitionRejected -Action {
        Write-EventJson -Type "rejected" -Text ""
    } | Out-Null

    Write-EventJson -Type "status" -Text "ready"
    $engine.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)
    while ($true) {
        Start-Sleep -Milliseconds 100
    }
}
catch {
    Write-EventJson -Type "error" -Text "" -Error $_.Exception.Message
    exit 1
}
finally {
    if ($engine) {
        try { $engine.RecognizeAsyncCancel() } catch {}
        try { $engine.Dispose() } catch {}
    }
}
