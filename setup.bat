@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Koemo - セットアップ

echo.
echo   ============================================
echo      Koemo セットアップ
echo   ============================================
echo.

:: ── Pythonチェック ──
python --version >nul 2>&1
if errorlevel 1 (
    echo   [エラー] Pythonが見つかりません。
    echo   https://www.python.org/downloads/ から Python 3.10以上をインストールし、
    echo   「Add Python to PATH」にチェックを入れてください。
    pause
    exit /b 1
)
echo   [OK] Python を確認しました
echo.

echo   [1/4] pip を更新中...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo   [エラー] pip の更新に失敗しました。
    pause
    exit /b 1
)

echo.
echo   [2/4] 必要なライブラリをインストール中...（初回は数分）
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo   [エラー] インストールに失敗しました。インターネット接続を確認してください。
    pause
    exit /b 1
)
echo   [OK] ライブラリ導入完了

echo.
echo   [3/4] AIモデルをダウンロード中...
echo         文字起こし(Whisper large-v3-turbo)・要約(Qwen2.5-3B 3GB)・話者分離(30MB)
echo         ※初回のみ。回線により数分かかります。
echo         ライブ字幕はWindows純正、正式文字起こしはWhisper high accuracyを使います。
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); print('  [OK] 文字起こしモデル large-v3-turbo')"
if errorlevel 1 (
    echo   [エラー] 文字起こしモデル large-v3-turbo の取得に失敗しました。
    pause
    exit /b 1
)
python -c "from huggingface_hub import snapshot_download; snapshot_download('jncraton/Qwen2.5-3B-Instruct-ct2-int8'); print('  [OK] 要約モデル Qwen2.5-3B')"
if errorlevel 1 (
    echo   [エラー] 要約モデル Qwen2.5-3B の取得に失敗しました。
    pause
    exit /b 1
)
python -c "from koemo.diarize import download_diarization_models; download_diarization_models()"
if errorlevel 1 (
    echo   [エラー] 話者分離モデルの取得に失敗しました。
    pause
    exit /b 1
)

echo.
echo   [4/4] GPU高速化を確認中...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [情報] NVIDIA GPU未検出。CPUで動作します（要約はやや時間がかかります）。
) else (
    echo   NVIDIA GPU検出。CUDAランタイムを導入中（約1.2GB、初回のみ）...
    python -m pip install --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12
    if errorlevel 1 (
        echo   [エラー] CUDAランタイムの導入に失敗しました。
        pause
        exit /b 1
    )
    echo   [OK] GPU高速化を有効化しました
)

:: ── デスクトップショートカット ──
set "SCRIPT_DIR=%~dp0"
set "SHORTCUT=%USERPROFILE%\Desktop\Koemo.lnk"
powershell -Command "$ws=New-Object -ComObject WScript.Shell; $sc=$ws.CreateShortcut('%SHORTCUT%'); $sc.TargetPath='pythonw'; $sc.Arguments='\"%SCRIPT_DIR%koemo.pyw\"'; $sc.WorkingDirectory='%SCRIPT_DIR%'; $sc.Description='Koemo'; $sc.Save()" >nul 2>&1

echo.
echo   ============================================
echo      セットアップ完了！
echo   ============================================
echo.
echo   起動: デスクトップの「Koemo」 または start.bat
echo   録音/停止: Ctrl + Shift + R
echo   既定では録音・文字起こし・要約はローカルで動作します（APIキー不要）。
echo   Ollama/OpenAI互換バックエンドを選んだ場合のみ、設定先へ文字起こし/チャット本文を送ります。
echo.
pause
