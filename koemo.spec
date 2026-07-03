# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


root = Path(SPECPATH)


def safe_collect_data_files(package):
    try:
        return collect_data_files(package)
    except Exception:
        return []


def safe_collect_submodules(package):
    try:
        return collect_submodules(package)
    except Exception:
        return []


datas = [
    (str(root / "assets" / "koemo.png"), "assets"),
    (str(root / "assets" / "koemo.ico"), "assets"),
    (str(root / "koemo" / "native_speech_bridge.ps1"), "koemo"),
    (str(root / "koemo" / "native_speech_file.ps1"), "koemo"),
    (str(root / "koemo" / "data" / "native_corrections.json"), "koemo/data"),
]
datas += safe_collect_data_files("faster_whisper")
datas += safe_collect_data_files("sherpa_onnx")

hiddenimports = [
    "av",
    "ctranslate2",
    "faster_whisper",
    "huggingface_hub",
    "keyboard",
    "keyboard._canonical_names",
    "keyboard._generic",
    "keyboard._keyboard_event",
    "keyboard._winkeyboard",
    "keyboard._winmouse",
    "numpy",
    "openai",
    "psutil",
    "sherpa_onnx",
    "soundcard",
    "tokenizers",
    "transformers",
    "winrt.windows.globalization",
    "winrt.windows.media.speechrecognition",
]
hiddenimports += safe_collect_submodules("sherpa_onnx")
hiddenimports += safe_collect_submodules("winrt.windows.globalization")
hiddenimports += safe_collect_submodules("winrt.windows.media.speechrecognition")

excludes = [
    "IPython",
    "PIL",
    "Pythonwin",
    "_pytest",
    "_tkinter",
    "aiohttp",
    "datasets",
    "fastapi",
    "matplotlib",
    "mypy",
    "numba",
    "pandas",
    "pyarrow",
    "pytest",
    "pytest_asyncio",
    "pytest_timeout",
    "pytest_xdist",
    "scipy",
    "sklearn",
    "starlette",
    "sympy",
    "tensorflow",
    "tkinter",
    "torch",
    "torchaudio",
    "torchvision",
    "tornado",
    "uvicorn",
]

a = Analysis(
    ["koemo.pyw"],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
a.datas = [
    item for item in a.datas
    if not (
        len(item) >= 1
        and ".dist-info" in str(item[0]).lower()
        and str(item[0]).lower().replace("/", "\\").split("\\")[0].startswith((
            "aiohttp-", "datasets-", "fastapi-", "mypy-", "pandas-", "pillow-",
            "pyarrow-", "pytest_", "scikit_learn-", "scipy-", "starlette-", "torch-",
            "torchaudio-", "torchvision-", "tornado-", "uvicorn-"
        ))
    )
]
a.pure = [
    item for item in a.pure
    if not (
        len(item) >= 1
        and (
            str(item[0]).lower().startswith("_pytest")
            or str(item[0]).lower().endswith(".conftest")
        )
    )
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Koemo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / "assets" / "koemo.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Koemo",
)
