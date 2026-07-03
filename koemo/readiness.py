"""初回起動・配布版向けのモデル準備状態チェック。"""
from pathlib import Path


def _hf_hub_dir(hub_dir=None):
    return Path(hub_dir) if hub_dir else Path.home() / ".cache" / "huggingface" / "hub"


def _diarization_models_dir(models_dir=None):
    return Path(models_dir) if models_dir else Path.home() / ".koemo" / "models"


def whisper_model_available(model_name, hub_dir=None):
    """faster-whisper のHFキャッシュが存在するかを、DLなしで軽く判定する。"""
    name = (model_name or "").strip().lower()
    if not name:
        return False
    hub = _hf_hub_dir(hub_dir)
    if not hub.is_dir():
        return False
    candidates = []
    if "/" in name:
        owner, repo = name.split("/", 1)
        candidates.append(f"models--{owner}--{repo}")
    candidates.extend([
        f"models--*faster-whisper*{name}*",
        f"models--*whisper*{name}*",
    ])
    for pattern in candidates:
        for root in hub.glob(pattern):
            for snap in (root / "snapshots").glob("*"):
                if (snap / "model.bin").is_file() or (snap / "config.json").is_file():
                    return True
    return False


def diarization_models_available(models_dir=None):
    base = _diarization_models_dir(models_dir)
    if not base.is_dir():
        return False
    seg = list(base.glob("**/model.onnx")) + list(base.glob("**/model.int8.onnx"))
    emb_patterns = ("**/speakernet.onnx", "**/*speakerverification*.onnx", "**/*_sv_*.onnx", "**/*resnet*.onnx")
    emb = []
    for pattern in emb_patterns:
        emb.extend(base.glob(pattern))
    return bool(seg and emb)


def model_statuses(cfg, hub_dir=None, models_dir=None):
    """配布版の初回準備状態を返す。DLは行わず、ユーザー案内だけに使う。"""
    statuses = []
    native_only = cfg.get("native_only_transcription", False)
    whisper_model = cfg.get("whisper_model", "large-v3-turbo")
    if native_only:
        statuses.append({
            "name": "正式文字起こし",
            "ok": True,
            "required": False,
            "message": "Windows純正のみモードです。",
        })
    else:
        ok = whisper_model_available(whisper_model, hub_dir=hub_dir)
        statuses.append({
            "name": f"Whisper {whisper_model}",
            "ok": ok,
            "required": True,
            "message": "初回の正式文字起こし時に自動取得を試みます。" if not ok else "準備済みです。",
        })

    summary_backend = cfg.get("summary_backend", "local")
    full_llm_summary = not cfg.get("fast_summary", True)
    if summary_backend == "local":
        from .backends import find_summary_model

        ok = bool(find_summary_model(cfg.get("summary_model_dir", "")))
        statuses.append({
            "name": "ローカルLLM要約モデル",
            "ok": ok or not full_llm_summary,
            "required": full_llm_summary,
            "message": (
                "詳細LLM要約を使う場合は setup.bat で取得してください。"
                if not ok else "準備済みです。"
            ),
        })
    else:
        statuses.append({
            "name": "リモート要約バックエンド",
            "ok": True,
            "required": False,
            "message": "要約/チャット本文は設定先へ送信されます。",
        })

    diar_ok = diarization_models_available(models_dir=models_dir)
    statuses.append({
        "name": "話者分離モデル",
        "ok": diar_ok or not cfg.get("enable_diarization", True),
        "required": False,
        "message": "未取得でも録音・文字起こしは継続します。" if not diar_ok else "準備済みです。",
    })
    return statuses


def first_run_notice(cfg, hub_dir=None, models_dir=None):
    missing = [s for s in model_statuses(cfg, hub_dir=hub_dir, models_dir=models_dir) if not s["ok"]]
    if not missing:
        return ""
    lines = ["初回モデル準備が必要です。"]
    for item in missing[:3]:
        lines.append(f"- {item['name']}: {item['message']}")
    lines.append("ネットワーク接続後に setup.bat を実行すると事前取得できます。")
    return "\n".join(lines)


def model_load_error(kind, name, error):
    return (
        f"{kind}モデル `{name}` を準備できませんでした。\n"
        "初回起動ではモデル取得にネットワーク接続と数分の時間が必要です。\n"
        "復旧手順: Koemoフォルダの setup.bat を実行し、モデル取得が完了してから再起動してください。\n"
        f"詳細: {error}"
    )
