"""話者分離（sherpa-onnx, ONNX・torch不要）。チャンネル内の複数話者を 相手1/相手2… に細分化。"""
import glob
from pathlib import Path

_MODELS = Path.home() / ".koemo" / "models"
_sd = None


def _seg_model():
    for n in ("model.onnx", "model.int8.onnx"):
        hits = glob.glob(str(_MODELS / "**" / n), recursive=True)
        if hits:
            return hits[0]
    return None


def _emb_model():
    for pat in ("speakernet.onnx", "*speakerverification*.onnx", "*_sv_*.onnx", "*resnet*.onnx"):
        hits = glob.glob(str(_MODELS / "**" / pat), recursive=True)
        if hits:
            return hits[0]
    return None


def available():
    return bool(_seg_model() and _emb_model())


# scripts/koemo_diarize_bench.py 実測: 閾値0.3は過分割、0.4-0.6が安定帯。中央の0.5を既定にする。
def _get(num_speakers=-1, threshold=0.5):
    global _sd
    if _sd is None:
        import sherpa_onnx
        seg, emb = _seg_model(), _emb_model()
        if not (seg and emb):
            raise RuntimeError("話者分離モデルが見つかりません（~/.koemo/models）")
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg)),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.5,
        )
        _sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return _sd


def diarize(audio):
    """16kHz mono float32 → 話者ターン [(start, end, speaker_int), ...]。失敗時は空。"""
    try:
        sd = _get()
        res = sd.process(audio).sort_by_start_time()
        return [(seg.start, seg.end, seg.speaker) for seg in res]
    except Exception:
        return []


def download_diarization_models():
    """話者分離モデルを ~/.koemo/models へDL・展開（setup用）。"""
    import urllib.request, tarfile
    _MODELS.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
    if not _seg_model():
        seg_tar = _MODELS / "seg.tar.bz2"
        urllib.request.urlretrieve(
            base + "/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", seg_tar)
        with tarfile.open(seg_tar, "r:bz2") as t:
            t.extractall(_MODELS)
    if not _emb_model():
        urllib.request.urlretrieve(
            base + "/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx",
            _MODELS / "speakernet.onnx")
    print("  [OK] 話者分離モデル")


def assign_speakers(segments, turns, base_label):
    """文字起こし [(st,en,txt)] に話者ターン [(st,en,spk)] を時間重複で割当し
    [(st, label, txt)] を返す。複数話者なら base_label+番号、1人/不明なら base_label。
    重複ゼロのセグメント（短い相槌等）は時間的に最も近いターンの話者へ寄せる。"""
    n_spk = len(set(t[2] for t in turns)) if turns else 0
    rows = []
    for (st, en, txt) in segments:
        sp, best = None, 0.0
        for (ts, te, k) in turns:
            ov = max(0.0, min(en, te) - max(st, ts))
            if ov > best:
                best, sp = ov, k
        if sp is None and turns:
            sp = min(turns, key=lambda t: max(t[0] - en, st - t[1], 0.0))[2]
        label = base_label if (sp is None or n_spk <= 1) else f"{base_label}{sp + 1}"
        rows.append((st, label, txt))
    return rows
