"""話者分離ベンチ — 埋め込みモデル×クラスタ閾値をTTS合成会話(正解付き)で実測比較する。

使い方: 先に .codex_tmp/diar_bench/ に ja_0..3.wav / en_0..3.wav (16kHz mono) を用意して
    python scripts/koemo_diarize_bench.py
指標: 正解発話フレーム(10ms)に対する最良ラベル対応での一致率 + 検出話者数。
"""
import sys
import wave
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_DIR = ROOT / ".codex_tmp" / "diar_bench"
MODELS = Path.home() / ".koemo" / "models"
SR = 16000
GAP = int(0.7 * SR)

EMB_MODELS = {
    "speakernet": MODELS / "speakernet.onnx",
    "eres2net": MODELS / "eres2net.onnx",
    "wespeaker_resnet34": MODELS / "wespeaker_resnet34.onnx",
}
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def pitch_shift(audio, factor=0.82):
    """簡易ピッチシフト(線形リサンプル)。擬似的な別話者を作る。"""
    idx = np.arange(0, len(audio) - 1, factor)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def build_conversation(utts):
    """utts=[(speaker_id, audio),...] → (mixed_audio, gt_turns[(st,en,spk)])"""
    pieces, turns, t = [], [], 0.0
    for spk, a in utts:
        turns.append((t, t + len(a) / SR, spk))
        pieces.append(a)
        pieces.append(np.zeros(GAP, dtype=np.float32))
        t += len(a) / SR + GAP / SR
    return np.concatenate(pieces), turns


def frame_labels(turns, total_sec, hop=0.01):
    n = int(total_sec / hop)
    lab = np.full(n, -1, dtype=np.int32)
    for st, en, spk in turns:
        lab[int(st / hop):int(en / hop)] = spk
    return lab


def score(gt_turns, hyp_turns, total_sec):
    """正解発話フレームのみで最良置換一致率を返す。"""
    gt = frame_labels(gt_turns, total_sec)
    hyp = frame_labels(hyp_turns, total_sec)
    mask = gt >= 0
    gt_ids = sorted(set(gt[mask].tolist()))
    hyp_ids = sorted(set(int(s) for _, _, s in hyp_turns))
    if not hyp_ids:
        return 0.0
    best = 0.0
    for perm in permutations(hyp_ids, min(len(hyp_ids), len(gt_ids))):
        mapping = {h: g for h, g in zip(perm, gt_ids)}
        mapped = np.array([mapping.get(int(h), -2) for h in hyp[mask]])
        best = max(best, float(np.mean(mapped == gt[mask])))
    return best


def run_diarize(audio, emb_path, threshold):
    import sherpa_onnx
    seg = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_path)),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    res = sd.process(audio).sort_by_start_time()
    return [(s.start, s.end, s.speaker) for s in res]


def main():
    ja = [read_wav(BENCH_DIR / f"ja_{i}.wav") for i in range(4)]
    en = [read_wav(BENCH_DIR / f"en_{i}.wav") for i in range(4)]
    ja_low = [pitch_shift(a) for a in ja]

    cases = {
        "2spk_alt": build_conversation(
            [(0, ja[0]), (1, en[0]), (0, ja[1]), (1, en[1]), (0, ja[2]), (1, en[2])]),
        "2spk_uneven": build_conversation(
            [(0, ja[0]), (0, ja[1]), (1, en[0]), (0, ja[2]), (0, ja[3]), (1, en[1])]),
        "3spk": build_conversation(
            [(0, ja[0]), (1, en[0]), (2, ja_low[1]), (0, ja[2]), (1, en[2]), (2, ja_low[3])]),
        "1spk": build_conversation([(0, ja[0]), (0, ja[1]), (0, ja[2])]),
    }

    print(f"{'model':<20} {'thr':>4} | " + " | ".join(f"{k:>16}" for k in cases))
    for name, path in EMB_MODELS.items():
        if not path.is_file():
            print(f"{name:<20} SKIP (model not found)")
            continue
        for thr in THRESHOLDS:
            cells = []
            for key, (audio, gt) in cases.items():
                total = len(audio) / SR
                hyp = run_diarize(audio, path, thr)
                n_gt = len(set(s for _, _, s in gt))
                n_hyp = len(set(s for _, _, s in hyp))
                acc = score(gt, hyp, total)
                cells.append(f"{acc*100:5.1f}% ({n_hyp}/{n_gt}spk)")
            print(f"{name:<20} {thr:>4} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
