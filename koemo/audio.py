"""録音（マイク＋システム音声の2ch同時録音）と AEC（音響エコー除去）。"""
import wave
import threading
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc

from .config import MIC_LABEL, SYS_LABEL


def write_wav(path, sample_rate, audio_i16, channels=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


# ── AEC: ループバック参照を使う周波数領域適応フィルタ ──────────────

def _align_ref(mic, ref, max_lag=8000):
    """mic と ref の遅延を相互相関で推定し、ref を mic に合わせてシフト。"""
    n = min(len(mic), len(ref))
    if n < 2048:
        return ref
    m = mic[:n] - mic[:n].mean()
    r = ref[:n] - ref[:n].mean()
    L = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(m, L) * np.conj(np.fft.rfft(r, L)), L)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    lag = int(np.argmax(np.abs(cc)) - max_lag)
    if lag > 0:
        ref = np.concatenate([np.zeros(lag, dtype=ref.dtype), ref])[:len(ref)]
    elif lag < 0:
        ref = np.concatenate([ref[-lag:], np.zeros(-lag, dtype=ref.dtype)])
    return ref


def cancel_echo(mic, ref, frame=8192, hop=2048, eps=1e-6, res_beta=0.5):
    """ref（システム音声ループバック）のエコーを mic から除去。失敗時は mic を返す。

    停止後の一括処理なので、発散し得る逐次NLMSではなく**閉形式の周波数領域
    Wienerエコーキャンセラ**を使う: 各周波数ビンでエコー経路 H=E[D·conj(X)]/E[|X|²]
    を全フレームの平均から推定し、エコー成分 H·X だけを差し引く。近端（本人の声）
    は ref と無相関なので H 推定を歪めず、残差にそのまま保存される（ダブルトークで
    破綻しない）。さらに残響抑圧（Wiener後置フィルタ）で線形除去後の残エコーを抑える。
    前提: 録音中はマイク/スピーカー位置がほぼ不変＝エコー経路が時間的に一定。
    """
    try:
        n = min(len(mic), len(ref))
        if n < frame * 2:
            return mic
        mic = mic[:n].astype(np.float32)
        ref = _align_ref(mic, ref[:n].astype(np.float32))[:n]
        win = np.hanning(frame).astype(np.float32)
        nfr = 1 + (n - frame) // hop
        nbin = frame // 2 + 1
        BATCH = 256   # フレームをバッチ処理してピークメモリを O(BATCH*frame) に抑える

        # Pass 1: 全フレームを走査してビンごとのクロススペクトル和・参照パワー和を
        # 逐次積算（全STFTを一度に持たない）。近端は無相関なので分子の平均で消える。
        sxx_sum = np.zeros(nbin, dtype=np.complex128)
        sdx_sum = np.zeros(nbin, dtype=np.complex128)
        count = 0
        for b0 in range(0, nfr, BATCH):
            b1 = min(b0 + BATCH, nfr)
            idx = np.arange(frame)[None, :] + hop * np.arange(b0, b1)[:, None]
            Mf = np.fft.rfft(mic[idx] * win, axis=1)
            Rf = np.fft.rfft(ref[idx] * win, axis=1)
            sxx_sum += np.sum(np.abs(Rf) ** 2, axis=0)
            sdx_sum += np.sum(Mf * np.conj(Rf), axis=0)
            count += (b1 - b0)
        # ビンごとの最適エコー経路 H=E[D·conj(X)]/E[|X|²]（全フレーム平均から推定）。
        H = (sdx_sum / count) / ((sxx_sum / count) + eps)

        # Pass 2: 再度バッチ走査して各フレームのエコー H·X を差し引き、残響抑圧して
        # overlap-add。出力バッファだけ全長を確保し、作業領域はバッチ分のみ。
        out = np.zeros(n + frame, dtype=np.float32)
        wsum = np.zeros(n + frame, dtype=np.float32)
        win_sq = win ** 2
        for b0 in range(0, nfr, BATCH):
            b1 = min(b0 + BATCH, nfr)
            idx = np.arange(frame)[None, :] + hop * np.arange(b0, b1)[:, None]
            Mf = np.fft.rfft(mic[idx] * win, axis=1)
            Rf = np.fft.rfft(ref[idx] * win, axis=1)
            Echo = H[None, :] * Rf
            E = Mf - Echo
            # 残響抑圧: エコー推定が支配的なビン/フレームを抑え、近端優勢ビンは保存。
            if res_beta > 0:
                g = np.abs(E) ** 2 / (np.abs(E) ** 2 + res_beta * np.abs(Echo) ** 2 + eps)
                E = E * g
            frames_t = np.fft.irfft(E, axis=1) * win
            for j in range(b1 - b0):
                s = (b0 + j) * hop
                out[s:s + frame] += frames_t[j]
                wsum[s:s + frame] += win_sq
        return (out[:n] / (wsum[:n] + 1e-8)).astype(np.float32)
    except Exception:
        return mic


# ── 2ch同時録音 ──────────────────────────────────────────────

class DualRecorder:
    def __init__(self, cfg):
        self.cfg       = cfg
        self.recording = False
        self.SR        = cfg.get("sample_rate", 16000)
        self._threads  = []
        self._buf      = {}
        self._live     = {}
        self._live_samples = {}
        self._captured_samples = {}
        self._live_max_samples = int(max(30.0, float(cfg.get("live_window_sec", 8.0)) + 5.0) * self.SR)
        self._live_lock = threading.Lock()
        self._temp_lock = threading.Lock()
        self._start_ts = None
        self._temp_files = []
        self._thread_labels = {}

    def _capture_temp_file(self, label):
        suffix = "mic" if label == MIC_LABEL else "system"
        tmp_dir = Path(self.cfg.get("save_dir") or ".") / ".koemo_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=f"capture_{suffix}_", suffix=".f32", dir=str(tmp_dir))
        with self._temp_lock:
            self._temp_files.append(str(path))
        return os.fdopen(fd, "wb"), Path(path)

    def _append_live(self, label, samples):
        """ライブ用リングバッファへ末尾だけ保持し、総録音サンプル数は別に積算する。"""
        if label not in self._live:
            return
        chunk = samples.astype(np.float32, copy=True)
        self._live[label].append(chunk)
        self._live_samples[label] = self._live_samples.get(label, 0) + len(chunk)
        self._captured_samples[label] = self._captured_samples.get(label, 0) + len(chunk)
        while self._live_samples[label] > self._live_max_samples and self._live[label]:
            old = self._live[label].pop(0)
            self._live_samples[label] -= len(old)

    def _capture(self, get_dev, label):
        path = None
        try:
            dev = get_dev()
            samples_written = 0
            f, path = self._capture_temp_file(label)
            with f:
                with dev.recorder(samplerate=self.SR, channels=1, blocksize=1600) as r:
                    while self.recording:
                        chunk = r.record(numframes=1600)
                        samples = chunk[:, 0].astype(np.float32, copy=False)
                        samples.tofile(f)
                        samples_written += len(samples)
                        with self._live_lock:
                            self._append_live(label, samples)
            self._buf[label] = {"path": str(path), "samples": samples_written}
        except Exception as e:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            self._buf[label + "_err"] = str(e)

    def start(self):
        self._buf = {}
        self._live = {MIC_LABEL: [], SYS_LABEL: []}
        self._live_samples = {MIC_LABEL: 0, SYS_LABEL: 0}
        self._captured_samples = {MIC_LABEL: 0, SYS_LABEL: 0}
        self.recording = True
        self._start_ts = datetime.now()
        self._threads = []
        self._thread_labels = {}
        with self._temp_lock:
            self._temp_files = []

        if self.cfg.get("record_mic", True):
            mic_name = self.cfg.get("mic_name") or ""
            getter = ((lambda: sc.get_microphone(mic_name)) if mic_name
                      else (lambda: sc.default_microphone()))
            t = threading.Thread(target=self._capture, args=(getter, MIC_LABEL), daemon=True)
            self._threads.append(t)
            self._thread_labels[t] = MIC_LABEL
        if self.cfg.get("record_system", True):
            spk_name = self.cfg.get("speaker_name") or ""
            def sysget():
                spk = sc.get_speaker(spk_name) if spk_name else sc.default_speaker()
                return sc.get_microphone(spk.name, include_loopback=True)
            t = threading.Thread(target=self._capture, args=(sysget, SYS_LABEL), daemon=True)
            self._threads.append(t)
            self._thread_labels[t] = SYS_LABEL
        for t in self._threads:
            t.start()

    def stop(self):
        self.recording = False
        for t in self._threads:
            t.join(timeout=5)

        # 録音スレッドの例外を _capture が *_err に記録している。停止時に拾って
        # 呼び出し側へ surface する（黙って握り潰さない）。
        capture_errors = {}
        for t in self._threads:
            if t.is_alive():
                label = self._thread_labels.get(t, "audio")
                capture_errors[label] = f"{label} の録音スレッドが停止しませんでした。音声デバイスを確認してください。"
        for label in (MIC_LABEL, SYS_LABEL):
            msg = self._buf.get(label + "_err")
            if msg:
                capture_errors[label] = msg

        chans = {}
        with self._temp_lock:
            temp_files = list(dict.fromkeys(self._temp_files))
        for label in (MIC_LABEL, SYS_LABEL):
            a = self._buf.get(label)
            if isinstance(a, dict) and a.get("path"):
                path = Path(a["path"])
                samples = int(a.get("samples") or 0)
                if str(path) not in temp_files:
                    temp_files.append(str(path))
                if samples > 0 and path.is_file():
                    chans[label] = np.memmap(str(path), dtype=np.float32, mode="r", shape=(samples,))
            elif a is not None and len(a) > 0:
                chans[label] = a
        if not chans:
            ts = (self._start_ts or datetime.now()).strftime("%Y%m%d_%H%M%S")
            self._buf = {}
            with self._live_lock:
                self._live = {}
                self._live_samples = {}
                self._captured_samples = {}
            if capture_errors:
                return {"channels": {}, "duration": 0, "ts": ts, "sr": self.SR,
                        "files": {}, "write_errors": {},
                        "capture_errors": capture_errors,
                        "temp_files": temp_files}
            for path in temp_files:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            return None

        n = min(len(a) for a in chans.values())   # チャンネル長を揃える
        for k in chans:
            chans[k] = chans[k][:n]
        duration = int(n / self.SR)

        if self.cfg.get("enable_aec", True) and MIC_LABEL in chans and SYS_LABEL in chans:
            chans[MIC_LABEL] = cancel_echo(chans[MIC_LABEL], chans[SYS_LABEL])

        ts = self._start_ts.strftime("%Y%m%d_%H%M%S")
        write_errors = {}
        files = {}
        save_dir = Path(self.cfg["save_dir"])
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            write_errors["_save_dir"] = f"{save_dir}: {e}"
            save_dir = None
        for label, a in chans.items():
            if save_dir is None:
                continue
            i16 = np.clip(a * 32767, -32768, 32767).astype(np.int16)
            suffix = "mic" if label == MIC_LABEL else "system"
            wav_path = save_dir / f"recording_{ts}_{suffix}.wav"
            try:
                write_wav(wav_path, self.SR, i16, 1)
                files[label] = str(wav_path)
            except Exception as e:
                write_errors[label] = f"{wav_path}: {e}"

        self._buf = {}
        with self._live_lock:
            self._live = {}
            self._live_samples = {}
            self._captured_samples = {}
        return {"channels": chans, "duration": duration, "ts": ts, "sr": self.SR,
                "files": files,
                "write_errors": write_errors,
                "capture_errors": capture_errors,
                "temp_files": temp_files}

    def elapsed(self):
        return int((datetime.now() - self._start_ts).total_seconds()) if self._start_ts else 0

    def captured_seconds(self, label):
        """ライブプレビュー用に、現在までに取得できた秒数を返す。"""
        with self._live_lock:
            samples = self._captured_samples.get(label)
            if samples is None:
                chunks = self._live.get(label, [])
                samples = sum(len(c) for c in chunks)
        return samples / float(self.SR) if self.SR else 0.0

    def snapshot(self, label, max_seconds=None):
        """ライブプレビュー用に、録音中バッファの連結コピーを返す。"""
        with self._live_lock:
            chunks = list(self._live.get(label, []))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if max_seconds is not None:
            max_samples = int(max_seconds * self.SR)
            if max_samples > 0:
                picked = []
                total = 0
                for chunk in reversed(chunks):
                    picked.append(chunk)
                    total += len(chunk)
                    if total >= max_samples:
                        break
                chunks = list(reversed(picked))
        out = np.concatenate(chunks).astype(np.float32, copy=False)
        if max_seconds is not None and len(out) > int(max_seconds * self.SR):
            out = out[-int(max_seconds * self.SR):]
        return out.copy()


def list_devices():
    """設定UI用: (マイク名リスト, スピーカー名リスト) を返す。"""
    mics, spks = [], []
    try:
        mics = [m.name for m in sc.all_microphones(include_loopback=False)]
    except Exception:
        pass
    try:
        spks = [s.name for s in sc.all_speakers()]
    except Exception:
        pass
    return mics, spks
