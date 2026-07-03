"""GPU(CUDA)サポート — CUDAランタイムDLLの検索パス追加と利用可否判定。"""
import os


def enable_cuda_dlls():
    """pip の nvidia-* 由来 cuBLAS/cuDNN(DLL) を検索パスへ追加（GPU実行に必要）。

    CTranslate2 / faster-whisper のCUDA実行には cublas64_12.dll 等が必要だが、
    os.add_dll_directory だけでは内部ローダーが見つけられないため PATH にも追加する。
    無い環境では何もしない（CPUで動作）。
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        root = list(spec.submodule_search_locations)[0]
        bins = [os.path.join(root, d, "bin") for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d, "bin"))]
        if bins:
            os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
            for b in bins:
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass
    except Exception:
        pass


_GPU_OK = None


def gpu_ok():
    """CUDA GPUが使えて cuBLAS がロード可能か（一度だけ判定してキャッシュ）。"""
    global _GPU_OK
    if _GPU_OK is None:
        _GPU_OK = False
        try:
            import ctranslate2, ctypes
            if ctranslate2.get_cuda_device_count() > 0:
                ctypes.WinDLL("cublas64_12.dll")
                _GPU_OK = True
        except Exception:
            _GPU_OK = False
    return _GPU_OK
