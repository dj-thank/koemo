"""Zoom/Teams等の会議アプリ検出。"""
import threading

MEETING_PROCESSES = {
    "zoom.exe": "Zoom",
    "teams.exe": "Microsoft Teams",
    "ms-teams.exe": "Microsoft Teams",
    "webex.exe": "Webex",
}


class MeetingWatcher:
    def __init__(self, on_detected, interval_sec=15):
        self.on_detected = on_detected
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self._seen = set()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=1.0):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self):
        try:
            import psutil
        except ImportError:
            return
        while not self._stop.is_set():
            running = set()
            try:
                for proc in psutil.process_iter(["name"]):
                    name = (proc.info.get("name") or "").lower()
                    if name in MEETING_PROCESSES:
                        running.add(name)
            except Exception:
                running = set()

            for name in sorted(running - self._seen):
                self.on_detected(MEETING_PROCESSES[name])
            self._seen = running
            self._stop.wait(self.interval_sec)
