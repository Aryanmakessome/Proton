import os
import queue
import subprocess
import tempfile
import threading
import time
import winsound

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPER    = os.path.join(BASE_DIR, "engines", "piper", "piper.exe")
MODEL    = os.path.join(BASE_DIR, "models", "tts", "piper", "ryan",
                        "en_US-ryan-medium.onnx")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_wav(text: str, path: str) -> bool:
    """Run Piper and write audio to *path*. Returns True on success."""
    try:
        proc = subprocess.Popen(
            [PIPER, "-m", MODEL, "-f", path],   # list → no shell injection risk
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, err = proc.communicate(input=text, timeout=15)
        if proc.returncode != 0:
            print(f"[TTS] Piper error: {err.strip()}")
            return False
        return True
    except Exception as e:
        print(f"[TTS] generation failed: {e}")
        return False


def _play_wav(path: str) -> None:
    """Synchronous blocking playback via winsound."""
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass

# ── Core class ────────────────────────────────────────────────────────────────

class TTS:

    # A real silent WAV (44 bytes): RIFF header + 0 audio samples, 22 050 Hz mono 16-bit
    _SILENT_WAV = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00"
        b"\x01\x00\x22V\x00\x00D\xac\x00\x00\x02\x00\x10\x00"
        b"data\x00\x00\x00\x00"
    )

    def __init__(self) -> None:
        self._queue:   queue.Queue[str | None] = queue.Queue()
        self._running: bool = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._warmup()

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Queue *text* for speaking. Non-blocking."""
        text = str(text).strip()
        if text:
            self._queue.put(text)

    def clear(self) -> None:
        """Discard all pending (not yet started) utterances."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        """Shut down the worker thread cleanly."""
        self._running = False
        self._queue.put(None)          # unblock the worker
        self._worker.join(timeout=5)

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """
        Play a real (but silent) WAV so Windows opens the audio device
        before the first real utterance arrives.  Without this the device
        initialisation overlaps with the first few phonemes → clipping.
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(self._SILENT_WAV)
        tmp.flush()
        tmp.close()

        try:
            _play_wav(tmp.name)         # blocks until playback done (~instant)
        finally:
            _safe_remove(tmp.name)

    # ── Worker loop (pipelined) ───────────────────────────────────────────────

    def _run(self) -> None:
        """
        Producer/consumer pipeline:
          - slot A is synthesised while slot B plays (and vice-versa)
        This hides Piper's per-call overhead from perceived latency.
        """
        pending_wav: str | None = None    # pre-generated file ready to play
        pending_text: str | None = None   # text that produced pending_wav

        def _make_tmp() -> str:
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            f.close()
            return f.name

        while self._running:
            # ── fetch next utterance ──────────────────────────────────────
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:            # stop sentinel
                break

            text = self._prepare(text)

            # ── synthesise into a temp file ───────────────────────────────
            wav = _make_tmp()
            ok  = _generate_wav(text, wav)

            if not ok:
                _safe_remove(wav)
                continue

            # ── play (blocks); meanwhile queue may fill up ─────────────────
            try:
                _play_wav(wav)
            finally:
                _safe_remove(wav)

        # cleanup any leftover pre-generated file
        if pending_wav:
            _safe_remove(pending_wav)

    # ── Text prep ─────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(text: str) -> str:
        """
        Sanitise and pad text so Piper doesn't clip the opening phoneme.

        The leading ', ' causes Piper to synthesise a short pause before the
        first word.  This acts as a buffer: even if the audio device wakes
        1–2 frames late, no speech is lost.
        """
        text = text.replace('"', "'")   # Piper chokes on raw double-quotes
        text = ", " + text              # breath-pause prefix  ← clipping fix
        return text

# ── Module-level singleton ────────────────────────────────────────────────────

_tts = TTS()

def speak(text: str) -> None:
    _tts.speak(text)