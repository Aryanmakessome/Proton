import sounddevice as sd
import numpy as np
import queue
import time
import pvporcupine
from faster_whisper import WhisperModel


class STT:
    def __init__(self):

        # ---------------- WAKE WORD ----------------
        self.porcupine = pvporcupine.create(
            keywords=["computer"]
        )

        self.sample_rate = self.porcupine.sample_rate
        self.frame_length = self.porcupine.frame_length

        # ---------------- WHISPER ----------------
        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

        self.audio_queue = queue.Queue()

    # =====================================================
    # PUBLIC API
    # =====================================================
    def listen(self, max_seconds=10):
        print("[STT] Waiting for wake word...")

        while True:
            self._wait_for_wake()

            print("[STT] Wake detected!")
            audio = self._record_until_silence(max_seconds)

            print("[STT] Transcribing...")
            return self._transcribe(audio)

    # =====================================================
    # WAKE WORD DETECTION
    # =====================================================
    def _wait_for_wake(self):

        def callback(indata, frames, time_info, status):
            self.audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_length,
            callback=callback
        ):
            while True:
                if not self.audio_queue.empty():
                    pcm = self.audio_queue.get().flatten()
                    result = self.porcupine.process(pcm)

                    if result >= 0:
                        return

                time.sleep(0.01)

    # =====================================================
    # SILENCE DETECTION (NO WEBSRTC VAD)
    # =====================================================
    def _is_silence(self, chunk):
        volume = np.abs(chunk).mean()
        return volume < 0.01

    # =====================================================
    # RECORD UNTIL SILENCE
    # =====================================================
    def _record_until_silence(self, max_seconds=10):

        frames = []
        silence_counter = 0
        max_silence = 15  # tweakable (~0.5 sec)

        def callback(indata, frames_count, time_info, status):
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            callback=callback
        ):
            start = time.time()

            while True:
                if time.time() - start > max_seconds:
                    break

                if len(frames) > 0:
                    chunk = frames[-1]

                    if self._is_silence(chunk):
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    if silence_counter > max_silence:
                        break

                time.sleep(0.01)

        return np.concatenate(frames, axis=0) if frames else np.array([])

    # =====================================================
    # WHISPER TRANSCRIPTION
    # =====================================================
    def _transcribe(self, audio):

        if len(audio) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio,
            language="en"
        )

        return " ".join([s.text for s in segments]).strip().lower()


# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":
    stt = STT()

    while True:
        text = stt.listen()
        print("You said:", text)