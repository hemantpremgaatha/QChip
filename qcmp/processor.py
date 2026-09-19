import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import datetime

import numpy as np

from .chip import QuantumChip
from .config import ConfigManager
from .thermal import QuantumThermalManager

logger = logging.getLogger("quantum_music_processor")


class SyntheticSource:
    """Generates an A-minor chord (A3, C4, E4) that alternates with D-minor."""

    CHORDS = [(220.0, 261.63, 329.63), (146.83, 220.0, 293.66)]

    def __init__(self, buffer_size, rate, chord_seconds=2.0):
        self.buffer_size, self.rate = buffer_size, rate
        self.chord_chunks = max(1, int(chord_seconds * rate / buffer_size))
        self.n = 0

    def read(self):
        chord = self.CHORDS[(self.n // self.chord_chunks) % len(self.CHORDS)]
        t = (np.arange(self.buffer_size) + self.n * self.buffer_size) / self.rate
        sig = sum(np.sin(2 * np.pi * f * t) for f in chord) / len(chord)
        self.n += 1
        return (sig * 20000).astype(np.int16)

    def close(self):
        pass


class MicSource:
    def __init__(self, buffer_size, rate):
        import pyaudio  # optional dependency
        self.buffer_size = buffer_size
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                                   input=True, frames_per_buffer=buffer_size)

    def read(self):
        return np.frombuffer(self.stream.read(self.buffer_size), dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


class TermuxMicSource:
    """Microphone capture on Android via Termux.

    PyAudio needs PortAudio, which Termux lacks. This shells out to
    `termux-microphone-record` (pkg `termux-api` plus the Termux:API app with
    microphone permission) to record a short clip per read. It is chunked
    near-real-time capture, not low-latency streaming.
    """

    def __init__(self, buffer_size, rate):
        if shutil.which("termux-microphone-record") is None:
            raise RuntimeError(
                "termux-microphone-record not found. Run: pkg install termux-api "
                "and install the Termux:API app, then grant it microphone permission.")
        self.buffer_size, self.rate = buffer_size, rate
        self.tmpdir = tempfile.mkdtemp(prefix="qcmp_")

    def read(self):
        path = os.path.join(self.tmpdir, "clip.wav")
        dur = max(self.buffer_size / self.rate, 0.05)
        quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["termux-microphone-record", "-f", path, "-l", str(dur)], **quiet)
        subprocess.run(["termux-microphone-record", "-q"], **quiet)
        try:
            with wave.open(path, "rb") as wf:
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        except Exception:
            logger.warning("no readable clip this tick")
            data = np.zeros(self.buffer_size, dtype=np.int16)
        finally:
            if os.path.exists(path):
                os.remove(path)
        return np.resize(data, self.buffer_size)

    def close(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class QuantumMusicProcessor:
    def __init__(self, config_path=None, source="synthetic"):
        cm = ConfigManager(config_path) if config_path else ConfigManager()
        self.chip = QuantumChip(cm.get_quantum_config())
        self.thermal = QuantumThermalManager(cm.get_thermal_config())
        self.audio = cm.get_streaming_config()
        self.source_kind = source
        self.start_time = datetime.now()
        logger.info("System started at %s", self.start_time)

    def _open_source(self):
        bs, rate = self.audio["buffer_size"], self.audio["rate"]
        if self.source_kind == "mic":
            return MicSource(bs, rate)
        if self.source_kind == "termux":
            return TermuxMicSource(bs, rate)
        return SyntheticSource(bs, rate)

    async def process_audio(self, max_chunks=None, realtime=True):
        """Main loop. t=0 (the 'origin of time') is the first sample read."""
        src = self._open_source()
        bs, rate = self.audio["buffer_size"], self.audio["rate"]
        chunk_s = bs / rate
        logger.info("Audio stream started")
        audio_time, i, results = 0.0, 0, []
        try:
            while max_chunks is None or i < max_chunks:
                t0 = time.perf_counter()
                data = src.read()
                workload = float(np.abs(data).mean() / 32768 * 4)  # loudness -> heat
                temp = await self.thermal.apply_quantum_cooling(self.chip, min(workload, 1.0))
                res = self.chip.apply_quantum_fourier_transform(data, rate)
                freqs = [(round(f, 1), round(p, 3)) for f, p in res["dominant_frequencies"][:3]]
                logger.info("Audio time: %.3fs, T=%.2f mK, QFT: %s", audio_time, temp, freqs)
                results.append({"t": audio_time, "temp_mk": temp, **res})
                audio_time += chunk_s
                i += 1
                if realtime:
                    await asyncio.sleep(max(0.0, chunk_s - (time.perf_counter() - t0)))
                else:
                    await asyncio.sleep(0)
        finally:
            src.close()
        return results
