"""Real-time (or file/synthetic) audio pipeline that drives the QuantumChip and thermal manager."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config_manager import ConfigManager
from .quantum_chip import QuantumChip
from .thermal_manager import QuantumThermalManager

logger = logging.getLogger("quantum_music_processor")

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYAUDIO_AVAILABLE = False


class AudioSource:
    """Common interface: read(n_samples) -> int16 ndarray, or None at end of stream."""

    def read(self, n_samples: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MicrophoneSource(AudioSource):
    def __init__(self, rate: int, channels: int, format_bits: int):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is not installed; cannot open a microphone stream.")
        self._pa = pyaudio.PyAudio()
        sample_format = pyaudio.paInt16 if format_bits == 16 else pyaudio.paInt32
        self._stream = self._pa.open(
            format=sample_format, channels=channels, rate=rate, input=True, frames_per_buffer=1024
        )

    def read(self, n_samples: int) -> Optional[np.ndarray]:
        raw = self._stream.read(n_samples, exception_on_overflow=False)
        return np.frombuffer(raw, dtype=np.int16)

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


class TermuxMicrophoneSource(AudioSource):
    """Microphone capture for Android via Termux.

    PyAudio needs PortAudio, which isn't available under Termux — Android
    also sandboxes microphone access differently than desktop OSes. This
    shells out to `termux-microphone-record` (from the `termux-api` pkg,
    paired with the Termux:API companion app, which must be granted the
    RECORD_AUDIO permission in Android's app settings) to record a short
    clip per call, matching the AudioSource.read(n_samples) interface. It's
    chunked near-real-time capture, not continuous low-latency streaming —
    the practical adaptation for what Android actually allows.
    """

    def __init__(self, rate: int):
        if shutil.which("termux-microphone-record") is None:
            raise RuntimeError(
                "termux-microphone-record not found. Install with: "
                "pkg install termux-api  (and install the Termux:API app "
                "from F-Droid, then grant it microphone permission)."
            )
        self._rate = rate
        self._tmpdir = tempfile.mkdtemp(prefix="qcmp_")

    def read(self, n_samples: int) -> Optional[np.ndarray]:
        path = os.path.join(self._tmpdir, "clip.wav")
        duration_s = max(n_samples / self._rate, 0.05)
        subprocess.run(
            ["termux-microphone-record", "-f", path, "-l", str(duration_s)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        # -l auto-stops after duration_s, but an explicit -q keeps the
        # recorder process from lingering if it didn't exit on its own.
        subprocess.run(["termux-microphone-record", "-q"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        try:
            with wave.open(path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                data = np.frombuffer(raw, dtype=np.int16)
        except Exception:
            logger.warning("termux-microphone-record produced no readable clip this tick.")
            data = np.zeros(n_samples, dtype=np.int16)
        finally:
            if os.path.exists(path):
                os.remove(path)
        return np.resize(data, n_samples)

    def close(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class WavFileSource(AudioSource):
    def __init__(self, path: Union[str, Path]):
        self._wf = wave.open(str(path), "rb")

    def read(self, n_samples: int) -> Optional[np.ndarray]:
        raw = self._wf.readframes(n_samples)
        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.int16)

    def close(self) -> None:
        self._wf.close()


class SyntheticSource(AudioSource):
    """Generates a sweeping tone; useful for testing without a microphone or WAV file."""

    def __init__(self, rate: int, duration_s: float = 5.0, base_freq: float = 220.0):
        self._rate = rate
        self._t = 0.0
        self._duration_s = duration_s
        self._base_freq = base_freq

    def read(self, n_samples: int) -> Optional[np.ndarray]:
        if self._t >= self._duration_s:
            return None
        t = np.arange(n_samples) / self._rate + self._t
        freq = self._base_freq * (1 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
        signal = 0.6 * np.sin(2 * np.pi * freq * t)
        self._t += n_samples / self._rate
        return (signal * 32767).astype(np.int16)


class QuantumMusicProcessor:
    """Ties ConfigManager, QuantumChip, and QuantumThermalManager into an async audio loop."""

    def __init__(self, config_path: Optional[str] = None, audio_source: Optional[AudioSource] = None):
        self.config_manager = ConfigManager(config_path)
        self.quantum_chip = QuantumChip(self.config_manager.get_quantum_config())
        self.thermal_manager = QuantumThermalManager(self.config_manager.get_thermal_config())
        self.audio_config = self.config_manager.get_streaming_config()
        self._audio_source = audio_source
        self.start_time = datetime.now()
        logger.info("System started at %s", self.start_time)

    def _default_source(self) -> AudioSource:
        if PYAUDIO_AVAILABLE:
            return MicrophoneSource(
                self.audio_config["rate"], self.audio_config["channels"], self.audio_config["format_bits"]
            )
        if shutil.which("termux-microphone-record") is not None:
            logger.info("PyAudio unavailable; using Termux microphone capture.")
            return TermuxMicrophoneSource(self.audio_config["rate"])
        logger.warning("No microphone backend available; falling back to a synthetic test tone.")
        return SyntheticSource(self.audio_config["rate"])

    async def process_audio(self, max_chunks: Optional[int] = None) -> None:
        source = self._audio_source or self._default_source()
        logger.info("Audio stream started")
        audio_time = 0.0
        buffer_size = self.audio_config["buffer_size"]
        rate = self.audio_config["rate"]
        chunks_processed = 0

        try:
            while max_chunks is None or chunks_processed < max_chunks:
                data = source.read(buffer_size)
                if data is None:
                    logger.info("Audio stream ended")
                    break

                temp, workload = await self.thermal_manager.apply_quantum_cooling(self.quantum_chip)

                if workload > 0.5:
                    qft_result = self.quantum_chip.apply_quantum_fourier_transform(data)
                    if qft_result["status"] == "success":
                        logger.info(
                            "Audio time: %.3fs, Temp: %.2f mK, QFT dominant frequencies: %s",
                            audio_time,
                            temp,
                            qft_result["dominant_frequencies"],
                        )

                audio_time += buffer_size / rate
                chunks_processed += 1
                await asyncio.sleep(0.1)
        finally:
            source.close()
