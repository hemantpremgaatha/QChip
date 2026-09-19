import asyncio

import numpy as np

from qcmp import ConfigManager, QuantumChip, QuantumMusicProcessor


def chip():
    return QuantumChip(ConfigManager().get_quantum_config())


def test_qft_finds_tone():
    rate, n = 44100, 1024
    t = np.arange(n) / rate
    res = chip().apply_quantum_fourier_transform(np.sin(2 * np.pi * 1000 * t) * 1e4, rate)
    top = res["dominant_frequencies"][0][0]
    assert abs(top - 1000) < 60  # bin width ~43 Hz


def test_silence():
    assert chip().apply_quantum_fourier_transform(np.zeros(1024))["status"] == "silent"


def test_pattern():
    assert chip().recognize_pattern([0.1, 0.9, 0.3, 0.5])["status"] == "success"


def test_pipeline_temperature_stable():
    p = QuantumMusicProcessor()
    r = asyncio.run(p.process_audio(30, realtime=False))
    assert len(r) == 30
    assert all(x["temp_mk"] < 100 for x in r)
