import numpy as np

from qcmp import enhance as E


def tone_db(y, rate, f):
    spec = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y))))
    return 20 * np.log10(spec[int(round(f * len(y) / rate))] + 1e-12)


def test_bass_preset_boosts_low_not_mid(tmp_path):
    rate, n = 44100, 44100 * 2
    t = np.arange(n) / rate
    x = (0.1 * np.sin(2 * np.pi * 80 * t) + 0.1 * np.sin(2 * np.pi * 1000 * t))[:, None]
    src, dst = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    E.write_wav(src, x, rate)
    E.enhance(src, dst, "bass")
    y, r = E.read_wav(dst)
    assert r == rate and y.shape == x.shape
    low = tone_db(y, rate, 80) - tone_db(x, rate, 80)
    mid = tone_db(y, rate, 1000) - tone_db(x, rate, 1000)
    assert 3 < low < 8 and abs(mid) < 1


def test_no_clipping(tmp_path):
    rate = 44100
    t = np.arange(rate) / rate
    x = (0.95 * np.sin(2 * np.pi * 80 * t))[:, None]
    src, dst = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    E.write_wav(src, x, rate)
    E.enhance(src, dst, "bass", bass=6)
    y, _ = E.read_wav(dst)
    assert np.abs(y).max() <= 0.9
