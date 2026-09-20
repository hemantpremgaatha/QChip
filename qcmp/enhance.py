"""Offline audio EQ: python -m qcmp.enhance in.wav out.wav --preset bass

Pure NumPy (works in Termux). WAV is read natively; other formats (mp3, m4a,
flac...) go through ffmpeg if installed (`pkg install ffmpeg`).
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

# (frequency Hz, gain dB) control points; interpolated on a log-frequency axis.
PRESETS = {
    "flat":   [(20, 0), (20000, 0)],
    "bass":   [(20, 4), (60, 6), (150, 4), (400, 0), (20000, 0)],
    "vocal":  [(20, -3), (150, -2), (400, 0), (1500, 3), (3500, 4), (8000, 1), (20000, 0)],
    "bright": [(20, 0), (2000, 0), (6000, 3), (12000, 5), (20000, 4)],
    "night":  [(20, -6), (100, -3), (300, 0), (3000, 2), (10000, 0), (20000, -3)],
    "loud":   [(20, 4), (80, 5), (400, -1), (2500, 2), (8000, 4), (20000, 3)],
}
TAPS = 4095
BLOCK = 1 << 16


def build_curve(points, extra_bass=0.0, extra_treble=0.0):
    pts = sorted(points)
    f = np.array([p[0] for p in pts], float)
    g = np.array([p[1] for p in pts], float)
    f = np.concatenate([[1.0], f, [1e5]])
    g = np.concatenate([[g[0]], g, [g[-1]]])
    def curve(freqs):
        fr = np.maximum(freqs, 1.0)
        db = np.interp(np.log10(fr), np.log10(f), g)
        db += extra_bass * np.clip(1 - np.log10(fr / 60) / np.log10(300 / 60), 0, 1) * (fr < 300)
        db += extra_treble * np.clip(np.log10(fr / 4000) / np.log10(12000 / 4000), 0, 1) * (fr > 4000)
        return db
    return curve


def design_fir(curve, rate, highpass_hz=25.0):
    """Linear-phase FIR from a dB curve (frequency-sampling method)."""
    n_fft = 1 << 14
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    mag = 10 ** (curve(freqs) / 20)
    if highpass_hz:  # remove inaudible rumble that wastes headroom
        mag *= 1 / np.sqrt(1 + (highpass_hz / np.maximum(freqs, 1e-3)) ** 4)
    ir = np.fft.fftshift(np.fft.irfft(mag, n_fft))
    mid = n_fft // 2
    h = ir[mid - TAPS // 2: mid + TAPS // 2 + 1] * np.hanning(TAPS)
    return h


def apply_fir(x, h):
    """Overlap-add FFT convolution, same length as x (delay compensated)."""
    n_fft = 1 << int(np.ceil(np.log2(BLOCK + len(h) - 1)))
    H = np.fft.rfft(h, n_fft)
    out = np.zeros(len(x) + len(h) - 1)
    for s in range(0, len(x), BLOCK):
        blk = x[s:s + BLOCK]
        y = np.fft.irfft(np.fft.rfft(blk, n_fft) * H, n_fft)[: len(blk) + len(h) - 1]
        out[s:s + len(y)] += y
    d = len(h) // 2
    return out[d:d + len(x)]


def read_wav(path):
    with wave.open(path, "rb") as wf:
        ch, width, rate, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    if width != 2:
        raise ValueError(f"only 16-bit WAV supported natively (got {width * 8}-bit); use ffmpeg")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return x.reshape(-1, ch), rate


def write_wav(path, x, rate):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(x.shape[1]); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def _ffmpeg(*args):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (Termux: pkg install ffmpeg) - or use a 16-bit .wav file")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def enhance(src, dst, preset="vocal", bass=0.0, treble=0.0, normalize=False):
    tmp = tempfile.mkdtemp(prefix="qcmp_eq_")
    try:
        wav_in = src
        if not src.lower().endswith(".wav"):
            wav_in = os.path.join(tmp, "in.wav")
            _ffmpeg("-i", src, "-acodec", "pcm_s16le", wav_in)
        x, rate = read_wav(wav_in)
        h = design_fir(build_curve(PRESETS[preset], bass, treble), rate)
        y = np.stack([apply_fir(x[:, c], h) for c in range(x.shape[1])], axis=1)
        peak = np.abs(y).max()
        if peak > 0.98 or (normalize and peak > 0):  # avoid clipping; optionally maximize
            y *= 0.891 / peak  # -1 dBFS
        if dst.lower().endswith(".wav"):
            write_wav(dst, y, rate)
        else:
            wav_out = os.path.join(tmp, "out.wav")
            write_wav(wav_out, y, rate)
            _ffmpeg("-i", wav_out, dst)
        return {"rate": rate, "channels": x.shape[1], "seconds": len(x) / rate,
                "peak_in": float(np.abs(x).max()), "peak_out": float(np.abs(y).max())}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Apply an EQ preset to an audio file")
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="vocal")
    ap.add_argument("--bass", type=float, default=0.0, help="extra bass dB (below ~300 Hz)")
    ap.add_argument("--treble", type=float, default=0.0, help="extra treble dB (above ~4 kHz)")
    ap.add_argument("--normalize", action="store_true", help="scale peak to -1 dBFS")
    a = ap.parse_args()
    info = enhance(a.input, a.output, a.preset, a.bass, a.treble, a.normalize)
    print(f"{a.input} -> {a.output}  [{a.preset}]  {info['seconds']:.1f}s, "
          f"{info['channels']}ch @ {info['rate']} Hz, peak {info['peak_in']:.2f} -> {info['peak_out']:.2f}")


if __name__ == "__main__":
    main()
