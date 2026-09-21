# QCMP - Quantum Computing Music Processor

Simulated 8-qubit superconducting chip that analyses audio with an amplitude-encoded QFT,
with a PID-controlled cryogenic thermal model.

## Desktop
    pip install -r requirements.txt
    python -m qcmp --chunks 100            # synthetic chord
    python -m qcmp --source mic            # needs pyaudio
    pytest

## Android phone via Termux
Qiskit-Aer is hard to install on Android, so QCMP falls back to a NumPy simulation of the
same QFT when Qiskit is missing.

    pkg install python numpy termux-api
    # install the Termux:API app (F-Droid) and grant it microphone permission
    git clone https://github.com/hemantpremgaatha/QChip && cd QChip
    python -m qcmp --source termux --chunks 100

Use `--source synthetic` to test without the mic. Mic capture is chunked (one short
recording per read), so it is slower than real time.

## Improve an audio file (EQ)
    python -m qcmp.enhance song.wav song_eq.wav --preset vocal
    python -m qcmp.enhance song.mp3 song_eq.mp3 --preset bass --treble 2 --normalize

Presets: `flat`, `bass`, `vocal`, `bright`, `night`, `loud`. Plain NumPy (no scipy). WAV works
out of the box; mp3/m4a/flac need `ffmpeg` (Termux: `pkg install ffmpeg`).

## QSched: scheduling to use your CPU better (phone + PC)
    python -m qcmp.sched info                 # CPU/RAM/battery/power plan
    python -m qcmp.sched bench --tasks 48     # real multiprocess run: round-robin vs LPT vs anneal
    python -m qcmp.sched qaoa                 # simulated QAOA vs classical, small instances
    python -m qcmp.sched offload              # modeled phone<->PC split (assumptions, not measured)

Results are saved to `results/`. Works in Termux (NumPy only). The accompanying paper is in
`paper/QSCHED.md` / `paper/QSCHED.pdf` (rebuild with `python paper/build_pdf.py`). Only the
PC was measured; run `bench` in Termux to add phone numbers.

## Native Android app
See `android/` (Kotlin, offline). Build: `cd android && gradlew assembleDebug`.
