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

## Native Android app
See `android/` (Kotlin, offline). Build: `cd android && gradlew assembleDebug`.
