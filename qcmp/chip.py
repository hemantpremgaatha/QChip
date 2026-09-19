import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate, StatePreparation
    from qiskit_aer import AerSimulator
    HAVE_QISKIT = True
except ImportError:  # e.g. Termux/Android, where Qiskit-Aer is hard to install
    HAVE_QISKIT = False


class QuantumChip:
    """Simulated n-qubit superconducting chip running QFT and pattern circuits."""

    def __init__(self, config):
        self.num_qubits = config["num_qubits"]
        self.coherence_time_us = config["coherence_time_us"]
        self.entanglement_fidelity = config["entanglement_fidelity"]
        self.shots = config.get("shots", 4096)
        self.simulator = AerSimulator() if HAVE_QISKIT else None
        self._rng = np.random.default_rng()
        self.error_rates = {"decoherence": 0.0, "gate": 0.0, "readout": 0.0}

    @property
    def num_points(self):
        return 2 ** self.num_qubits

    # -- circuits ---------------------------------------------------------
    def create_qft_circuit(self, samples):
        """Amplitude-encode `samples` (length 2^n) then apply the QFT."""
        vec = np.asarray(samples, dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("cannot encode an all-zero signal")
        qc = QuantumCircuit(self.num_qubits)
        qc.append(StatePreparation(vec / norm), range(self.num_qubits))
        qc.append(QFTGate(self.num_qubits), range(self.num_qubits))
        qc.measure_all()
        return qc

    def create_pattern_circuit(self, features):
        """Encode features as RY rotations, entangle neighbours with CNOTs."""
        qc = QuantumCircuit(self.num_qubits)
        f = np.resize(np.asarray(features, dtype=float), self.num_qubits)
        f = np.pi * (f - f.min()) / (np.ptp(f) or 1.0)
        for i, a in enumerate(f):
            qc.ry(float(a), i)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def execute_circuit(self, circuit):
        compiled = transpile(circuit, self.simulator, optimization_level=1)
        counts = self.simulator.run(compiled, shots=self.shots).result().get_counts()
        total = sum(counts.values())
        return {"status": "success",
                "probabilities": {s: c / total for s, c in counts.items()}}

    # -- algorithms -------------------------------------------------------
    def apply_quantum_fourier_transform(self, audio_data, rate=44100, top_k=5):
        """Return the dominant frequencies (Hz, probability) of an audio chunk.

        The chunk is block-averaged down to 2^n points (a crude low-pass), so
        the effective sample rate is rate / block.
        """
        n = self.num_points
        data = np.asarray(audio_data, dtype=float)
        if len(data) < n:
            data = np.resize(data, n)
        block = len(data) // n
        pts = data[: block * n].reshape(n, block).mean(axis=1)
        pts = pts - pts.mean()  # remove DC so bin 0 does not dominate
        if not np.any(pts):
            return {"status": "silent", "dominant_frequencies": []}
        eff_rate = rate / block
        if HAVE_QISKIT:
            result = self.execute_circuit(self.create_qft_circuit(pts))
        else:
            result = self._numpy_qft(pts)
        # real input -> symmetric spectrum; fold bin k and n-k together
        folded = {}
        for state, p in result["probabilities"].items():
            k = int(state, 2)
            k = min(k, n - k)
            folded[k] = folded.get(k, 0.0) + p
        ranked = sorted(folded.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return {"status": "success",
                "dominant_frequencies": [(k * eff_rate / n, p) for k, p in ranked]}

    def _numpy_qft(self, pts):
        """Qiskit-free path: the QFT unitary applied to the amplitude-encoded
        state (equal to sqrt(N)*ifft), then `shots` measurements sampled."""
        state = np.fft.ifft(pts / np.linalg.norm(pts)) * np.sqrt(len(pts))
        p = np.abs(state) ** 2
        counts = self._rng.multinomial(self.shots, p / p.sum())
        width = self.num_qubits
        return {"status": "success",
                "probabilities": {format(i, f"0{width}b"): c / self.shots
                                  for i, c in enumerate(counts) if c}}

    def recognize_pattern(self, features):
        if not HAVE_QISKIT:
            raise RuntimeError("pattern recognition needs qiskit + qiskit-aer")
        result = self.execute_circuit(self.create_pattern_circuit(features))
        best = max(result["probabilities"].items(), key=lambda kv: kv[1])
        return {"status": "success", "pattern": best[0], "probability": best[1]}

    def update_error_rates(self, temp_mk, optimal_mk=15.0):
        """Error rates grow with temperature above the optimum."""
        excess = max(0.0, temp_mk - optimal_mk) / optimal_mk
        self.error_rates = {"decoherence": 1e-4 * (1 + excess),
                            "gate": 1e-3 * (1 + excess),
                            "readout": 1e-2 * (1 + 0.5 * excess)}
