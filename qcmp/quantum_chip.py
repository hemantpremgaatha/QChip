"""Simulated superconducting quantum chip for QCMP.

Executes two circuit families on Qiskit's AerSimulator (falling back to a
probabilistic stand-in if Qiskit isn't installed):

* Quantum Fourier Transform (QFT) over audio samples encoded as qubit
  rotations, used to estimate dominant frequencies.
* A pattern-recognition circuit that entangles feature-encoded qubits with a
  CNOT chain and reports pairwise correlations from the measurement stats.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("quantum_music_processor")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available; QuantumChip will use fallback simulation.")


class QuantumChip:
    """Emulates an N-qubit superconducting chip executing QFT / pattern-recognition circuits."""

    def __init__(self, quantum_config: Dict[str, Any]):
        self.num_qubits: int = quantum_config["num_qubits"]
        self.coherence_time_us: float = quantum_config["coherence_time_us"]
        self.entanglement_fidelity: float = quantum_config["entanglement_fidelity"]
        self.shots: int = quantum_config["shots"]
        self.optimization_level: int = quantum_config["optimization_level"]

        # Error rates start optimistic; QuantumThermalManager updates these with temperature.
        self.decoherence_rate: float = 0.0
        self.gate_error_rate: float = 1.0 - self.entanglement_fidelity
        self.readout_error_rate: float = 0.02

        self._backend = AerSimulator() if QISKIT_AVAILABLE else None

    # ------------------------------------------------------------------ #
    # Circuit construction
    # ------------------------------------------------------------------ #
    def create_quantum_fourier_circuit(self, normalized_data: "np.ndarray | None" = None):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is not installed; cannot build a QuantumCircuit.")

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        if normalized_data is not None:
            for qubit, amplitude in enumerate(normalized_data[: self.num_qubits]):
                qc.ry(float(amplitude) * math.pi, qubit)

        qc.append(QFT(self.num_qubits, do_swaps=True), range(self.num_qubits))
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def create_pattern_recognition_circuit(self, features: np.ndarray):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is not installed; cannot build a QuantumCircuit.")

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        for qubit, value in enumerate(features[: self.num_qubits]):
            qc.rx(float(value) * math.pi, qubit)
        for qubit in range(self.num_qubits - 1):
            qc.cx(qubit, qubit + 1)
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute_circuit(self, circuit) -> Dict[str, Any]:
        if QISKIT_AVAILABLE and circuit is not None:
            try:
                transpiled = transpile(circuit, self._backend, optimization_level=self.optimization_level)
                job = self._backend.run(transpiled, shots=self.shots)
                counts = job.result().get_counts()
                total = sum(counts.values())
                probabilities = {state.replace(" ", ""): count / total for state, count in counts.items()}
                return {"status": "success", "probabilities": probabilities}
            except Exception as exc:  # keep the pipeline alive on backend errors
                logger.error("Circuit execution failed, falling back to simulated outcome: %s", exc)

        return self._fallback_execution()

    def _fallback_execution(self) -> Dict[str, Any]:
        """Generates a plausible probability distribution without Qiskit installed."""
        num_states = 2 ** self.num_qubits
        weights = np.random.exponential(scale=1.0, size=num_states)
        weights /= weights.sum()
        probabilities = {format(i, f"0{self.num_qubits}b"): float(weights[i]) for i in range(num_states)}
        return {"status": "simulated", "probabilities": probabilities}

    # ------------------------------------------------------------------ #
    # Algorithms
    # ------------------------------------------------------------------ #
    def apply_quantum_fourier_transform(self, audio_data: np.ndarray) -> Dict[str, Any]:
        sample_points = 2 ** self.num_qubits

        if len(audio_data) > sample_points:
            indices = np.linspace(0, len(audio_data) - 1, sample_points, dtype=int)
            sampled_data = audio_data[indices]
        else:
            sampled_data = np.resize(audio_data, sample_points)

        peak = np.max(np.abs(sampled_data))
        normalized_data = sampled_data / peak if peak > 0 else sampled_data.astype(float)

        circuit = None
        if QISKIT_AVAILABLE:
            circuit = self.create_quantum_fourier_circuit(normalized_data)

        results = self.execute_circuit(circuit)
        if results["status"] not in ("success", "simulated"):
            return results

        probabilities = results.get("probabilities", {})
        sorted_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        dominant_frequencies: List[Tuple[float, float]] = []
        sample_rate = 44100
        for state, prob in sorted_states[:5]:
            freq_index = int(state, 2) if isinstance(state, str) else state
            freq_value = freq_index * (sample_rate / sample_points)
            dominant_frequencies.append((freq_value, prob))

        return {"status": "success", "dominant_frequencies": dominant_frequencies}

    def apply_pattern_recognition(self, features: np.ndarray) -> Dict[str, Any]:
        sample_points = self.num_qubits
        if len(features) > sample_points:
            indices = np.linspace(0, len(features) - 1, sample_points, dtype=int)
            sampled = features[indices]
        else:
            sampled = np.resize(features, sample_points)

        peak = np.max(np.abs(sampled))
        normalized = sampled / peak if peak > 0 else sampled.astype(float)

        circuit = None
        if QISKIT_AVAILABLE:
            circuit = self.create_pattern_recognition_circuit(normalized)

        results = self.execute_circuit(circuit)
        if results["status"] not in ("success", "simulated"):
            return results

        probabilities = results.get("probabilities", {})
        correlations = self._pairwise_correlations(probabilities)
        return {"status": "success", "correlations": correlations, "probabilities": probabilities}

    def _pairwise_correlations(self, probabilities: Dict[str, float]) -> Dict[Tuple[int, int], float]:
        """Estimates <Z_i Z_j> correlation for each entangled neighbor pair from measurement stats."""
        correlations: Dict[Tuple[int, int], float] = {}
        for qubit in range(self.num_qubits - 1):
            expectation = 0.0
            for state, prob in probabilities.items():
                bit_i = 1 if state[-(qubit + 1)] == "1" else -1
                bit_j = 1 if state[-(qubit + 2)] == "1" else -1
                expectation += prob * bit_i * bit_j
            correlations[(qubit, qubit + 1)] = expectation
        return correlations

    # ------------------------------------------------------------------ #
    # Called by QuantumThermalManager
    # ------------------------------------------------------------------ #
    def update_error_rates(self, temperature_mk: float, optimal_temp_mk: float) -> None:
        """Degrades error rates as temperature rises above the optimal cryogenic point."""
        excess = max(0.0, temperature_mk - optimal_temp_mk)
        self.decoherence_rate = min(1.0, 0.001 * excess)
        self.gate_error_rate = min(1.0, (1.0 - self.entanglement_fidelity) + 0.0005 * excess)
        self.readout_error_rate = min(1.0, 0.02 + 0.0003 * excess)
