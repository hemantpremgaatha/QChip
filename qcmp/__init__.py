"""Quantum Computing Music Processor (QCMP) - simulated superconducting-qubit audio analysis."""
from .config import ConfigManager
from .chip import QuantumChip
from .thermal import QuantumThermalManager
from .processor import QuantumMusicProcessor

__all__ = ["ConfigManager", "QuantumChip", "QuantumThermalManager", "QuantumMusicProcessor"]
