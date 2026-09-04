"""Configuration management for the Quantum Computing Music Processor (QCMP)."""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict, Union

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.ini"

_DEFAULTS = {
    "quantum": {
        "num_qubits": "8",
        "coherence_time_us": "150",
        "entanglement_fidelity": "0.98",
        "backend": "aer_simulator",
        "shots": "1024",
        "optimization_level": "1",
    },
    "thermal": {
        "optimal_temp_mk": "15",
        "max_temp_mk": "100",
        "pid_kp": "0.6",
        "pid_ki": "0.08",
        "pid_kd": "0.02",
        "algorithm": "quantum_adaptive_cooling",
    },
    "streaming": {
        "buffer_size": "1024",
        "rate": "44100",
        "channels": "1",
        "format_bits": "16",
    },
}


class ConfigManager:
    """Loads QCMP settings from an INI file, falling back to built-in defaults."""

    def __init__(self, config_path: Union[str, Path, None] = None):
        self._parser = configparser.ConfigParser()
        self._parser.read_dict(_DEFAULTS)
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if path.exists():
            self._parser.read(path)

    def get_quantum_config(self) -> Dict:
        s = self._parser["quantum"]
        return {
            "num_qubits": s.getint("num_qubits"),
            "coherence_time_us": s.getfloat("coherence_time_us"),
            "entanglement_fidelity": s.getfloat("entanglement_fidelity"),
            "backend": s.get("backend"),
            "shots": s.getint("shots"),
            "optimization_level": s.getint("optimization_level"),
        }

    def get_thermal_config(self) -> Dict:
        s = self._parser["thermal"]
        return {
            "optimal_temp_mk": s.getfloat("optimal_temp_mk"),
            "max_temp_mk": s.getfloat("max_temp_mk"),
            "pid_kp": s.getfloat("pid_kp"),
            "pid_ki": s.getfloat("pid_ki"),
            "pid_kd": s.getfloat("pid_kd"),
            "algorithm": s.get("algorithm"),
        }

    def get_streaming_config(self) -> Dict:
        s = self._parser["streaming"]
        return {
            "buffer_size": s.getint("buffer_size"),
            "rate": s.getint("rate"),
            "channels": s.getint("channels"),
            "format_bits": s.getint("format_bits"),
        }
