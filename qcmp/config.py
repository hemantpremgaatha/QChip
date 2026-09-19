import configparser
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.ini"


class ConfigManager:
    """Loads quantum / thermal / streaming settings from an INI file."""

    def __init__(self, path=DEFAULT_PATH):
        self.parser = configparser.ConfigParser()
        if not self.parser.read(path):
            raise FileNotFoundError(f"config file not found: {path}")

    def get_quantum_config(self):
        s = self.parser["quantum"]
        return {
            "num_qubits": s.getint("num_qubits"),
            "coherence_time_us": s.getfloat("coherence_time_us"),
            "entanglement_fidelity": s.getfloat("entanglement_fidelity"),
            "shots": s.getint("shots"),
        }

    def get_thermal_config(self):
        s = self.parser["thermal"]
        return {k: s.getfloat(k) for k in
                ("optimal_temp_mk", "max_temp_mk", "kp", "ki", "kd")}

    def get_streaming_config(self):
        s = self.parser["streaming"]
        return {"buffer_size": s.getint("buffer_size"), "rate": s.getint("rate")}
