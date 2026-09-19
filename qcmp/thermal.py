import numpy as np


class QuantumThermalManager:
    """Toy dilution-refrigerator model with a PID-controlled cooling power."""

    def __init__(self, config, seed=None):
        self.target = config["optimal_temp_mk"]
        self.max_temp = config["max_temp_mk"]
        self.kp, self.ki, self.kd = config["kp"], config["ki"], config["kd"]
        self.temp = self.target
        self.cooling = 0.5
        self.helium_used = 0.0
        self._integral = 0.0
        self._prev_err = 0.0
        self.rng = np.random.default_rng(seed)

    def step(self, workload, dt=1.0):
        """Advance one tick. Workload (0-1) heats the chip; PID cools it."""
        heat = 0.5 * workload + 0.05  # mK per tick
        noise = self.rng.normal(0, 0.02)
        self.temp += heat + noise - 1.0 * self.cooling
        err = self.temp - self.target
        self._integral = float(np.clip(self._integral + err * dt, -50, 50))
        deriv = (err - self._prev_err) / dt
        self._prev_err = err
        self.cooling = float(np.clip(
            0.5 + self.kp * err + self.ki * self._integral + self.kd * deriv, 0, 5))
        self.temp = max(self.temp, 5.0)
        self.helium_used += 0.001 * self.cooling
        return self.temp

    async def apply_quantum_cooling(self, chip, workload):
        temp = self.step(workload)
        chip.update_error_rates(temp, self.target)
        if temp > self.max_temp:
            raise RuntimeError(f"chip overheated: {temp:.1f} mK > {self.max_temp} mK")
        return temp
