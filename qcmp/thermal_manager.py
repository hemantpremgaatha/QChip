"""Cryogenic thermal management simulation for QCMP (dilution-refrigerator emulation)."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, Tuple

logger = logging.getLogger("quantum_music_processor")


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, measurement: float, dt: float = 1.0) -> float:
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class QuantumThermalManager:
    """Simulates dilution-refrigerator dynamics: PID cooling, helium use, and thermal noise."""

    def __init__(self, thermal_config: Dict[str, Any]):
        self.optimal_temp_mk: float = thermal_config["optimal_temp_mk"]
        self.max_temp_mk: float = thermal_config["max_temp_mk"]
        self.algorithm: str = thermal_config["algorithm"]

        self._pid = PIDController(
            thermal_config["pid_kp"],
            thermal_config["pid_ki"],
            thermal_config["pid_kd"],
            setpoint=self.optimal_temp_mk,
        )

        self.current_temp_mk: float = self.optimal_temp_mk
        self.helium_consumed_l: float = 0.0

    async def apply_quantum_cooling(self, quantum_chip) -> Tuple[float, float]:
        """Advances the thermal model by one control step and updates chip error rates.

        Returns (current_temperature_mk, workload), where workload in [0, 1] models
        the computational/heat load on the chip for this tick.
        """
        workload = min(1.0, 0.4 + 0.6 * random.random())
        heat_leak = workload * (0.15 * quantum_chip.num_qubits)
        thermal_noise = random.gauss(0, 0.3)

        self.current_temp_mk += heat_leak + thermal_noise

        cooling_power = max(0.0, self._pid.step(self.current_temp_mk))
        self.current_temp_mk -= cooling_power
        self.current_temp_mk = max(self.optimal_temp_mk * 0.9, self.current_temp_mk)

        if self.algorithm == "quantum_adaptive_cooling":
            overshoot = max(0.0, self.current_temp_mk - self.optimal_temp_mk)
            self.current_temp_mk -= 0.3 * overshoot

        self.helium_consumed_l += 0.002 * max(cooling_power, 0.1)

        if self.current_temp_mk > self.max_temp_mk:
            logger.warning(
                "Chip temperature %.2f mK exceeds max %.2f mK", self.current_temp_mk, self.max_temp_mk
            )

        quantum_chip.update_error_rates(self.current_temp_mk, self.optimal_temp_mk)

        await asyncio.sleep(0)  # yields control, mirroring async I/O to a real fridge controller
        return self.current_temp_mk, workload
