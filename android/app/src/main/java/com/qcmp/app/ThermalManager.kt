package com.qcmp.app

import kotlin.random.Random

/** Toy dilution-refrigerator model with a PID-controlled cooling power. */
class ThermalManager(
    val targetMk: Double = 15.0,
    val maxMk: Double = 100.0,
    private val kp: Double = 0.8,
    private val ki: Double = 0.05,
    private val kd: Double = 0.1,
) {
    var temp = targetMk; private set
    var cooling = 0.5; private set
    var helium = 0.0; private set
    private var integral = 0.0
    private var prevErr = 0.0
    private val rng = Random(7)

    fun step(workload: Double): Double {
        val noise = (rng.nextDouble() + rng.nextDouble() + rng.nextDouble() - 1.5) * 0.04
        temp += 0.5 * workload + 0.05 + noise - cooling
        val err = temp - targetMk
        integral = (integral + err).coerceIn(-50.0, 50.0)
        val deriv = err - prevErr
        prevErr = err
        cooling = (0.5 + kp * err + ki * integral + kd * deriv).coerceIn(0.0, 5.0)
        temp = maxOf(temp, 5.0)
        helium += 0.001 * cooling
        return temp
    }
}
