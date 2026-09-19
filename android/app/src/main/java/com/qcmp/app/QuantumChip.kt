package com.qcmp.app

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log2
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

data class Peak(val hz: Double, val prob: Double) {
    val note: String
        get() {
            if (hz < 20) return "-"
            val midi = (69 + 12 * log2(hz / 440.0)).roundToInt()
            val names = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
            return names[((midi % 12) + 12) % 12] + (midi / 12 - 1)
        }
}

class QftResult(val spectrum: DoubleArray, val peaks: List<Peak>)

/** Simulated n-qubit chip: state-vector simulator running an amplitude-encoded QFT. */
class QuantumChip(val numQubits: Int = 8, private val shots: Int = 4096) {
    val size = 1 shl numQubits
    private val rng = Random(1)

    // Error rates shown in the UI; they grow as the fridge warms.
    var gateError = 1e-3
    var readoutError = 1e-2

    fun updateErrorRates(tempMk: Double, optimalMk: Double) {
        val excess = maxOf(0.0, tempMk - optimalMk) / optimalMk
        gateError = 1e-3 * (1 + excess)
        readoutError = 1e-2 * (1 + 0.5 * excess)
    }

    /** Gate-level QFT (H + controlled phases + bit-reversal swaps) on a state vector. */
    private fun qft(re: DoubleArray, im: DoubleArray) {
        val n = numQubits
        for (q in n - 1 downTo 0) {
            hadamard(re, im, q)
            for (c in q - 1 downTo 0) controlledPhase(re, im, c, q, PI / (1 shl (q - c)))
        }
        for (i in 0 until n / 2) swap(re, im, i, n - 1 - i)
    }

    private fun hadamard(re: DoubleArray, im: DoubleArray, q: Int) {
        val bit = 1 shl q
        val s = 1.0 / sqrt(2.0)
        for (i in 0 until size) if (i and bit == 0) {
            val j = i or bit
            val ar = re[i]; val ai = im[i]; val br = re[j]; val bi = im[j]
            re[i] = (ar + br) * s; im[i] = (ai + bi) * s
            re[j] = (ar - br) * s; im[j] = (ai - bi) * s
        }
    }

    private fun controlledPhase(re: DoubleArray, im: DoubleArray, c: Int, t: Int, theta: Double) {
        val mask = (1 shl c) or (1 shl t)
        val cs = cos(theta); val sn = sin(theta)
        for (i in 0 until size) if (i and mask == mask) {
            val r = re[i]; val m = im[i]
            re[i] = r * cs - m * sn; im[i] = r * sn + m * cs
        }
    }

    private fun swap(re: DoubleArray, im: DoubleArray, a: Int, b: Int) {
        for (i in 0 until size) {
            val ba = (i shr a) and 1; val bb = (i shr b) and 1
            if (ba == 0 && bb == 1) {
                val j = i xor (1 shl a) xor (1 shl b)
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
    }

    /** Returns null for silence. The chunk is block-averaged down to 2^n points. */
    fun quantumFourierTransform(chunk: ShortArray, rate: Int, topK: Int = 3): QftResult? {
        val block = maxOf(1, chunk.size / size)
        val pts = DoubleArray(size) { i ->
            var s = 0.0
            for (k in 0 until block) s += chunk[(i * block + k).coerceAtMost(chunk.size - 1)]
            s / block
        }
        val mean = pts.average()
        var norm = 0.0
        for (i in pts.indices) { pts[i] -= mean; norm += pts[i] * pts[i] }
        norm = sqrt(norm)
        if (norm < 1e-9) return null

        // amplitude encoding, then QFT
        val re = DoubleArray(size) { pts[it] / norm }
        val im = DoubleArray(size)
        qft(re, im)

        // measurement: sample `shots` outcomes from |amplitude|^2
        val cdf = DoubleArray(size)
        var acc = 0.0
        for (i in 0 until size) { acc += re[i] * re[i] + im[i] * im[i]; cdf[i] = acc }
        val counts = IntArray(size)
        repeat(shots) {
            val r = rng.nextDouble() * acc
            var lo = 0; var hi = size - 1
            while (lo < hi) { val m = (lo + hi) / 2; if (cdf[m] < r) lo = m + 1 else hi = m }
            counts[lo]++
        }

        // real input -> symmetric spectrum: fold bin k and n-k
        val half = size / 2
        val spectrum = DoubleArray(half + 1)
        for (k in 0 until size) spectrum[min(k, size - k)] += counts[k].toDouble() / shots
        val effRate = rate.toDouble() / block
        val peaks = (1..half).sortedByDescending { spectrum[it] }.take(topK)
            .map { Peak(it * effRate / size, spectrum[it]) }
        return QftResult(spectrum, peaks)
    }
}
