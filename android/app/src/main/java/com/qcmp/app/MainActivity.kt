package com.qcmp.app

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class MainActivity : Activity() {
    private val rate = 44100
    private val bufferSize = 1024
    private val chip = QuantumChip()
    private val thermal = ThermalManager()

    private lateinit var status: TextView
    private lateinit var notes: TextView
    private lateinit var spectrum: SpectrumView
    private lateinit var micBtn: Button
    private lateinit var demoBtn: Button
    @Volatile private var running = false
    private var worker: Thread? = null

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(0x0B, 0x0B, 0x12))
            setPadding(pad, pad * 2, pad, pad)
        }
        root.addView(TextView(this).apply {
            text = "Quantum Music Processor"; textSize = 22f; setTextColor(Color.WHITE)
        })
        root.addView(TextView(this).apply {
            text = "8 simulated superconducting qubits - QFT on live audio"
            textSize = 13f; setTextColor(Color.LTGRAY)
        })
        status = TextView(this).apply {
            textSize = 15f; setTextColor(Color.rgb(0x80, 0xDE, 0xEA)); setPadding(0, pad, 0, pad)
            text = "Idle. Chip at ${thermal.targetMk} mK."
        }
        root.addView(status)
        spectrum = SpectrumView(this)
        root.addView(spectrum, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        notes = TextView(this).apply {
            textSize = 20f; setTextColor(Color.WHITE); setPadding(0, pad, 0, pad)
            text = "Dominant: -"
        }
        root.addView(notes)
        val row = LinearLayout(this)
        micBtn = Button(this).apply { text = "Listen (mic)"; setOnClickListener { onMic() } }
        demoBtn = Button(this).apply { text = "Demo chord"; setOnClickListener { toggle(false) } }
        row.addView(micBtn, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        row.addView(demoBtn, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        root.addView(row)
        setContentView(root)
    }

    private fun onMic() {
        if (running) { stop(); return }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        } else toggle(true)
    }

    override fun onRequestPermissionsResult(code: Int, p: Array<out String>, r: IntArray) {
        if (r.isNotEmpty() && r[0] == PackageManager.PERMISSION_GRANTED) toggle(true)
        else status.text = "Microphone permission denied - use Demo chord."
    }

    private fun toggle(useMic: Boolean) {
        if (running) { stop(); return }
        running = true
        micBtn.text = if (useMic) "Stop" else "Listen (mic)"
        demoBtn.text = if (useMic) "Demo chord" else "Stop"
        (if (useMic) demoBtn else micBtn).isEnabled = false
        worker = Thread { loop(useMic) }.also { it.start() }
    }

    private fun stop() {
        running = false
        worker?.join(1000)
        micBtn.text = "Listen (mic)"; demoBtn.text = "Demo chord"
        micBtn.isEnabled = true; demoBtn.isEnabled = true
    }

    override fun onPause() { super.onPause(); if (running) stop() }

    private fun loop(useMic: Boolean) {
        val buf = ShortArray(bufferSize)
        var rec: AudioRecord? = null
        if (useMic) {
            val min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            rec = AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(min, bufferSize * 4))
            rec.startRecording()
        }
        var n = 0L
        val chords = arrayOf(doubleArrayOf(220.0, 261.63, 329.63), doubleArrayOf(146.83, 220.0, 293.66))
        try {
            while (running) {
                val t0 = System.nanoTime()
                if (rec != null) {
                    var got = 0
                    while (got < bufferSize && running) {
                        val r = rec.read(buf, got, bufferSize - got)
                        if (r <= 0) break
                        got += r
                    }
                } else {
                    val chord = chords[((n / 86) % 2).toInt()]  // ~2 s per chord
                    for (i in 0 until bufferSize) {
                        val t = (n * bufferSize + i).toDouble() / rate
                        buf[i] = (chord.sumOf { sin(2 * PI * it * t) } / 3 * 20000).toInt().toShort()
                    }
                    val sleep = bufferSize * 1000L / rate - (System.nanoTime() - t0) / 1_000_000
                    if (sleep > 0) Thread.sleep(sleep)
                }
                n++
                val loud = buf.sumOf { abs(it.toInt()) } / bufferSize / 32768.0
                val temp = thermal.step(minOf(loud * 4, 1.0))
                chip.updateErrorRates(temp, thermal.targetMk)
                val res = chip.quantumFourierTransform(buf, rate)
                runOnUiThread {
                    status.text = "T = %.2f mK   cooling %.2f   He %.3f\ngate err %.2e   readout err %.2e"
                        .format(temp, thermal.cooling, thermal.helium, chip.gateError, chip.readoutError)
                    if (res != null) {
                        spectrum.setData(res.spectrum)
                        notes.text = "Dominant: " + res.peaks.joinToString("  ") {
                            "%s (%.0f Hz)".format(it.note, it.hz)
                        }
                    } else notes.text = "Dominant: silence"
                }
            }
        } finally {
            rec?.stop(); rec?.release()
        }
    }
}
