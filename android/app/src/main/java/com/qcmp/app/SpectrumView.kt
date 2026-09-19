package com.qcmp.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Bar chart of the measured QFT probability per frequency bin. */
class SpectrumView(context: Context) : View(context) {
    @Volatile private var data = DoubleArray(0)
    private val bar = Paint().apply { color = Color.rgb(0x7C, 0x4D, 0xFF) }
    private val axis = Paint().apply { color = Color.GRAY; textSize = 28f }

    fun setData(spectrum: DoubleArray) { data = spectrum; postInvalidate() }

    override fun onDraw(c: Canvas) {
        c.drawColor(Color.rgb(0x12, 0x12, 0x1A))
        val d = data
        if (d.isEmpty()) return
        val max = (d.maxOrNull() ?: 1.0).coerceAtLeast(1e-9)
        val w = width.toFloat() / d.size
        for (i in d.indices) {
            val h = (d[i] / max * (height - 40)).toFloat()
            c.drawRect(i * w, height - 30 - h, (i + 1) * w - 1f, height - 30f, bar)
        }
        c.drawText("0 Hz", 4f, height - 4f, axis)
        c.drawText("frequency bin ->", width / 2f - 80f, height - 4f, axis)
    }
}
