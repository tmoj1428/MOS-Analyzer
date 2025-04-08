package com.example.mosanalyzer

import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow

/** Mel Filterbank with normalization (mimicking librosa.filters.mel with norm="slaney") */
class MelFilterbank(
    private val fs: Float,
    private val frameSize: Int = 320,
    private val numFilters: Int = 120
) {
    private val nFft = frameSize + 1
    private val numBins = (nFft + 1) / 2
    val filterbank = Array(numFilters) { FloatArray(numBins) }

    init {
        val melMax = freqToMel(fs / 2)
        val hzPoints = (0..numFilters + 1).map {
            melToFreq(it * melMax / (numFilters + 1))
        }.toFloatArray()

        val binPoints = hzPoints.map { hz ->
            floor((nFft * hz / fs).toDouble()).toInt()
        }.toIntArray()

        for (m in 1..numFilters) {
            val left = binPoints[m-1]
            val center = binPoints[m]
            val right = binPoints[m+1]
            val scale = 2.0f / (right - left)

            for (k in left until center) {
                filterbank[m-1][k] = (k - left).toFloat() / (center - left) * scale
            }
            for (k in center until right) {
                filterbank[m-1][k] = (right - k).toFloat() / (right - center) * scale
            }
        }
    }

    private fun freqToMel(hz: Float) = 2595f * log10(1 + hz/700f)
    private fun melToFreq(mel: Float) = 700f * (10.0.pow(mel/2595.0) - 1).toFloat()
}