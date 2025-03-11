package com.example.mosanalyzer

import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow

class MelFilterbank(
    private val fs: Float,
    private val frameSize: Int = 320,   // frameSize from Spectrogram (used to compute n_fft below)
    private val numFilters: Int = 120   // n_mels from Python
) {
    // In librosa, n_fft = frameSize + 1
    private val nFftPlus = frameSize + 1    // e.g., 321
    private val nyquistFs = fs / 2

    // We will build filters for bins [0, nFftPlus/2)
    // nFftPlus/2 is 321/2 = 160 (integer division)
    val filterbank = Array(numFilters) { FloatArray(nFftPlus / 2) }

    init {
        // Compute mel points (numFilters+2 points) between 0 and freqToMel(nyquist)
        val melMin = 0f
        val melMax = freqToMel(nyquistFs)
        val melPoints = FloatArray(numFilters + 2)
        val hzPoints = FloatArray(numFilters + 2)
        val deltaMel = (melMax - melMin) / (numFilters + 1)
        for (m in 0 until numFilters + 2) {
            melPoints[m] = melMin + deltaMel * m
            hzPoints[m] = melToFreq(melPoints[m])
        }
        // Compute FFT bin indices using librosa’s formula: floor((n_fft+1) * hz / sr)
        val binPoints = IntArray(numFilters + 2) { i ->
            floor((nFftPlus * hzPoints[i] / fs).toDouble()).toInt()
        }
        // Build triangular filters
        for (m in 1 until numFilters + 1) {
            val binMin = binPoints[m - 1]
            val binCenter = binPoints[m]
            val binMax = binPoints[m + 1]
            // Rising slope
            for (k in binMin until binCenter) {
                filterbank[m - 1][k] = (k - binMin).toFloat() / (binCenter - binMin)
            }
            // Falling slope
            for (k in binCenter until binMax) {
                filterbank[m - 1][k] = (binMax - k).toFloat() / (binMax - binCenter)
            }
        }
    }

    private fun freqToMel(hz: Float): Float = 2595f * log10(1 + hz / 700)
    private fun melToFreq(mel: Float): Float = 700f * (10.0.pow((mel / 2595).toDouble()) - 1).toFloat()
}
