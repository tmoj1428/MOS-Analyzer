package com.example.mosanalyzer

import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.*

/** Spectrogram calculation with center-padding and proper mel normalization */
class Spectrogram(
    private val fs: Int = 16000,
    private val frameSize: Int = 320,
    private val nMels: Int = 120
) {
    private val nFft = frameSize + 1
    private val hopLength = 160
    private val numBins = (nFft + 1) / 2
    private val hannWindow = FloatArray(nFft) { i ->
        0.5f - 0.5f * cos(2.0 * PI * i / (nFft - 1)).toFloat()
    }
    private val fft = FloatFFT_1D(nFft.toLong())
    private val melFilterbank = MelFilterbank(fs.toFloat(), frameSize, nMels)
    val melSpectrogram = mutableListOf<FloatArray>()

    fun processSegment(segment: FloatArray) {
        val padSize = nFft / 2
        val paddedSegment = padArray(segment, padSize)
        val numFrames = 1 + (paddedSegment.size - nFft) / hopLength
        melSpectrogram.clear()

        for (i in 0 until numFrames) {
            val start = i * hopLength
            val frame = paddedSegment.copyOfRange(start, start + nFft)

            // Apply Hann window
            for (j in frame.indices) frame[j] *= hannWindow[j]

            // FFT calculation
            val fftData = frame.copyOf()
            fft.realForward(fftData)

            // Power spectrum with librosa's epsilon
            val powerSpectrum = FloatArray(numBins).apply {
                this[0] = fftData[0].pow(2) + 1e-10f
                for (k in 1 until numBins) {
                    val real = fftData[2*k - 1]
                    val imag = fftData[2*k]
                    this[k] = real.pow(2) + imag.pow(2) + 1e-10f
                }
            }

            // Filter application
            val melEnergies = FloatArray(nMels) { m ->
                var sum = 0f
                for (k in powerSpectrum.indices) {
                    sum += powerSpectrum[k] * melFilterbank.filterbank[m][k]
                }
                sum
            }
            melSpectrogram.add(melEnergies)
        }
        normalizeSpectrogram()
    }

    private fun padArray(arr: FloatArray, pad: Int): FloatArray {
        val padded = FloatArray(arr.size + 2 * pad).apply {
            System.arraycopy(arr, 0, this, pad, arr.size)
            for (i in 0 until pad) {
                this[pad - 1 - i] = arr[i.coerceAtMost(arr.size - 1)]
                this[pad + arr.size + i] = arr[(arr.size - 1 - i).coerceAtLeast(0)]
            }
        }
        return padded
    }

    private fun normalizeSpectrogram() {
        val globalMax = melSpectrogram.maxOf { it.maxOrNull() ?: 0f }
        val eps = 1e-10f

        melSpectrogram.forEachIndexed { i, frame ->
            for (j in frame.indices) {
                val ratio = (frame[j] / globalMax).coerceAtLeast(eps)
                val dB = 10 * log10(ratio)
                melSpectrogram[i][j] = (dB + 40) / 40
            }
        }
    }
}
