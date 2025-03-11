package com.example.mosanalyzer

import android.graphics.Bitmap
import android.graphics.Color
import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.*

class Spectrogram(
    private val fs: Int = 16000,           // Sampling rate
    private val frameSize: Int = 320,        // frame_size from Python
    private val nMels: Int = 120             // n_mels from Python
) {
    // In Python, n_fft is frameSize + 1
    private val nFft = frameSize + 1          // 321
    private val hopLength = 160               // as in Python
    private val numFilters = nMels

    // Hann window of length nFft (we use nFft here for FFT computation)
    private val hannWindow = FloatArray(nFft) { i ->
        0.5f - 0.5f * cos(2.0 * Math.PI * i / nFft).toFloat()
    }

    // FFT instance
    private val fft = FloatFFT_1D(nFft.toLong())

    // Instantiate our Mel filterbank using our updated class (see below)
    private val melFilterbank = MelFilterbank(fs.toFloat(), frameSize, nMels)

    // The computed mel spectrogram frames (each is a FloatArray of length nMels)
    val melSpectrogram = mutableListOf<FloatArray>()

    /**
     * Process a segment of audio. The input segment should already be of fixed length.
     * In the Python code, each segment is audio_seg[:-160]; that is, the last 160 samples are dropped.
     */
    fun processSegment(segment: ShortArray) {
        // Convert to FloatArray
        val audioFloat = FloatArray(segment.size) { i -> segment[i].toFloat() }
        // Determine the number of frames using: 1 + floor((len(segment) - nFft) / hopLength)
        val numFrames = 1 + ((audioFloat.size - nFft) / hopLength)
        for (i in 0 until numFrames) {
            val start = i * hopLength
            // Extract frame of length nFft
            val frame = FloatArray(nFft) { j -> audioFloat[start + j] }
            // Apply Hann window
            for (j in frame.indices) {
                frame[j] *= hannWindow[j]
            }
            // Compute FFT in-place
            fft.realForward(frame)
            // Compute power spectrum for the first nFft/2 bins
            val nBins = nFft / 2  // integer division (321/2 = 160)
            val powerSpectrum = FloatArray(nBins)
            for (k in 0 until nBins) {
                // In real FFT output, real and imaginary parts are interleaved:
                val real = frame[2 * k]
                val imag = frame[2 * k + 1]
                powerSpectrum[k] = real * real + imag * imag
            }
            // Apply the Mel filterbank: for each filter, sum the weighted power
            val melEnergies = FloatArray(numFilters)
            for (m in 0 until numFilters) {
                var sum = 0f
                for (k in 0 until nBins) {
                    sum += melFilterbank.filterbank[m][k] * powerSpectrum[k]
                }
                melEnergies[m] = sum
            }
            // Save the raw mel energies for this frame
            melSpectrogram.add(melEnergies)
        }
        // After processing all frames, apply global dB conversion and normalization
        finalizeNormalization()
    }

    /**
     * Normalize the mel spectrogram as in Python:
     *   - Compute the global maximum over all frames.
     *   - For each value, compute: 10 * log10(value / globalMax + eps)
     *   - Then normalize: (dB + 40) / 40
     */
    private fun finalizeNormalization() {
        if (melSpectrogram.isEmpty()) return
        var globalMax = 0f
        for (frame in melSpectrogram) {
            for (value in frame) {
                if (value > globalMax) globalMax = value
            }
        }
        if (globalMax <= 0f) globalMax = 1e-8f

        for (i in melSpectrogram.indices) {
            for (j in 0 until numFilters) {
                val ratio = (melSpectrogram[i][j] / globalMax).coerceAtLeast(1e-8f)
                val dB = 10 * log10(ratio)
                melSpectrogram[i][j] = (dB + 40) / 40
            }
        }
    }

    /**
     * (Optional) Create a bitmap visualization of the spectrogram.
     */
    fun getSpectrogramBitmap(): Bitmap {
        val width = numFilters
        val height = melSpectrogram.size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        var index = 0
        for (i in 0 until height) {
            for (j in 0 until width) {
                val intensity = (melSpectrogram[i][j] * 255).toInt().coerceIn(0, 255)
                pixels[index++] = Color.rgb(intensity, intensity, intensity)
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
