package com.example.mosanalyzer

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.Crossfade
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import ai.onnxruntime.*
import kotlinx.coroutines.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        setContent { MOSAnalyzerApp() }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MOSAnalyzerApp() {
    val context = LocalContext.current
    var selectedFileName by remember { mutableStateOf("No file selected") }
    var mosScore by remember { mutableStateOf<Float?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    val pickAudioFile = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            isLoading = true
            selectedFileName = getFileNameFromUri(context, it)
            val filePath = getFilePathFromUri(context, it)
            processMOS(context, filePath) { score ->
                mosScore = score
                isLoading = false
            }
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = { TopAppBar(title = { Text("MOS Analyzer") }) }
    ) { padding ->
        Column(
            modifier = Modifier.padding(padding).fillMaxSize().padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Selected File: $selectedFileName", modifier = Modifier.padding(8.dp))
            Button(onClick = { pickAudioFile.launch("audio/*") }) {
                Text("Select Audio File")
            }
            Spacer(modifier = Modifier.height(16.dp))
            Crossfade(targetState = isLoading) { loading ->
                if (loading) {
                    CircularProgressIndicator()
                } else {
                    mosScore?.let { Text("MOS Score: $it", modifier = Modifier.padding(8.dp)) }
                }
            }
        }
    }
}

fun loadAudioFile(filePath: String): FloatArray {
    val file = File(filePath)
    val inputStream = FileInputStream(file)
    val byteArray = inputStream.readBytes()
    inputStream.close()

    val shortArray = ShortArray(byteArray.size / 2)
    ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shortArray)

    return shortArray.map { it / 32768.0f }.toFloatArray() // Normalize audio
}

fun processMOS(context: Context, filePath: String, callback: (Float?) -> Unit) {
    CoroutineScope(Dispatchers.IO).launch {
        // Load and normalize audio.
        val audioData = loadAudioFile(filePath)
        val shortAudioData = audioData.map { (it * 32768).toInt().toShort() }.toShortArray()
        val targetLengthSamples = (9.01 * 16000).toInt()
        val paddedAudio = ensureFixedAudioLength(shortAudioData, targetLengthSamples)
        // Drop the last 160 samples (mimicking the Python code: audio_seg[:-160])
        val segment = paddedAudio.copyOfRange(0, paddedAudio.size - 160)

        // Process the segment with the updated Spectrogram (using frameSize=320 and nMels=120)
        val spectrogram = Spectrogram(fs = 16000, frameSize = 320, nMels = 120)
        spectrogram.processSegment(segment)
        // At this point, spectrogram.melSpectrogram is a List<FloatArray>
        var melSpec = spectrogram.melSpectrogram.toTypedArray()  // shape: [T, 120]

        // Ensure the spectrogram has exactly 900 frames.
        val targetFrames = 900
        if (melSpec.size < targetFrames) {
            // If less, pad at the end by repeating the last frame.
            val padded = Array(targetFrames) { FloatArray(120) }
            for (i in melSpec.indices) {
                padded[i] = melSpec[i]
            }
            for (i in melSpec.size until targetFrames) {
                padded[i] = melSpec.last()
            }
            melSpec = padded
        } else if (melSpec.size > targetFrames) {
            // If more, truncate.
            melSpec = melSpec.take(targetFrames).toTypedArray()
        }

        val mosCalculator = ComputeMOS(context)
        // The ONNX model expects an input of shape [batch, 900, 120]. Wrap in a List for batch size 1.
        val score = mosCalculator.predictMOS(listOf(melSpec))
        withContext(Dispatchers.Main) { callback(score) }
    }
}


fun ensureFixedAudioLength(audio: ShortArray, targetLength: Int): ShortArray {
    return if (audio.size < targetLength) {
        // Pad with repeated data
        val repeatedAudio = ShortArray(targetLength)
        for (i in repeatedAudio.indices) {
            repeatedAudio[i] = audio[i % audio.size]
        }
        repeatedAudio
    } else {
        // Trim to required length
        audio.copyOfRange(0, targetLength)
    }
}




fun getFilePathFromUri(context: Context, uri: Uri): String {
    val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
    val file = File(context.cacheDir, "temp_audio.wav")
    val outputStream = FileOutputStream(file)
    inputStream?.copyTo(outputStream)
    inputStream?.close()
    outputStream.close()
    return file.absolutePath
}

fun getFileNameFromUri(context: Context, uri: Uri): String {
    val cursor = context.contentResolver.query(uri, null, null, null, null)
    val nameIndex = cursor?.getColumnIndex(OpenableColumns.DISPLAY_NAME)
    cursor?.moveToFirst()
    val fileName = nameIndex?.let { cursor.getString(it) }
    cursor?.close()
    return fileName ?: "Unknown"
}

//fun computeMelSpectrogram(audio: ShortArray, nMels: Int): Array<FloatArray> {
//    val spectrogram = Spectrogram(fs = 16000, fftSize = 640, melFilterbankSize = nMels)
//    spectrogram.processAudio(audio)
//
//    // Convert spectrogram data to Mel spectrogram format
//    val melSpectrograms = mutableListOf<FloatArray>()
//
//    for (frame in spectrogram.mSpectrogram) {
//        melSpectrograms.add(frame.copyOf(nMels)) // Ensure we only take `nMels` features
//    }
//
//    // Ensure exactly 900 frames (truncate or pad)
//    return if (melSpectrograms.size >= 900) {
//        melSpectrograms.take(900).toTypedArray()
//    } else {
//        val paddedMelSpectrograms = Array(900) { FloatArray(nMels) }
//        for (i in melSpectrograms.indices) {
//            paddedMelSpectrograms[i] = melSpectrograms[i]
//        }
//        paddedMelSpectrograms
//    }
//}
//


class ComputeMOS(context: Context) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelFile = copyModelToInternalStorage(context, "model_v8.onnx")
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = ortEnv.createSession(modelFile.absolutePath, options)
    }

    private fun copyModelToInternalStorage(context: Context, fileName: String): File {
        val modelFile = File(context.filesDir, fileName)
        if (!modelFile.exists()) {
            context.assets.open(fileName).use { input -> modelFile.outputStream().use { output -> input.copyTo(output) } }
        }
        return modelFile
    }

    fun predictMOS(melSpectrograms: List<Array<FloatArray>>?): Float? {
        return try {
            melSpectrograms?.mapNotNull { melSpec ->
                val inputTensor = OnnxTensor.createTensor(ortEnv, arrayOf(melSpec))
                val output = session.run(Collections.singletonMap("input_1", inputTensor))
                (output[0].value as Array<FloatArray>)[0][0]
            }?.average()?.toFloat()
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}
