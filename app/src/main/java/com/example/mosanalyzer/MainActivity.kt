package com.example.mosanalyzer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
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
import androidx.documentfile.provider.DocumentFile
import android.util.Log
import kotlinx.coroutines.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import kotlin.math.floor
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MOSAnalyzerApp() }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MOSAnalyzerApp() {
    val context = LocalContext.current
    var selectedFileName by remember { mutableStateOf("No file selected") }
    var mosScore by remember { mutableStateOf<Float?>(null) }
    var folderResult by remember { mutableStateOf("No folder processed yet") }
    var isLoading by remember { mutableStateOf(false) }

    // Launcher for a single audio file.
    val pickAudioFile = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            isLoading = true
            selectedFileName = getFileNameFromUri(context, it)
            val filePath = getFilePathFromUri(context, it)
            CoroutineScope(Dispatchers.IO).launch {
                val score = processMOSFile(context, filePath)
                withContext(Dispatchers.Main) {
                    mosScore = score
                    isLoading = false
                }
                // Delete the temporary file after processing.
                File(filePath).delete()
            }
        }
    }

    // Launcher for selecting a folder.
    val pickFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { folderUri: Uri? ->
        folderUri?.let {
            isLoading = true
            CoroutineScope(Dispatchers.IO).launch {
                val result = processFolderIncrementally(context, it)
                withContext(Dispatchers.Main) {
                    folderResult = result
                    isLoading = false
                }
            }
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = { TopAppBar(title = { Text("MOS Analyzer") }) }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Selected File: $selectedFileName", modifier = Modifier.padding(8.dp))
            Button(onClick = { pickAudioFile.launch("audio/*") }) {
                Text("Select Audio File")
            }
            Spacer(modifier = Modifier.height(16.dp))
            Text("Folder Processing Result:", modifier = Modifier.padding(8.dp))
            Text(folderResult, modifier = Modifier.padding(8.dp))
            Button(onClick = { pickFolder.launch(null) }) {
                Text("Select Folder")
            }
            Spacer(modifier = Modifier.height(16.dp))
            Crossfade(targetState = isLoading) { loading ->
                if (loading) {
                    CircularProgressIndicator()
                } else {
                    mosScore?.let { Text("MOS Score for file: $it", modifier = Modifier.padding(8.dp)) }
                }
            }
        }
    }
}

/** Suspend function to process one audio file and return its MOS score */
@SuppressLint("DefaultLocale")
suspend fun processMOSFile(context: Context, filePath: String): Float? = withContext(Dispatchers.IO) {
    try {
        val samplingRate = 16000
        val inputLength = 9.01
        val audioData = loadAudioFile(filePath)
        val lenSamples = (inputLength * samplingRate).roundToInt()
        var audio = audioData.copyOf()
        while (audio.size < lenSamples) {
            audio += audio
        }
        val numHops = (floor(audio.size.toDouble() / samplingRate) - inputLength).toInt() + 1
        val allMelSpecs = mutableListOf<Array<FloatArray>>()
        for (idx in 0 until numHops) {
            val start = (idx * samplingRate).toInt()
            val end = ((idx + inputLength) * samplingRate).toInt()
            if (end > audio.size) continue
            val audioSeg = audio.sliceArray(start until end)
            if (audioSeg.size < lenSamples) continue
            // Remove last 160 samples (audio_seg[:-160])
            val segment = audioSeg.copyOfRange(0, audioSeg.size - 160)
            val spectrogram = Spectrogram(fs = 16000, frameSize = 320, nMels = 120)
            spectrogram.processSegment(segment)
            val melSpec = spectrogram.melSpectrogram.toTypedArray()
            allMelSpecs.add(melSpec)
        }
        val mosCalculator = ComputeMOS(context)
        mosCalculator.predictMOS(allMelSpecs)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

/** Suspend function to process every audio file in the given folder incrementally.
 * It processes each file, writes its MOS result immediately to a result file, and then releases resources.
 */
suspend fun processFolderIncrementally(context: Context, folderUri: Uri): String = withContext(Dispatchers.IO) {
    val outputFile = File(context.filesDir, "mos_results.txt")
    val writer = BufferedWriter(FileWriter(outputFile, false))
    val pickedDir = DocumentFile.fromTreeUri(context, folderUri)
    var processedCount = 0

    if (pickedDir != null && pickedDir.isDirectory) {
        val allFiles = pickedDir.listFiles().filter {
            it.isFile && (it.type?.startsWith("audio/") == true)
        }

        val batches = allFiles.chunked(200)
        for ((batchIndex, batch) in batches.withIndex()) {
            Log.d("FolderMOS", "Processing batch ${batchIndex + 1}/${batches.size}")

            // New instance per batch
            val mosCalculator = ComputeMOS(context)

            for (file in batch) {
                val fileName = file.name ?: "Unknown"
                try {
                    val filePath = getFilePathFromUri(context, file.uri)
                    val score = processMOSFileWithInstance(context, filePath, mosCalculator)
                    writer.write("$fileName : ${score ?: "Error"}\n")
                    writer.flush()
                    processedCount++
                    File(filePath).delete()
                } catch (ex: Exception) {
                    ex.printStackTrace()
                    writer.write("$fileName : Exception\n")
                    writer.flush()
                }
                yield() // avoid blocking
            }

            // Clean up
            mosCalculator.close() // Close ONNX resources
            System.gc() // Suggest GC to clean up aggressively
            delay(100)  // slight delay to let the system breathe
        }
    }

    writer.close()
    "Processed $processedCount files. Results saved to: ${outputFile.absolutePath}"
}

/** Write a given text to a file in internal storage */
fun writeResultsToFile(context: Context, content: String, filename: String) {
    try {
        val file = File(context.filesDir, filename)
        file.writeText(content)
        Log.d("FolderMOS", "Saved results to: ${file.absolutePath}")
    } catch (e: Exception) {
        Log.e("FolderMOS", "Error writing file", e)
    }
}

/** Returns the file path for the Uri by copying its content into a temporary file */
fun getFilePathFromUri(context: Context, uri: Uri): String {
    val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
    // Create a temporary file with a unique name.
    val tempFile = File(context.cacheDir, "temp_audio_${System.currentTimeMillis()}.wav")
    FileOutputStream(tempFile).use { outputStream ->
        inputStream?.copyTo(outputStream)
    }
    inputStream?.close()
    return tempFile.absolutePath
}

/** Returns the display name of the file from its Uri */
fun getFileNameFromUri(context: Context, uri: Uri): String {
    val cursor = context.contentResolver.query(uri, null, null, null, null)
    val nameIndex = cursor?.getColumnIndex(OpenableColumns.DISPLAY_NAME)
    cursor?.moveToFirst()
    val fileName = nameIndex?.let { cursor.getString(it) }
    cursor?.close()
    return fileName ?: "Unknown"
}

/** Loads a WAV file from the given file path and converts it to normalized FloatArray */
fun loadAudioFile(filePath: String): FloatArray {
    val file = File(filePath)
    val inputStream = FileInputStream(file)
    val byteArray = inputStream.readBytes()
    inputStream.close()
    val headerSize = 44  // Standard WAV header size
    val audioBytes = byteArray.copyOfRange(headerSize, byteArray.size)
    val shortArray = ShortArray(audioBytes.size / 2)
    ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shortArray)
    return shortArray.map { it / Short.MAX_VALUE.toFloat() }.toFloatArray()
}

/** (Assuming your Spectrogram, ComputeMOS, etc. remain unchanged) */
@SuppressLint("DefaultLocale")
fun processMOS(context: Context, filePath: String, callback: (Float?) -> Unit) {
    CoroutineScope(Dispatchers.IO).launch {
        val score = processMOSFile(context, filePath)
        callback(score)
    }
}

suspend fun processMOSFileWithInstance(context: Context, filePath: String, mosCalculator: ComputeMOS): Float? = withContext(Dispatchers.IO) {
    try {
        val samplingRate = 16000
        val inputLength = 9.01
        val audioData = loadAudioFile(filePath)
        val lenSamples = (inputLength * samplingRate).roundToInt()
        var audio = audioData.copyOf()
        while (audio.size < lenSamples) {
            audio += audio
        }
        val numHops = (floor(audio.size.toDouble() / samplingRate) - inputLength).toInt() + 1
        val allMelSpecs = mutableListOf<Array<FloatArray>>()
        for (idx in 0 until numHops) {
            val start = (idx * samplingRate).toInt()
            val end = ((idx + inputLength) * samplingRate).toInt()
            if (end > audio.size) continue
            val audioSeg = audio.sliceArray(start until end)
            if (audioSeg.size < lenSamples) continue
            val segment = audioSeg.copyOfRange(0, audioSeg.size - 160)
            val spectrogram = Spectrogram(fs = 16000, frameSize = 320, nMels = 120)
            spectrogram.processSegment(segment)
            val melSpec = spectrogram.melSpectrogram.toTypedArray()
            allMelSpecs.add(melSpec)
        }
        mosCalculator.predictMOS(allMelSpecs)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

class ComputeMOS(context: Context) : Closeable {
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
            context.assets.open(fileName).use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return modelFile
    }

    fun predictMOS(melSpectrograms: List<Array<FloatArray>>?): Float? {
        return try {
            melSpectrograms?.mapNotNull { melSpec ->
                OnnxTensor.createTensor(ortEnv, arrayOf(melSpec)).use { inputTensor ->
                    val output = session.run(Collections.singletonMap("input_1", inputTensor))
                    (output[0].value as Array<FloatArray>)[0][0]
                }
            }?.average()?.toFloat()
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    override fun close() {
        session.close()
        // ortEnv.close()  <- don't close this unless you're done with everything
    }
}
