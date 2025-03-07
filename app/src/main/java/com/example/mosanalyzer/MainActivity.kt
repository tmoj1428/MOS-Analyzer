package com.example.mosanalyzer

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import ai.onnxruntime.*
import android.content.Context
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.animation.Crossfade
import androidx.compose.ui.Alignment
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.*
import java.util.Collections

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

fun processMOS(context: Context, filePath: String, callback: (Float?) -> Unit) {
    CoroutineScope(Dispatchers.IO).launch {
        val melSpectrograms = computeMelSpectrogram(filePath)
        val mosCalculator = ComputeMOS(context)
        val score = mosCalculator.predictMOS(melSpectrograms)
        withContext(Dispatchers.Main) { callback(score) }
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

fun computeMelSpectrogram(audioPath: String): List<Array<FloatArray>>? {
    return try {
        val py = Python.getInstance().getModule("mel_spectrogram")
        val melSpecsList = py.callAttr("get_melspec", audioPath)
        melSpecsList.asList().map { it.asList().map { row -> row.asList().map { it.toFloat() }.toFloatArray() }.toTypedArray() }
    } catch (e: Exception) {
        Log.e("PythonError", "Error computing Mel Spectrogram: ${e.message}")
        null
    }
}

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