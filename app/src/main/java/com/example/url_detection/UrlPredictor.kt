package com.example.url_detection

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

data class TfidfParams(
    val vocabulary: Map<String, Int>?,
    val idf_weights: List<Double>?
)

class UrlPredictor private constructor( // Private constructor
    private val context: Context,
    private val interpreter: Interpreter?,
    private val tfidfParams: TfidfParams?
) {

    /**
     * Returns the probability (0..1) that [url] is malicious, or null if the model
     * is not loaded or the prediction fails. The model outputs 1 for Safe, so this
     * is 1 minus the raw score. The Interpreter is not thread-safe, hence @Synchronized.
     */
    @Synchronized
    fun maliciousProbability(url: String): Float? {
        val (label, confidence) = predict(url)
        return when {
            label.contains("Safe") -> 1f - confidence
            label.contains("Phishing") -> confidence
            else -> null
        }
    }

    fun predict(url: String): Pair<String, Float> {
        if (interpreter == null || tfidfParams?.vocabulary == null || tfidfParams.idf_weights == null) {
            return Pair("Model Not Loaded", 0.0f)
        }
        if (url.isEmpty()){
            return Pair("Invalid URL", 0.0f)
        }

        // Step 1: TF-IDF Vectorization
        val tfidfVector = calculateTfidf(url, tfidfParams)

        // Step 2: Run Prediction
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        val expectedSize = inputTensor.shape().last()

        if (tfidfVector.size != expectedSize) {
            Log.e("UrlPredictor", "Feature size mismatch. Model expects $expectedSize, but received ${tfidfVector.size}")
            return Pair("Prediction Error", 0.0f)
        }

        val inputBuffer = ByteBuffer.allocateDirect(4 * expectedSize).order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(tfidfVector)

        val outputArray = Array(1) { FloatArray(outputTensor.shape().last()) }
        interpreter.run(inputBuffer, outputArray)
        val predictionScore = outputArray[0][0]

        // Step 3: Format and Return Result
        // Python code says: 1 for Safe, 0 for Malicious. So > 0.5 is Safe.
        val confidence = if (predictionScore > 0.5f) predictionScore else 1.0f - predictionScore
        val label = if (predictionScore > 0.5f) "🟢 Safe" else "🔴 Phishing"

        return Pair(label, confidence)
    }

    private fun calculateTfidf(url: String, tfidfParams: TfidfParams): FloatArray {
        val vocabulary = tfidfParams.vocabulary!! // We know this is not null from the check in predict()
        val idf = tfidfParams.idf_weights!!
        val tfidfVector = FloatArray(vocabulary.size) { 0.0f }

        // Tokenize the URL similar to sklearn's default
        val tokens = url.lowercase().split(Regex("""\W+""")).filter { it.length > 1 }
        val tokenCounts = tokens.groupingBy { it }.eachCount()

        if (tokens.isNotEmpty()) {
            for ((token, count) in tokenCounts) {
                val index = vocabulary[token]
                if (index != null && index < idf.size) {
                    // Python code uses raw count for TF
                    val tf = count.toFloat()
                    val idfValue = idf[index].toFloat()
                    tfidfVector[index] = tf * idfValue
                }
            }
        }

        // L2 Normalization
        val norm = sqrt(tfidfVector.sumOf { (it * it).toDouble() }).toFloat()
        if (norm > 0) {
            for (i in tfidfVector.indices) {
                tfidfVector[i] /= norm
            }
        }
        return tfidfVector
    }

    fun close() {
        interpreter?.close()
    }

    companion object {
        fun create(context: Context): UrlPredictor {
            var interpreter: Interpreter? = null
            var tfidfParams: TfidfParams? = null

            try {
                interpreter = Interpreter(loadModelFile(context, "Keras_NN_Traditional_Dataset_final_mem_safe.tflite"))

                val gson = Gson()
                context.assets.open("tfidf_data.json").bufferedReader().use {
                    val type = object : TypeToken<TfidfParams>() {}.type
                    tfidfParams = gson.fromJson(it, type)
                }
            } catch (t: Throwable) { // CATCH ALL THROWABLES
                Log.e("UrlPredictor", "CRITICAL: Failed to initialize UrlPredictor. Check asset files.", t)
            }

            return UrlPredictor(context, interpreter, tfidfParams)
        }

        private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
            val fileDescriptor = context.assets.openFd(modelName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }
}
