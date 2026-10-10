package com.example.url_detection.data

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import android.util.Log
import com.example.url_detection.BuildConfig
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/** Google's threat lists, numbered as in the v5 API. Labels are short so they fit a signal row. */
enum class ThreatType(val code: Int, val label: String) {
    Malware(1, "Malware"),
    SocialEngineering(2, "Phishing"),
    UnwantedSoftware(3, "Unwanted app"),
    HarmfulApp(4, "Harmful app");

    companion object {
        fun fromCode(code: Int): ThreatType? = entries.firstOrNull { it.code == code }
    }
}

/**
 * Stored with each check. NotChecked is for checks made before the online check
 * existed; Off means the user turned the online check off.
 */
enum class SafeBrowsingStatus { NotChecked, Off, Unavailable, Clean, Threat }

data class SafeBrowsingResult(val status: SafeBrowsingStatus, val threats: Set<ThreatType> = emptySet()) {
    companion object {
        val Off = SafeBrowsingResult(SafeBrowsingStatus.Off)
        val Unavailable = SafeBrowsingResult(SafeBrowsingStatus.Unavailable)
    }
}

/** Threat types are stored in Room as comma-separated names. */
fun encodeThreats(threats: Set<ThreatType>): String = threats.joinToString(",") { it.name }

fun decodeThreats(text: String): Set<ThreatType> =
    text.split(',').mapNotNull { name -> ThreatType.entries.firstOrNull { it.name == name } }.toSet()

/**
 * Asks Google Safe Browsing (v5 urls:search) whether a link is on its threat lists.
 * Never throws: no key, no network, a timeout or a bad answer all come back as Unavailable,
 * so the on-device verdict still stands.
 */
class SafeBrowsingClient(
    context: Context,
    private val apiKey: String = BuildConfig.SAFE_BROWSING_API_KEY,
) {
    private val packageName = context.packageName

    // Sent so the key can be restricted to this app in Google Cloud Console.
    private val certSha1 = signingCertSha1(context)

    private val http = OkHttpClient.Builder()
        .callTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .build()

    private val cache = object : LinkedHashMap<String, Pair<SafeBrowsingResult, Long>>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Pair<SafeBrowsingResult, Long>>) =
            size > CACHE_SIZE
    }

    suspend fun check(raw: String): SafeBrowsingResult {
        if (apiKey.isBlank()) {
            Log.w(TAG, "No SAFE_BROWSING_API_KEY in local.properties; skipping the online check.")
            return SafeBrowsingResult.Unavailable
        }
        val url = openableUrl(raw)
        cached(url)?.let { return it }

        val request = Request.Builder()
            .url(
                ENDPOINT.toHttpUrl().newBuilder()
                    .addQueryParameter("urls", url)
                    .addQueryParameter("key", apiKey)
                    .build()
            )
            .header("Accept", "application/x-protobuf")
            .header("X-Android-Package", packageName)
            .apply { certSha1?.let { header("X-Android-Cert", it) } }
            .build()

        return try {
            val (code, body) = http.newCall(request).await()
            if (code != 200) {
                // The error body is JSON from Google, e.g. an invalid or restricted key. It never contains the key.
                Log.w(TAG, "Safe Browsing answered HTTP $code: ${body.decodeToString().take(300)}")
                return SafeBrowsingResult.Unavailable
            }
            val parsed = SearchUrlsResponse.parse(body)
            val result = if (parsed.matched) {
                SafeBrowsingResult(SafeBrowsingStatus.Threat, parsed.threats)
            } else {
                SafeBrowsingResult(SafeBrowsingStatus.Clean)
            }
            store(url, result, parsed.cacheSeconds)
            result
        } catch (e: IOException) {
            Log.w(TAG, "Safe Browsing unreachable: ${e.javaClass.simpleName}")
            SafeBrowsingResult.Unavailable
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "Safe Browsing sent an answer we couldn't read", e)
            SafeBrowsingResult.Unavailable
        }
    }

    @Synchronized
    private fun cached(url: String): SafeBrowsingResult? {
        val (result, expiresAt) = cache[url] ?: return null
        if (SystemClock.elapsedRealtime() < expiresAt) return result
        cache.remove(url)
        return null
    }

    /** Keeps the answer for as long as Google says it stays valid. */
    @Synchronized
    private fun store(url: String, result: SafeBrowsingResult, seconds: Long) {
        if (seconds > 0) cache[url] = result to SystemClock.elapsedRealtime() + seconds * 1000
    }

    private companion object {
        const val TAG = "SafeBrowsing"
        const val ENDPOINT = "https://safebrowsing.googleapis.com/v5/urls:search"
        const val TIMEOUT_SECONDS = 4L
        const val CACHE_SIZE = 100
    }
}

/** Runs the call off the main thread and cancels it if the caller is cancelled. */
private suspend fun Call.await(): Pair<Int, ByteArray> = suspendCancellableCoroutine { cont ->
    cont.invokeOnCancellation { cancel() }
    enqueue(object : Callback {
        override fun onFailure(call: Call, e: IOException) {
            if (cont.isActive) cont.resumeWith(Result.failure(e))
        }

        override fun onResponse(call: Call, response: Response) {
            val result = runCatching { response.use { it.code to it.body!!.bytes() } }
            if (cont.isActive) cont.resumeWith(result.recoverCatching { throw IOException(it) })
        }
    })
}

/** SHA-1 of the app's signing certificate as uppercase hex, the form X-Android-Cert expects. */
private fun signingCertSha1(context: Context): String? = runCatching {
    val pm = context.packageManager
    val cert = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        pm.getPackageInfo(context.packageName, PackageManager.GET_SIGNING_CERTIFICATES)
            .signingInfo?.apkContentsSigners?.firstOrNull()
    } else {
        @Suppress("DEPRECATION")
        pm.getPackageInfo(context.packageName, PackageManager.GET_SIGNATURES).signatures?.firstOrNull()
    } ?: return null
    MessageDigest.getInstance("SHA-1").digest(cert.toByteArray()).joinToString("") { "%02X".format(it) }
}.getOrNull()

/**
 * The v5 answer is protobuf only (it refuses JSON):
 *
 *   SearchUrlsResponse { repeated ThreatUrl threats = 1; Duration cacheDuration = 2; }
 *   ThreatUrl          { string url = 1; repeated ThreatType threatTypes = 2; }
 *   Duration           { int64 seconds = 1; int32 nanos = 2; }
 *
 * Small enough to read by hand instead of adding a protobuf library.
 * Throws IllegalArgumentException on malformed input.
 */
internal object SearchUrlsResponse {
    data class Parsed(val matched: Boolean, val threats: Set<ThreatType>, val cacheSeconds: Long)

    fun parse(bytes: ByteArray): Parsed {
        var matched = false
        val threats = mutableSetOf<ThreatType>()
        var cacheSeconds = 0L
        val r = ProtoReader(bytes, 0, bytes.size)
        while (r.hasMore()) {
            val (field, wire) = r.tag()
            when {
                field == 1 && wire == LEN -> {
                    matched = true
                    readThreatUrl(r.message(), threats)
                }
                field == 2 && wire == LEN -> cacheSeconds = readDurationSeconds(r.message())
                else -> r.skip(wire)
            }
        }
        return Parsed(matched, threats, cacheSeconds)
    }

    private fun readThreatUrl(r: ProtoReader, into: MutableSet<ThreatType>) {
        while (r.hasMore()) {
            val (field, wire) = r.tag()
            when {
                // Repeated enums usually arrive packed, but a single unpacked value is valid too.
                field == 2 && wire == LEN -> r.message().let { p -> while (p.hasMore()) addThreat(p.varint(), into) }
                field == 2 && wire == VARINT -> addThreat(r.varint(), into)
                else -> r.skip(wire)
            }
        }
    }

    private fun addThreat(code: Long, into: MutableSet<ThreatType>) {
        ThreatType.fromCode(code.toInt())?.let { into += it }
    }

    private fun readDurationSeconds(r: ProtoReader): Long {
        var seconds = 0L
        while (r.hasMore()) {
            val (field, wire) = r.tag()
            if (field == 1 && wire == VARINT) seconds = r.varint() else r.skip(wire)
        }
        return seconds
    }

    private const val VARINT = 0
    private const val LEN = 2
}

private class ProtoReader(private val buf: ByteArray, private var pos: Int, private val end: Int) {
    fun hasMore() = pos < end

    fun varint(): Long {
        var result = 0L
        var shift = 0
        while (true) {
            require(pos < end) { "Truncated varint" }
            val b = buf[pos++].toInt()
            result = result or ((b and 0x7F).toLong() shl shift)
            if (b and 0x80 == 0) return result
            shift += 7
            require(shift < 64) { "Varint too long" }
        }
    }

    /** Field number and wire type. */
    fun tag(): Pair<Int, Int> {
        val t = varint()
        return (t ushr 3).toInt() to (t and 7).toInt()
    }

    /** A length-delimited field as its own reader. */
    fun message(): ProtoReader {
        val len = varint()
        require(len in 0..(end - pos).toLong()) { "Bad length $len" }
        val r = ProtoReader(buf, pos, pos + len.toInt())
        pos += len.toInt()
        return r
    }

    fun skip(wire: Int) {
        when (wire) {
            0 -> varint()
            1 -> advance(8)
            2 -> message()
            5 -> advance(4)
            else -> throw IllegalArgumentException("Unsupported wire type $wire")
        }
    }

    private fun advance(n: Int) {
        require(end - pos >= n) { "Truncated field" }
        pos += n
    }
}
