package com.example.url_detection.data

import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import kotlin.math.log2

/**
 * A link split for display. The pieces concatenate back to the trimmed input:
 * scheme + prefix + domain + suffix.
 */
data class UrlAnatomy(
    val scheme: String,   // "https://", "http://" or ""
    val prefix: String,   // userinfo and subdomains, e.g. "docs."
    val domain: String,   // registrable domain or IP, e.g. "google.com"
    val suffix: String,   // port, path, query, fragment
) {
    val host: String get() = (prefix.substringAfterLast('@') + domain).lowercase()
    val isHttp: Boolean get() = scheme.equals("http://", ignoreCase = true)
    val isHttps: Boolean get() = scheme.equals("https://", ignoreCase = true)
}

private val schemeRegex = Regex("^([a-zA-Z][a-zA-Z0-9+.-]*://)?([^/?#]*)(.*)$", RegexOption.DOT_MATCHES_ALL)
private val ipv4Regex = Regex("""^\d{1,3}(\.\d{1,3}){3}$""")

fun isIpAddress(host: String): Boolean =
    ipv4Regex.matches(host) || host.startsWith("[") || host.all { it.isDigit() } && host.isNotEmpty()

/** Finds the registrable domain (eTLD+1) with the public suffix list. Parsing only, no network. */
fun registrableDomain(host: String): String? {
    if (host.isEmpty() || isIpAddress(host)) return null
    return "https://$host/".toHttpUrlOrNull()?.topPrivateDomain()
}

fun parseUrl(raw: String): UrlAnatomy {
    val text = raw.trim()
    val m = schemeRegex.find(text) ?: return UrlAnatomy("", "", "", text)
    val scheme = m.groupValues[1]
    val authority = m.groupValues[2]
    val rest = m.groupValues[3]

    val hostStart = authority.lastIndexOf('@') + 1
    val hostAndPort = authority.substring(hostStart)
    val hostText = if (hostAndPort.startsWith("[")) {
        hostAndPort.substringBefore("]") + "]"
    } else {
        hostAndPort.substringBefore(':')
    }
    val bareHost = hostText.trimEnd('.').lowercase()

    val domain = if (isIpAddress(bareHost)) bareHost else registrableDomain(bareHost) ?: bareHost
    val domainStart = if (domain.isNotEmpty() && bareHost.endsWith(domain)) {
        hostStart + bareHost.length - domain.length
    } else {
        return UrlAnatomy(scheme, "", "", authority + rest)
    }
    val domainEnd = domainStart + domain.length
    return UrlAnatomy(
        scheme = scheme,
        prefix = authority.substring(0, domainStart),
        domain = authority.substring(domainStart, domainEnd),
        suffix = authority.substring(domainEnd) + rest,
    )
}

/** Host for list rows, e.g. "docs.google.com". Falls back to the raw text. */
fun displayHost(raw: String): String = parseUrl(raw).host.ifEmpty { raw.trim() }

/** A loose check before running the model: something with a dot and no spaces. */
fun looksLikeUrl(raw: String): Boolean {
    val t = raw.trim()
    if (t.isEmpty() || t.any { it.isWhitespace() }) return false
    val host = parseUrl(t).host
    return host.contains('.') || isIpAddress(host)
}

/** The URI to hand to a browser. Links without a scheme open as https. */
fun openableUrl(raw: String): String {
    val t = raw.trim()
    return if (schemeRegex.find(t)?.groupValues?.get(1).isNullOrEmpty()) "https://$t" else t
}

// ---- Why this result ----

enum class SignalLevel { Good, Warn, Bad }

data class Signal(val label: String, val value: String, val level: SignalLevel)

/** Shannon entropy in bits per character of the link without its scheme. */
fun urlEntropy(raw: String): Double {
    val a = parseUrl(raw)
    val s = a.prefix + a.domain + a.suffix
    if (s.isEmpty()) return 0.0
    val n = s.length.toDouble()
    return -s.groupingBy { it }.eachCount().values.sumOf { c -> (c / n) * log2(c / n) }
}

/**
 * Display-only signals read straight from the link. The model only sees TF-IDF,
 * so these explain what the link looks like, not how the model weighed it.
 */
fun allSignals(raw: String): List<Signal> {
    val a = parseUrl(raw)
    val https = when {
        a.isHttps -> Signal("Uses HTTPS", "Yes", SignalLevel.Good)
        a.isHttp -> Signal("Uses HTTPS", "No", SignalLevel.Bad)
        else -> Signal("Uses HTTPS", "Not given", SignalLevel.Warn)
    }

    val ip = isIpAddress(a.domain.lowercase())
    val ipSignal = Signal("IP address as domain", if (ip) "Yes" else "No", if (ip) SignalLevel.Bad else SignalLevel.Good)

    val subCount = if (ip) 0 else a.prefix.substringAfterLast('@').split('.').count { it.isNotEmpty() }
    val subLevel = when {
        subCount <= 1 -> SignalLevel.Good
        subCount <= 3 -> SignalLevel.Warn
        else -> SignalLevel.Bad
    }
    val subSignal = Signal("Subdomains", subCount.toString(), subLevel)

    val e = urlEntropy(raw)
    val (word, eLevel) = when {
        e < 3.5 -> "Low" to SignalLevel.Good
        e < 4.2 -> "Normal" to SignalLevel.Good
        e < 4.7 -> "High" to SignalLevel.Warn
        else -> "Very high" to SignalLevel.Bad
    }
    val randomSignal = Signal("URL randomness", word, eLevel)

    return listOf(https, ipSignal, subSignal, randomSignal)
}

/** Three rows for the result card: HTTPS first, then the two most notable of the rest. */
fun resultSignals(raw: String): List<Signal> {
    val all = allSignals(raw)
    val rest = all.drop(1).sortedByDescending { it.level.ordinal } // stable: keeps order on ties
    return listOf(all.first()) + rest.take(2)
}

/** The Google Safe Browsing row. Shown first on the result card because a match decides the verdict. */
fun safeBrowsingSignal(status: SafeBrowsingStatus, threats: Set<ThreatType>): Signal {
    val label = "Google Safe Browsing"
    return when (status) {
        SafeBrowsingStatus.Clean -> Signal(label, "No match", SignalLevel.Good)
        SafeBrowsingStatus.Threat -> Signal(label, threats.minByOrNull { it.code }?.label ?: "Listed", SignalLevel.Bad)
        SafeBrowsingStatus.Unavailable -> Signal(label, "Unreachable", SignalLevel.Warn)
        SafeBrowsingStatus.Off -> Signal(label, "Off", SignalLevel.Warn)
        SafeBrowsingStatus.NotChecked -> Signal(label, "Not checked", SignalLevel.Warn)
    }
}
