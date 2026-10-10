package com.example.url_detection.data

import kotlin.math.roundToInt

/** The three result states. Colour, symbol and word always move together. */
enum class Verdict(val label: String, val advice: String) {
    Safe("Safe", "Looks safe to open."),
    Suspicious("Suspicious", "Open only if you trust the source."),
    Dangerous("Dangerous", "Don't open this link.");

    companion object {
        // Cut-offs on the probability that a link is malicious. Tune these against
        // the validation set.
        const val SUSPICIOUS_FROM = 0.30f
        const val DANGEROUS_FROM = 0.70f

        fun fromProbability(maliciousProbability: Float): Verdict = when {
            maliciousProbability < SUSPICIOUS_FROM -> Safe
            maliciousProbability < DANGEROUS_FROM -> Suspicious
            else -> Dangerous
        }
    }
}

/**
 * Google Safe Browsing can only make the verdict stricter. A match is Dangerous whatever
 * the model says; no match keeps the model's verdict, because new phishing links are not
 * on Google's lists yet.
 */
fun finalVerdict(maliciousProbability: Float, safeBrowsing: SafeBrowsingStatus): Verdict =
    if (safeBrowsing == SafeBrowsingStatus.Threat) Verdict.Dangerous else Verdict.fromProbability(maliciousProbability)

/** How sure the model is of the side it leans to, 0.5..1. */
fun confidenceOf(maliciousProbability: Float): Float =
    if (maliciousProbability >= 0.5f) maliciousProbability else 1f - maliciousProbability

/** Filled confidence blocks out of ten: 93.69% fills 9. */
fun confidenceBlocks(confidence: Float): Int = (confidence * 10).roundToInt().coerceIn(0, 10)

fun formatConfidence(confidence: Float): String = "%.2f%%".format(confidence * 100)
