package com.example.url_detection.ui.screens

import com.example.url_detection.data.CheckEntity
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

private val timeFormat = DateTimeFormatter.ofLocalizedTime(FormatStyle.SHORT)
private val dayFormat = DateTimeFormatter.ofPattern("EEEE, d MMM")

fun localDateOf(millis: Long): LocalDate =
    Instant.ofEpochMilli(millis).atZone(ZoneId.systemDefault()).toLocalDate()

/** "Link · 2 min ago" today, "QR code · 9:14 PM" on earlier days. */
fun checkCaption(check: CheckEntity, now: Long = System.currentTimeMillis()): String {
    val whenText = if (localDateOf(check.checkedAt) == localDateOf(now)) {
        val minutes = (now - check.checkedAt) / 60_000
        when {
            minutes < 1 -> "Just now"
            minutes < 60 -> "$minutes min ago"
            else -> "${minutes / 60} h ago"
        }
    } else {
        Instant.ofEpochMilli(check.checkedAt).atZone(ZoneId.systemDefault()).format(timeFormat)
    }
    return "${check.source.label} · $whenText"
}

fun dayLabel(date: LocalDate, today: LocalDate = LocalDate.now()): String = when (date) {
    today -> "Today"
    today.minusDays(1) -> "Yesterday"
    else -> date.format(dayFormat)
}
