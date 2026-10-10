package com.example.url_detection.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.ArrowForward
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.FlashOff
import androidx.compose.material.icons.outlined.FlashOn
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.outlined.OpenInNew
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Link
import androidx.compose.material.icons.rounded.QrCode2
import androidx.compose.material.icons.rounded.Refresh

/** Stroke icons for everything except the verdict symbols, which are pixel sprites. */
object RestIcons {
    val Back = Icons.AutoMirrored.Rounded.ArrowBack
    val Forward = Icons.AutoMirrored.Rounded.ArrowForward
    val Profile = Icons.Outlined.Person
    val Link = Icons.Rounded.Link
    val Qr = Icons.Rounded.QrCode2
    val Chevron = Icons.Rounded.ChevronRight
    val Copy = Icons.Outlined.ContentCopy
    val Rescan = Icons.Rounded.Refresh
    @Suppress("DEPRECATION")
    val OpenExternal = Icons.Outlined.OpenInNew
    val Lock = Icons.Outlined.Lock
    val Trash = Icons.Outlined.Delete
    val Close = Icons.Rounded.Close
    val FlashOn = Icons.Outlined.FlashOn
    val FlashOff = Icons.Outlined.FlashOff
    val Gallery = Icons.Outlined.Image
}
