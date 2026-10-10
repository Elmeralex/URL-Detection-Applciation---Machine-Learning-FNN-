package com.example.url_detection.ui.theme

import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import com.example.url_detection.data.SignalLevel
import com.example.url_detection.data.Verdict

/** Colour roles from the Design tokens board. Code refers to roles, never to hex values. */
@Immutable
class ArcadeColors(
    val bg: Color, val surface: Color, val surface2: Color,
    val ink: Color, val ink2: Color, val line: Color, val lineSoft: Color,
    val accent: Color, val onAccent: Color, val sun: Color,
    val cool: Color, val onCool: Color, val link: Color,
    val leaf: Color, val warn: Color, val danger: Color, val onVerdict: Color,
    val isDark: Boolean,
) {
    fun verdict(v: Verdict): Color = when (v) {
        Verdict.Safe -> leaf
        Verdict.Suspicious -> warn
        Verdict.Dangerous -> danger
    }

    fun signal(level: SignalLevel): Color = when (level) {
        SignalLevel.Good -> leaf
        SignalLevel.Warn -> warn
        SignalLevel.Bad -> danger
    }
}

val LightArcade = ArcadeColors(
    bg = Color(0xFFFFF0D4), surface = Color(0xFFFFFBF2), surface2 = Color(0xFFFFE6B3),
    ink = Color(0xFF2B1230), ink2 = Color(0xFF6B4A5F),
    line = Color(0xFF2B1230), lineSoft = Color(0xFF2B1230),
    accent = Color(0xFFFF6A1F), onAccent = Color(0xFF2B1230), sun = Color(0xFFFFC82C),
    cool = Color(0xFF24408E), onCool = Color(0xFFFFF6E0), link = Color(0xFF24408E),
    leaf = Color(0xFF7FD35E), warn = Color(0xFFFFC82C), danger = Color(0xFFFF5C47),
    onVerdict = Color(0xFF2B1230),
    isDark = false,
)

val DarkArcade = ArcadeColors(
    bg = Color(0xFF170A1C), surface = Color(0xFF241230), surface2 = Color(0xFF36193F),
    ink = Color(0xFFFFF0D4), ink2 = Color(0xFFCFB5C8),
    line = Color(0xFFFFF0D4), lineSoft = Color(0xFF5B3368),
    accent = Color(0xFFFF8A3D), onAccent = Color(0xFF170A1C), sun = Color(0xFFFFC82C),
    cool = Color(0xFF9DB4FF), onCool = Color(0xFF170A1C), link = Color(0xFF9DB4FF),
    leaf = Color(0xFF7FD35E), warn = Color(0xFFFFC82C), danger = Color(0xFFFF5C47),
    onVerdict = Color(0xFF2B1230),
    isDark = true,
)

/** Fixed colours that are the same in both themes. */
object Fixed {
    val plum = Color(0xFF2B1230)       // pixel outline ink
    val cream = Color(0xFFFFFBF2)      // shield inner fill
    val creamTile = Color(0xFFFFF0D4)  // tile behind duck-ae
    val pumpkin = Color(0xFFFF6A1F)
    val maple = Color(0xFFE23D2E)
    val mustard = Color(0xFFFFC82C)
    val camera = Color(0xFF1C0F22)
    val scanLine = Color(0xFFFF8A3D)
}

val LocalArcade = staticCompositionLocalOf { LightArcade }
