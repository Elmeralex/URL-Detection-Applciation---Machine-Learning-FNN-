package com.example.url_detection.ui.theme

import android.provider.Settings
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import com.example.url_detection.data.ThemeMode

/** True when the phone's animations are turned off: show end states instead of motion. */
val LocalReduceMotion = staticCompositionLocalOf { false }

object Rest {
    val colors: ArcadeColors
        @Composable @ReadOnlyComposable get() = LocalArcade.current
    val reduceMotion: Boolean
        @Composable @ReadOnlyComposable get() = LocalReduceMotion.current
}

@Composable
fun isDarkFor(mode: ThemeMode): Boolean = when (mode) {
    ThemeMode.System -> isSystemInDarkTheme()
    ThemeMode.Light -> false
    ThemeMode.Dark -> true
}

@Composable
private fun animated(target: Color): Color =
    animateColorAsState(target, animationSpec = tween(300), label = "themeColor").value

/** Cross-fades every role when the theme changes. */
@Composable
private fun animatedColors(t: ArcadeColors) = ArcadeColors(
    bg = animated(t.bg), surface = animated(t.surface), surface2 = animated(t.surface2),
    ink = animated(t.ink), ink2 = animated(t.ink2), line = animated(t.line), lineSoft = animated(t.lineSoft),
    accent = animated(t.accent), onAccent = animated(t.onAccent), sun = t.sun,
    cool = animated(t.cool), onCool = animated(t.onCool), link = animated(t.link),
    leaf = t.leaf, warn = t.warn, danger = t.danger, onVerdict = t.onVerdict,
    isDark = t.isDark,
)

@Composable
fun RestTheme(mode: ThemeMode, content: @Composable () -> Unit) {
    val dark = isDarkFor(mode)
    val c = animatedColors(if (dark) DarkArcade else LightArcade)
    val context = LocalContext.current
    val reduceMotion = remember {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
    val scheme = if (dark) {
        darkColorScheme(background = c.bg, surface = c.surface, primary = c.accent, onPrimary = c.onAccent, onSurface = c.ink, onBackground = c.ink)
    } else {
        lightColorScheme(background = c.bg, surface = c.surface, primary = c.accent, onPrimary = c.onAccent, onSurface = c.ink, onBackground = c.ink)
    }
    CompositionLocalProvider(LocalArcade provides c, LocalReduceMotion provides reduceMotion) {
        MaterialTheme(colorScheme = scheme, typography = RestTypography) {
            CompositionLocalProvider(
                LocalContentColor provides c.ink,
                LocalTextStyle provides RestType.body,
                content = content,
            )
        }
    }
}
