package com.example.url_detection.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.example.url_detection.data.ThemeMode
import com.example.url_detection.ui.components.ArcadeDialog
import com.example.url_detection.ui.components.HeroBlock
import com.example.url_detection.ui.components.ListCard
import com.example.url_detection.ui.components.LogoTile
import com.example.url_detection.ui.components.RestIcons
import com.example.url_detection.ui.components.ScreenHeader
import com.example.url_detection.ui.components.SectionTitle
import com.example.url_detection.ui.components.SoftDivider
import com.example.url_detection.ui.components.ThemeCard
import com.example.url_detection.ui.theme.DarkArcade
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.LightArcade
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType
import kotlinx.coroutines.launch

@Composable
fun ProfileScreen(
    themeMode: ThemeMode,
    onlineCheck: Boolean,
    historyCount: Int,
    onThemeChange: (ThemeMode) -> Unit,
    onOnlineCheckChange: (Boolean) -> Unit,
    onClearHistory: suspend () -> Unit,
    onBack: () -> Unit,
) {
    val c = Rest.colors
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var confirmClear by rememberSaveable { mutableStateOf(false) }
    val version = remember {
        runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }.getOrNull() ?: "?"
    }

    Scaffold(
        containerColor = c.bg,
        contentWindowInsets = WindowInsets(0),
        snackbarHost = { SnackbarHost(snackbar) },
    ) { inner ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(inner)
                .background(c.bg)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 16.dp),
        ) {
            ScreenHeader("Profile", onBack, big = true)
            Spacer(Modifier.height(20.dp))

            HeroBlock(fill = c.sun) {
                Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                    LogoTile(size = 88.dp, radius = 18.dp, shadow = 0.dp, line = Fixed.plum)
                    Spacer(Modifier.width(14.dp))
                    Column {
                        Text("AE", style = RestType.greeting, color = c.onVerdict)
                        Text("Secure your browsing", style = RestType.bodyLarge, color = c.onVerdict)
                    }
                }
            }
            Spacer(Modifier.height(24.dp))

            SectionTitle("Theme")
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth().selectableGroup(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                ThemeCard("System", themeMode == ThemeMode.System, swatch = {
                    Row(Modifier.fillMaxSize()) {
                        Box(Modifier.weight(1f).fillMaxHeight().background(LightArcade.bg))
                        Box(Modifier.weight(1f).fillMaxHeight().background(DarkArcade.bg))
                    }
                }, onClick = { onThemeChange(ThemeMode.System) }, modifier = Modifier.weight(1f))
                ThemeCard("Light", themeMode == ThemeMode.Light, swatch = {
                    Box(Modifier.fillMaxSize().background(LightArcade.bg))
                }, onClick = { onThemeChange(ThemeMode.Light) }, modifier = Modifier.weight(1f))
                ThemeCard("Dark", themeMode == ThemeMode.Dark, swatch = {
                    Box(Modifier.fillMaxSize().background(DarkArcade.bg))
                }, onClick = { onThemeChange(ThemeMode.Dark) }, modifier = Modifier.weight(1f))
            }
            Spacer(Modifier.height(6.dp))
            Text("System follows your phone's setting.", style = RestType.body, color = c.ink2)
            Spacer(Modifier.height(22.dp))

            ListCard(padding = 18.dp) {
                Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    IconTile(RestIcons.Lock, fill = c.cool, tint = c.onCool)
                    Spacer(Modifier.width(14.dp))
                    Column {
                        Text("On-device detection", style = RestType.section.copy(fontSize = RestType.bodyLarge.fontSize.times(1.1f)), color = c.ink)
                        Spacer(Modifier.height(2.dp))
                        Text("Links and QR codes are analysed on your phone.", style = RestType.body, color = c.ink2)
                    }
                }
            }
            Spacer(Modifier.height(14.dp))

            ListCard(
                padding = 18.dp,
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .toggleable(value = onlineCheck, role = Role.Switch, onValueChange = onOnlineCheckChange),
            ) {
                Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    IconTile(RestIcons.Link, fill = c.sun, tint = c.onVerdict)
                    Spacer(Modifier.width(14.dp))
                    Column(Modifier.weight(1f)) {
                        Text("Online check", style = RestType.section.copy(fontSize = RestType.bodyLarge.fontSize.times(1.1f)), color = c.ink)
                        Spacer(Modifier.height(2.dp))
                        Text("Also asks Google Safe Browsing. The link you check is sent to Google.", style = RestType.body, color = c.ink2)
                    }
                    Spacer(Modifier.width(12.dp))
                    PixelSwitch(onlineCheck)
                }
            }
            Spacer(Modifier.height(14.dp))

            ListCard(
                padding = 18.dp,
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .clickable(role = Role.Button) {
                        if (historyCount == 0) {
                            scope.launch { snackbar.showSnackbar("History is already empty") }
                        } else {
                            confirmClear = true
                        }
                    },
            ) {
                Row(Modifier.heightIn(min = 52.dp), verticalAlignment = Alignment.CenterVertically) {
                    IconTile(RestIcons.Trash, fill = c.danger, tint = c.onVerdict)
                    Spacer(Modifier.width(14.dp))
                    Text("Clear history", style = RestType.section, color = c.ink)
                }
            }
            Spacer(Modifier.height(24.dp))

            SectionTitle("About")
            Spacer(Modifier.height(10.dp))
            ListCard(padding = 18.dp) {
                AboutRow("Detection model", "Feed-forward network", mono = false)
                SoftDivider()
                AboutRow("Version", version, mono = true)
            }
            Spacer(Modifier.height(12.dp))
        }
    }

    if (confirmClear) {
        ArcadeDialog(
            title = "Clear history?",
            message = "This removes all $historyCount checks from this phone. You can't undo this.",
            confirmText = "Clear history",
            onConfirm = {
                confirmClear = false
                scope.launch {
                    onClearHistory()
                    snackbar.showSnackbar("History cleared")
                }
            },
            dismissText = "Cancel",
            onDismiss = { confirmClear = false },
            confirmIsDestructive = true,
        )
    }
}

@Composable
private fun IconTile(icon: ImageVector, fill: Color, tint: Color) {
    val shape = RoundedCornerShape(10.dp)
    Box(
        Modifier.size(44.dp).clip(shape).background(fill).border(2.dp, Fixed.plum, shape),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(24.dp))
    }
}

/** Square knob on a track, filled leaf when on. Display only: the whole card is the toggle. */
@Composable
private fun PixelSwitch(on: Boolean) {
    val c = Rest.colors
    val shape = RoundedCornerShape(8.dp)
    Box(
        Modifier
            .size(width = 52.dp, height = 30.dp)
            .clip(shape)
            .background(if (on) c.leaf else c.surface2)
            .border(2.dp, Fixed.plum, shape)
            .padding(4.dp),
        contentAlignment = if (on) Alignment.CenterEnd else Alignment.CenterStart,
    ) {
        Box(Modifier.size(18.dp).clip(RoundedCornerShape(4.dp)).background(Fixed.cream).border(2.dp, Fixed.plum, RoundedCornerShape(4.dp)))
    }
}

@Composable
private fun AboutRow(label: String, value: String, mono: Boolean) {
    val c = Rest.colors
    Row(Modifier.fillMaxWidth().heightIn(min = 52.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = RestType.label.copy(fontSize = RestType.bodyLarge.fontSize), color = c.ink, modifier = Modifier.weight(1f))
        Text(value, style = if (mono) RestType.urlList else RestType.bodyLarge, color = c.ink2)
    }
}
