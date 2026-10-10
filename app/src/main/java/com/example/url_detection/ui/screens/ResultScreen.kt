package com.example.url_detection.ui.screens

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import com.example.url_detection.data.CheckEntity
import com.example.url_detection.data.SafeBrowsingStatus
import com.example.url_detection.data.Verdict
import com.example.url_detection.data.formatConfidence
import com.example.url_detection.data.openableUrl
import com.example.url_detection.data.resultSignals
import com.example.url_detection.data.safeBrowsingSignal
import com.example.url_detection.ui.components.ArcadeButton
import com.example.url_detection.ui.components.ArcadeDialog
import com.example.url_detection.ui.components.ConfidenceBlocks
import com.example.url_detection.ui.components.HeroBlock
import com.example.url_detection.ui.components.ListCard
import com.example.url_detection.ui.components.PixelShield
import com.example.url_detection.ui.components.RestIcons
import com.example.url_detection.ui.components.ScreenHeader
import com.example.url_detection.ui.components.SecondaryButton
import com.example.url_detection.ui.components.SectionTitle
import com.example.url_detection.ui.components.SignalTile
import com.example.url_detection.ui.components.SoftDivider
import com.example.url_detection.ui.components.TextAction
import com.example.url_detection.ui.components.UrlAnatomyText
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType
import kotlinx.coroutines.launch

/**
 * The scorecard. Links are only ever opened from Open link (Safe) or from the
 * confirmation dialog behind Open anyway (Suspicious and Dangerous).
 */
@Composable
fun ResultScreen(
    check: CheckEntity,
    onBack: () -> Unit,
    onGoBack: () -> Unit,
    onRescan: () -> Unit,
) {
    val c = Rest.colors
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var confirmOpen by rememberSaveable { mutableStateOf(false) }

    val verdict = check.verdict
    val confidenceText = formatConfidence(check.confidence)
    // A Safe Browsing match decides the verdict, so it replaces the model's confidence.
    val flagged = check.safeBrowsing == SafeBrowsingStatus.Threat
    val threatText = check.threats.sortedBy { it.code }.map { it.label.lowercase() }.let { names ->
        if (names.size <= 1) names.firstOrNull() ?: "unsafe" else names.dropLast(1).joinToString(", ") + " and " + names.last()
    }
    val evidence = if (flagged) "Flagged by Google Safe Browsing" else "Confidence $confidenceText"
    val signals = remember(check.url, check.safeBrowsing, check.threatTypes) {
        listOf(safeBrowsingSignal(check.safeBrowsing, check.threats)) + resultSignals(check.url)
    }

    fun openLink() {
        try {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(openableUrl(check.url))))
        } catch (e: ActivityNotFoundException) {
            scope.launch { snackbar.showSnackbar("No app on this phone can open this link.") }
        }
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
            ScreenHeader("Analysis result", onBack)
            Spacer(Modifier.height(20.dp))

            // Verdict block: read out by TalkBack when the screen opens.
            HeroBlock(
                fill = c.verdict(verdict),
                modifier = Modifier.clearAndSetSemantics {
                    liveRegion = LiveRegionMode.Polite
                    contentDescription = "${verdict.label}. $evidence. ${verdict.advice}"
                },
            ) {
                Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                    PixelShield(verdict, scale = 7.dp)
                    Spacer(Modifier.width(16.dp))
                    Column(Modifier.weight(1f)) {
                        Text(verdict.label, style = RestType.verdict, color = c.onVerdict)
                        Spacer(Modifier.height(6.dp))
                        Text(evidence, style = RestType.label, color = c.onVerdict)
                        if (!flagged) {
                            Spacer(Modifier.height(8.dp))
                            ConfidenceBlocks(check.confidence)
                        }
                        Spacer(Modifier.height(10.dp))
                        Text(verdict.advice, style = RestType.bodyLarge, color = c.onVerdict)
                    }
                }
            }
            Spacer(Modifier.height(18.dp))

            ListCard(padding = 18.dp) {
                Spacer(Modifier.height(4.dp))
                Text("LINK", style = RestType.overline, color = c.ink2)
                Spacer(Modifier.height(6.dp))
                UrlAnatomyText(check.url)
                Spacer(Modifier.height(6.dp))
            }
            Spacer(Modifier.height(18.dp))

            ListCard(padding = 18.dp) {
                Spacer(Modifier.height(4.dp))
                SectionTitle("Why this result")
                Spacer(Modifier.height(6.dp))
                signals.forEachIndexed { i, s ->
                    if (i > 0) SoftDivider()
                    Row(
                        Modifier.fillMaxWidth().heightIn(min = 52.dp).padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        SignalTile(s)
                        Spacer(Modifier.width(14.dp))
                        Text(s.label, style = RestType.label.copy(fontSize = RestType.bodyLarge.fontSize), color = c.ink, modifier = Modifier.weight(1f))
                        Text(s.value, style = RestType.urlList, color = c.ink)
                    }
                }
            }
            Spacer(Modifier.height(24.dp))

            if (verdict == Verdict.Safe) {
                ArcadeButton("Open link", ::openLink, icon = RestIcons.OpenExternal)
            } else {
                ArcadeButton("Go back", onGoBack)
            }
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth().padding(end = 4.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SecondaryButton(
                    "Copy",
                    onClick = {
                        clipboard.setText(AnnotatedString(check.url))
                        scope.launch { snackbar.showSnackbar("Link copied") }
                    },
                    icon = RestIcons.Copy,
                    modifier = Modifier.weight(1f),
                )
                SecondaryButton("Rescan", onRescan, icon = RestIcons.Rescan, modifier = Modifier.weight(1f))
            }
            if (verdict != Verdict.Safe) {
                Spacer(Modifier.height(8.dp))
                Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    TextAction(
                        if (verdict == Verdict.Dangerous) "Open anyway (not recommended)" else "Open anyway",
                        onClick = { confirmOpen = true },
                    )
                }
            }
        }
    }

    if (confirmOpen) {
        val dangerous = verdict == Verdict.Dangerous
        ArcadeDialog(
            title = if (dangerous) "Open a dangerous link?" else "Open this link?",
            message = if (flagged) {
                "Google Safe Browsing lists this link as $threatText. Opening it could put your accounts or your phone at risk."
            } else if (dangerous) {
                "REST thinks this link is very likely phishing or malware. Opening it could put your accounts or your phone at risk."
            } else {
                "This link looks suspicious. Only open it if you know and trust where it came from."
            },
            confirmText = "Open anyway",
            onConfirm = { confirmOpen = false; openLink() },
            dismissText = "Go back",
            onDismiss = { confirmOpen = false },
            verdict = verdict,
        )
    }
}
