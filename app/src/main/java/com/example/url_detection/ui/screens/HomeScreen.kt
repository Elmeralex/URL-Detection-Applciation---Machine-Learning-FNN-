package com.example.url_detection.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.example.url_detection.data.CheckEntity
import com.example.url_detection.data.displayHost
import com.example.url_detection.data.looksLikeUrl
import com.example.url_detection.ui.components.ArcadeButton
import com.example.url_detection.ui.components.CheckRow
import com.example.url_detection.ui.components.Duck
import com.example.url_detection.ui.components.HeaderButton
import com.example.url_detection.ui.components.HeroBlock
import com.example.url_detection.ui.components.LinkField
import com.example.url_detection.ui.components.ListCard
import com.example.url_detection.ui.components.LogoTile
import com.example.url_detection.ui.components.PixelLeaf
import com.example.url_detection.ui.components.RestIcons
import com.example.url_detection.ui.components.ScanTile
import com.example.url_detection.ui.components.SectionTitle
import com.example.url_detection.ui.components.SoftDivider
import com.example.url_detection.ui.components.TextAction
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType

@Composable
fun HomeScreen(
    recent: List<CheckEntity>,
    onCheck: (String) -> Unit,
    onScan: () -> Unit,
    onOpenCheck: (Long) -> Unit,
    onSeeAll: () -> Unit,
    onProfile: () -> Unit,
) {
    val c = Rest.colors
    val clipboard = LocalClipboardManager.current
    val focus = LocalFocusManager.current
    var link by rememberSaveable { mutableStateOf("") }
    var error by rememberSaveable { mutableStateOf<String?>(null) }

    fun submit() {
        val t = link.trim()
        error = when {
            t.isEmpty() -> "Paste or type a link first."
            !looksLikeUrl(t) -> "That doesn't look like a link. Try something like example.com."
            else -> null
        }
        if (error == null) {
            focus.clearFocus()
            link = ""
            onCheck(t)
        }
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(c.bg)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        // Header: logo tile, wordmark, profile.
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            LogoTile()
            Spacer(Modifier.width(10.dp))
            Text("REST", style = RestType.wordmark, color = c.ink, modifier = Modifier.weight(1f).semantics { heading() })
            HeaderButton(RestIcons.Profile, "Profile and settings", onProfile)
        }
        Spacer(Modifier.height(18.dp))

        // Mustard hero with the duck overlapping the bottom edge.
        Box {
            HeroBlock(fill = c.sun) {
                Column(Modifier.padding(start = 20.dp, top = 22.dp, bottom = 26.dp, end = 132.dp)) {
                    Text("SECURE YOUR BROWSING", style = RestType.overline, color = c.onVerdict, maxLines = 1, softWrap = false)
                    Spacer(Modifier.height(8.dp))
                    Text("Hey, AE!", style = RestType.greeting, color = c.onVerdict, modifier = Modifier.semantics { heading() })
                    Spacer(Modifier.height(10.dp))
                    Text("Got a link that looks fishy? I'll sniff it out first.", style = RestType.bodyLarge, color = c.onVerdict)
                }
                PixelLeaf(Fixed.pumpkin, 24.dp, Modifier.align(Alignment.TopEnd).offset(x = (-44).dp, y = 14.dp))
            }
            Duck(
                Modifier
                    .align(Alignment.BottomEnd)
                    .padding(end = 18.dp)
                    .width(118.dp)
                    .offset(y = 14.dp),
            )
        }
        Spacer(Modifier.height(30.dp))

        // Drop a link.
        Column(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(c.surface)
                .border(3.dp, c.line, RoundedCornerShape(20.dp))
                .padding(18.dp),
        ) {
            SectionTitle("Drop a link")
            Spacer(Modifier.height(12.dp))
            LinkField(
                value = link,
                onValueChange = { link = it; error = null },
                onPaste = {
                    clipboard.getText()?.text?.trim()?.takeIf { it.isNotEmpty() }?.let { link = it; error = null }
                },
                onGo = ::submit,
                error = error,
            )
            Spacer(Modifier.height(14.dp))
            ArcadeButton("Check link", ::submit, icon = RestIcons.Forward)
        }
        Spacer(Modifier.height(20.dp))

        ScanTile(onScan)
        Spacer(Modifier.height(22.dp))

        // Recent checks.
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            SectionTitle("Recent checks", Modifier.weight(1f).semantics { heading() })
            if (recent.isNotEmpty()) TextAction("See all", onSeeAll, color = c.link)
        }
        Spacer(Modifier.height(8.dp))
        ListCard {
            if (recent.isEmpty()) {
                Text(
                    "No checks yet. Your latest checks will show up here.",
                    style = RestType.body,
                    color = c.ink2,
                    modifier = Modifier.padding(vertical = 14.dp),
                )
            } else {
                recent.forEachIndexed { i, check ->
                    if (i > 0) SoftDivider()
                    CheckRow(
                        host = displayHost(check.url),
                        caption = checkCaption(check),
                        verdict = check.verdict,
                        onClick = { onOpenCheck(check.id) },
                    )
                }
            }
        }
        Spacer(Modifier.height(12.dp))
    }
}
