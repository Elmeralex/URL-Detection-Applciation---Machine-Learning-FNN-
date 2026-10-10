package com.example.url_detection.ui.screens

import android.os.SystemClock
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.url_detection.data.SafeBrowsingResult
import com.example.url_detection.data.SignalLevel
import com.example.url_detection.data.allSignals
import com.example.url_detection.data.displayHost
import com.example.url_detection.ui.components.ArcadeButton
import com.example.url_detection.ui.components.BobbingDuck
import com.example.url_detection.ui.components.HeaderButton
import com.example.url_detection.ui.components.ListCard
import com.example.url_detection.ui.components.PixelLeaf
import com.example.url_detection.ui.components.PixelSymbol
import com.example.url_detection.ui.components.RestIcons
import com.example.url_detection.ui.components.SecondaryButton
import com.example.url_detection.ui.components.SoftDivider
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType
import kotlinx.coroutines.async
import kotlinx.coroutines.delay

private val LOCAL_STEPS = listOf("Reading the link", "Extracting URL features", "Running the on-device model")
private const val ONLINE_STEP = "Checking Google Safe Browsing"

// Blocks filled once each step is done: the bar follows the pipeline, not the clock.
private val BLOCKS_LOCAL = listOf(0, 3, 6, 10)
private val BLOCKS_ONLINE = listOf(0, 2, 4, 7, 10)

// Each step stays on screen at least this long, so the whole screen lasts >= 400 ms.
private const val MIN_STEP_MS = 150L

/**
 * Runs the pipeline: three on-device steps, then Google Safe Browsing when [onlineCheck]
 * is on. [runModel] returns the malicious probability or null on failure; [checkOnline]
 * never fails, it answers Unavailable instead. [onDone] receives both and decides where to go next.
 */
@Composable
fun AnalyzingScreen(
    url: String,
    onlineCheck: Boolean,
    runModel: suspend (String) -> Float?,
    checkOnline: suspend (String) -> SafeBrowsingResult,
    onDone: suspend (Float, SafeBrowsingResult) -> Unit,
    onCancel: () -> Unit,
) {
    val c = Rest.colors
    var done by remember { mutableIntStateOf(0) }
    var failed by remember { mutableStateOf(false) }
    val latestOnDone by rememberUpdatedState(onDone)
    val steps = if (onlineCheck) LOCAL_STEPS + ONLINE_STEP else LOCAL_STEPS
    val blocksAfterStep = if (onlineCheck) BLOCKS_ONLINE else BLOCKS_LOCAL

    LaunchedEffect(url) {
        suspend fun step(work: suspend () -> Unit) {
            val start = SystemClock.uptimeMillis()
            work()
            val left = MIN_STEP_MS - (SystemClock.uptimeMillis() - start)
            if (left > 0) delay(left)
            done++
        }
        // Started first so the network wait overlaps the on-device steps.
        val online = if (onlineCheck) async { checkOnline(url) } else null
        step { displayHost(url) }
        step { allSignals(url) }
        var p: Float? = null
        step { p = runModel(url) }
        val result = p
        if (result == null) {
            online?.cancel()
            failed = true
            return@LaunchedEffect
        }
        var safeBrowsing = SafeBrowsingResult.Off
        if (online != null) step { safeBrowsing = online.await() }
        latestOnDone(result, safeBrowsing)
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(c.bg)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        HeaderButton(RestIcons.Back, "Cancel and go back", onCancel)

        Column(
            Modifier.weight(1f).fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Box(Modifier.width(240.dp).height(190.dp), contentAlignment = Alignment.Center) {
                BobbingDuck(Modifier.width(150.dp))
                PixelLeaf(Fixed.pumpkin, 32.dp, Modifier.align(Alignment.TopStart).offset(x = 8.dp, y = 16.dp))
                PixelLeaf(Fixed.maple, 20.dp, Modifier.align(Alignment.TopEnd).offset(x = (-14).dp, y = 40.dp))
            }
            Spacer(Modifier.height(18.dp))
            Text(
                if (failed) "Couldn't check this link" else "Sniffing the link…",
                style = RestType.verdict,
                color = c.ink,
                textAlign = TextAlign.Center,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
            )
            Spacer(Modifier.height(12.dp))
            Text(
                displayHost(url),
                style = RestType.urlList,
                color = c.onVerdict,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier
                    .clip(RoundedCornerShape(10.dp))
                    .background(c.sun)
                    .border(3.dp, c.line, RoundedCornerShape(10.dp))
                    .padding(horizontal = 14.dp, vertical = 6.dp),
            )
            Spacer(Modifier.height(22.dp))
            if (failed) {
                Text(
                    "The on-device model didn't load or couldn't read this link. Nothing was opened.",
                    style = RestType.bodyLarge,
                    color = c.ink2,
                    textAlign = TextAlign.Center,
                )
            } else {
                StepBar(filled = blocksAfterStep[done], blinking = done < steps.size)
                Spacer(Modifier.height(18.dp))
                ListCard(padding = 18.dp) {
                    steps.forEachIndexed { i, label ->
                        if (i > 0) SoftDivider()
                        StepRow(label, finished = i < done)
                    }
                }
            }
        }

        if (failed) {
            ArcadeButton("Go back", onCancel)
        } else {
            SecondaryButton("Cancel", onCancel, modifier = Modifier.fillMaxWidth())
        }
    }
}

/** Ten accent blocks; the next one blinks while its step runs. */
@Composable
private fun StepBar(filled: Int, blinking: Boolean) {
    val c = Rest.colors
    val reduceMotion = Rest.reduceMotion
    var blinkOn by remember { mutableStateOf(true) }
    LaunchedEffect(blinking, reduceMotion) {
        if (!blinking || reduceMotion) { blinkOn = true; return@LaunchedEffect }
        while (true) {
            delay(500)
            blinkOn = !blinkOn
        }
    }
    Row(
        Modifier
            .fillMaxWidth()
            .clearAndSetSemantics { contentDescription = "Progress: $filled of 10" }
            .clip(RoundedCornerShape(14.dp))
            .background(c.surface)
            .border(3.dp, c.line, RoundedCornerShape(14.dp))
            .padding(7.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        repeat(10) { i ->
            val color = when {
                i < filled -> c.accent
                i == filled && blinking && blinkOn -> c.accent.copy(alpha = 0.3f)
                else -> c.surface2
            }
            Box(Modifier.weight(1f).height(16.dp).background(color))
        }
    }
}

@Composable
private fun StepRow(label: String, finished: Boolean) {
    val c = Rest.colors
    Row(Modifier.fillMaxWidth().height(52.dp), verticalAlignment = Alignment.CenterVertically) {
        val shape = RoundedCornerShape(8.dp)
        Box(
            Modifier
                .size(28.dp)
                .clip(shape)
                .background(if (finished) c.leaf else c.surface2)
                .border(2.dp, if (finished) Fixed.plum else c.lineSoft.copy(alpha = 0.4f), shape),
            contentAlignment = Alignment.Center,
        ) {
            if (finished) {
                PixelSymbol(SignalLevel.Good)
            } else {
                Box(Modifier.size(8.dp).background(c.ink2.copy(alpha = 0.4f)))
            }
        }
        Spacer(Modifier.width(14.dp))
        Text(label, style = RestType.label.copy(fontSize = RestType.bodyLarge.fontSize), color = c.ink)
    }
}
