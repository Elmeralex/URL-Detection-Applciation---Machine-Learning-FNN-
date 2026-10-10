package com.example.url_detection.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.url_detection.R
import com.example.url_detection.data.Signal
import com.example.url_detection.data.Verdict
import com.example.url_detection.data.confidenceBlocks
import com.example.url_detection.data.parseUrl
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType
import kotlinx.coroutines.delay

// ---- Mascot ----

private const val DUCK_ASPECT = 540f / 592f

/**
 * The AE duck. [flip] mirrors it with scaleX(-1); off by default because mirroring
 * would also mirror the AE lettering on its body.
 */
@Composable
fun Duck(modifier: Modifier = Modifier, flip: Boolean = false) {
    Image(
        painter = painterResource(R.drawable.duck_ae),
        contentDescription = null,
        modifier = modifier
            .aspectRatio(DUCK_ASPECT)
            .graphicsLayer { scaleX = if (flip) -1f else 1f },
    )
}

/** The duck bobbing 1 s in 2 steps. Still when animations are off. */
@Composable
fun BobbingDuck(modifier: Modifier = Modifier, flip: Boolean = false, lift: Dp = 4.dp) {
    val reduceMotion = Rest.reduceMotion
    var up by remember { mutableStateOf(false) }
    LaunchedEffect(reduceMotion) {
        if (reduceMotion) return@LaunchedEffect
        while (true) {
            delay(500)
            up = !up
        }
    }
    Duck(modifier.offset(y = if (up) -lift else 0.dp), flip = flip)
}

/** duck-ae on its cream tile: app mark in the header. */
@Composable
fun LogoTile(size: Dp = 46.dp, radius: Dp = 14.dp, shadow: Dp = 2.dp, line: Color = Rest.colors.line) {
    Box(
        Modifier
            .padding(end = shadow, bottom = shadow)
            .size(size)
            .arcadeBox(Fixed.creamTile, line, radius = radius, shadow = shadow, stroke = if (size > 60.dp) 3.dp else 2.dp)
            .padding(size * 0.12f),
        contentAlignment = Alignment.Center,
    ) {
        Duck()
    }
}

// ---- Screen scaffolding ----

@Composable
fun ScreenHeader(title: String, onBack: () -> Unit, big: Boolean = false) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        HeaderButton(
            icon = RestIcons.Back,
            contentDescription = "Back",
            onClick = onBack,
        )
        Spacer(Modifier.width(12.dp))
        Text(title, style = if (big) RestType.bigTitle else RestType.screenTitle, color = Rest.colors.ink)
    }
}

@Composable
fun SectionTitle(text: String, modifier: Modifier = Modifier) {
    Text(text, style = RestType.section, color = Rest.colors.ink, modifier = modifier)
}

@Composable
fun SoftDivider(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth().height(2.dp).background(Rest.colors.lineSoft))
}

/** Card with a quiet outline and no shadow. */
@Composable
fun ListCard(
    modifier: Modifier = Modifier,
    radius: Dp = 20.dp,
    padding: Dp = 16.dp,
    content: @Composable ColumnScope.() -> Unit,
) {
    val c = Rest.colors
    Column(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(radius))
            .background(c.surface)
            .border(3.dp, c.lineSoft, RoundedCornerShape(radius))
            .padding(horizontal = padding, vertical = padding - 4.dp),
        content = content,
    )
}

/** Big block with a strong outline and hard shadow. Content is in on-verdict. */
@Composable
fun HeroBlock(
    fill: Color,
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit,
) {
    Box(
        modifier
            .fillMaxWidth()
            .padding(end = 4.dp, bottom = 4.dp)
            .arcadeBox(fill, Rest.colors.line, radius = 20.dp),
        content = content,
    )
}

// ---- Verdict pieces ----

/** Ten blocks on cream inside a plum outline. Fills left to right, one every 60 ms. */
@Composable
fun ConfidenceBlocks(confidence: Float, modifier: Modifier = Modifier) {
    val target = confidenceBlocks(confidence)
    val reduceMotion = Rest.reduceMotion
    var shown by remember(target) { mutableIntStateOf(if (reduceMotion) target else 0) }
    LaunchedEffect(target, reduceMotion) {
        if (reduceMotion) { shown = target; return@LaunchedEffect }
        for (i in 1..target) {
            delay(60)
            shown = i
        }
    }
    Row(
        modifier
            .fillMaxWidth()
            .clearAndSetSemantics { }
            .clip(RoundedCornerShape(8.dp))
            .background(Fixed.cream)
            .border(3.dp, Fixed.plum, RoundedCornerShape(8.dp))
            .padding(6.dp),
        horizontalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        repeat(10) { i ->
            Box(
                Modifier
                    .weight(1f)
                    .height(12.dp)
                    .background(if (i < shown) Fixed.plum else Fixed.plum.copy(alpha = 0.16f)),
            )
        }
    }
}

@Composable
fun VerdictChip(verdict: Verdict, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(8.dp)
    Text(
        verdict.label,
        style = RestType.chip,
        color = Rest.colors.onVerdict,
        modifier = modifier
            .clip(shape)
            .background(Rest.colors.verdict(verdict), shape)
            .border(2.dp, Fixed.plum, shape)
            .padding(horizontal = 8.dp, vertical = 3.dp),
    )
}

/** 42 dp icon tile with the shield at 2x. */
@Composable
fun VerdictTile(verdict: Verdict, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(10.dp)
    Box(
        modifier
            .size(42.dp)
            .clip(shape)
            .background(Rest.colors.verdict(verdict), shape)
            .border(2.dp, Fixed.plum, shape),
        contentAlignment = Alignment.Center,
    ) {
        PixelShield(verdict, scale = 2.dp)
    }
}

/** 32 dp tile with only the symbol, for the Why this result rows. */
@Composable
fun SignalTile(signal: Signal, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(8.dp)
    Box(
        modifier
            .size(32.dp)
            .clip(shape)
            .background(Rest.colors.signal(signal.level), shape)
            .border(2.dp, Fixed.plum, shape),
        contentAlignment = Alignment.Center,
    ) {
        PixelSymbol(signal.level)
    }
}

/**
 * The link in mono with the registrable domain on a sun highlight and http:// on a
 * danger highlight. Wraps anywhere.
 */
@Composable
fun UrlAnatomyText(url: String, modifier: Modifier = Modifier, maxLines: Int = 4) {
    val c = Rest.colors
    val a = remember(url) { parseUrl(url) }
    // Zero-width spaces let the line break between any two characters.
    fun breakable(s: String) = s.toCharArray().joinToString("​")
    val text = buildAnnotatedString {
        if (a.isHttp) {
            withStyle(SpanStyle(background = c.danger, color = c.onVerdict, fontWeight = FontWeight.Bold)) { append(breakable(a.scheme)) }
        } else {
            withStyle(SpanStyle(color = c.ink2)) { append(breakable(a.scheme)) }
        }
        withStyle(SpanStyle(color = c.ink2)) { append(breakable(a.prefix)) }
        withStyle(SpanStyle(background = c.sun, color = c.onVerdict, fontWeight = FontWeight.Bold)) { append(breakable(a.domain)) }
        withStyle(SpanStyle(color = c.ink2)) { append(breakable(a.suffix)) }
    }
    Text(
        text,
        style = RestType.urlDetail,
        maxLines = maxLines,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier.clearAndSetSemantics { contentDescription = url },
    )
}

// ---- List rows ----

/** Tile, host in mono, caption, chip. Used on Home and History. */
@Composable
fun CheckRow(
    host: String,
    caption: String,
    verdict: Verdict,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = Rest.colors
    Row(
        modifier
            .fillMaxWidth()
            .heightIn(min = 64.dp)
            .clip(RoundedCornerShape(12.dp))
            .clickable(role = Role.Button, onClickLabel = "Open result", onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        VerdictTile(verdict)
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f)) {
            Text(host, style = RestType.urlList, color = c.ink, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Spacer(Modifier.height(2.dp))
            Text(caption, style = RestType.caption.copy(fontSize = RestType.body.fontSize), color = c.ink2, maxLines = 1)
        }
        Spacer(Modifier.width(10.dp))
        VerdictChip(verdict)
    }
}

/** Selected card: sun fill, outline, 4 dp shadow. Others: surface with a quiet outline. */
@Composable
fun ThemeCard(
    label: String,
    selected: Boolean,
    swatch: @Composable () -> Unit,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = Rest.colors
    PressableArcade(
        onClick = onClick,
        fill = if (selected) c.sun else c.surface,
        line = if (selected) c.line else c.lineSoft,
        radius = 16.dp,
        shadow = if (selected) 4.dp else 0.dp,
        room = 4.dp,
        role = Role.RadioButton,
        modifier = modifier,
        innerModifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Box(Modifier.fillMaxWidth().height(44.dp).clip(RoundedCornerShape(10.dp)).border(2.dp, Fixed.plum, RoundedCornerShape(10.dp))) {
                swatch()
            }
            Spacer(Modifier.height(8.dp))
            Text(label, style = RestType.buttonSecondary.copy(fontSize = RestType.textAction.fontSize), color = if (selected) c.onVerdict else c.ink)
        }
    }
}
