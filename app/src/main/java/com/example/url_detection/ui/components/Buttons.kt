package com.example.url_detection.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType

/**
 * A box that sits on a hard shadow and sinks 2 dp right and down while pressed.
 * Reserves [shadow] of room on the right and bottom so the shadow is not clipped.
 */
@Composable
fun PressableArcade(
    onClick: () -> Unit,
    fill: Color,
    line: Color,
    radius: Dp,
    modifier: Modifier = Modifier,
    shadow: Dp = 4.dp,
    room: Dp = shadow,
    stroke: Dp = 3.dp,
    enabled: Boolean = true,
    role: Role = Role.Button,
    contentAlignment: Alignment = Alignment.Center,
    innerModifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val sink = if (pressed) shadow / 2 else 0.dp
    Box(
        modifier
            .clickable(interaction, indication = null, enabled = enabled, role = role, onClick = onClick)
            .padding(end = room, bottom = room),
    ) {
        Box(
            innerModifier
                .offset(sink, sink)
                .arcadeBox(fill, line, radius = radius, shadow = shadow - sink, stroke = stroke),
            contentAlignment = contentAlignment,
        ) { content() }
    }
}

/** Primary action: accent fill, Bricolage label, 4 dp hard shadow. */
@Composable
fun ArcadeButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    fill: Color = Rest.colors.accent,
    contentColor: Color = Rest.colors.onAccent,
) {
    PressableArcade(
        onClick = onClick,
        fill = fill,
        line = Rest.colors.line,
        radius = 16.dp,
        modifier = modifier.fillMaxWidth(),
        innerModifier = Modifier.fillMaxWidth().height(56.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(text, style = RestType.buttonPrimary, color = contentColor)
            if (icon != null) {
                Spacer(Modifier.width(10.dp))
                Icon(icon, contentDescription = null, tint = contentColor, modifier = Modifier.size(22.dp))
            }
        }
    }
}

/** Secondary action: surface fill, outline, no shadow. */
@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    fill: Color = Rest.colors.surface,
) {
    val c = Rest.colors
    val shape = RoundedCornerShape(14.dp)
    Row(
        modifier
            .height(52.dp)
            .clip(shape)
            .background(fill, shape)
            .border(3.dp, c.line, shape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 14.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = c.ink, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
        }
        Text(text, style = RestType.buttonSecondary, color = c.ink)
    }
}

/** Underlined text with no container, e.g. Open anyway and See all. */
@Composable
fun TextAction(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    color: Color = Rest.colors.ink,
) {
    Box(
        modifier
            .defaultMinSize(minHeight = 44.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 6.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, style = RestType.textAction.copy(textDecoration = TextDecoration.Underline), color = color)
    }
}

/** 46 dp square icon button used for back and profile. */
@Composable
fun HeaderButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = Rest.colors
    PressableArcade(
        onClick = onClick,
        fill = c.surface,
        line = c.line,
        radius = 14.dp,
        shadow = 2.dp,
        modifier = modifier,
        innerModifier = Modifier.size(46.dp),
    ) {
        Icon(icon, contentDescription = contentDescription, tint = c.ink, modifier = Modifier.size(22.dp))
    }
}

/** Header button over the camera: plum fill, cream outline, the same in both themes. */
@Composable
fun ScannerButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    tint: Color,
    modifier: Modifier = Modifier,
) {
    val shape = RoundedCornerShape(14.dp)
    Box(
        modifier
            .size(46.dp)
            .clip(shape)
            .background(Fixed.camera, shape)
            .border(3.dp, Fixed.creamTile, shape)
            .clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = contentDescription, tint = tint, modifier = Modifier.size(22.dp))
    }
}
