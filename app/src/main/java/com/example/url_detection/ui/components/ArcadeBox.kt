package com.example.url_detection.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Fill, outline and a hard offset shadow (no blur) in the line colour. The shadow is
 * drawn outside the bounds, so leave [shadow] of room on the right and bottom.
 */
fun Modifier.arcadeBox(
    fill: Color,
    line: Color,
    radius: Dp = 20.dp,
    shadow: Dp = 4.dp,
    stroke: Dp = 3.dp,
): Modifier {
    val shape = RoundedCornerShape(radius)
    return this
        .drawBehind {
            if (shadow > 0.dp) {
                drawRoundRect(
                    color = line,
                    topLeft = Offset(shadow.toPx(), shadow.toPx()),
                    size = size,
                    cornerRadius = CornerRadius(radius.toPx()),
                )
            }
        }
        .background(fill, shape)
        .border(stroke, line, shape)
}
