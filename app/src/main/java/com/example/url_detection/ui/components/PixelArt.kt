package com.example.url_detection.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.url_detection.data.SignalLevel
import com.example.url_detection.data.Verdict
import com.example.url_detection.ui.theme.Fixed
import kotlin.math.floor
import kotlin.math.max

// Sprite grids from the handover board. X is outline ink, c is inner fill, a dot is empty.
private val SHIELD = listOf(
    ".XXXXXXXXXX.",
    "XccccccccccX",
    "XccccccccccX",
    "XccccccccccX",
    "XccccccccccX",
    "XccccccccccX",
    "XccccccccccX",
    ".XccccccccX.",
    ".XccccccccX.",
    "..XccccccX..",
    "...XccccX...",
    "....XccX....",
    ".....XX.....",
)

private fun symbolGrid(rows: Map<Int, String>): List<String> =
    List(13) { rows[it] ?: "............" }

private val CHECK = symbolGrid(
    mapOf(3 to "........XX..", 4 to ".......XX...", 5 to "..XX..XX....", 6 to "...XXXX.....", 7 to "....XX......")
)
private val EXCLAMATION = symbolGrid(
    mapOf(2 to ".....XX.....", 3 to ".....XX.....", 4 to ".....XX.....", 5 to ".....XX.....", 6 to ".....XX.....", 8 to ".....XX.....")
)
private val CROSS = symbolGrid(
    mapOf(2 to "..XX....XX..", 3 to "...XX..XX...", 4 to "....XXXX....", 5 to "....XXXX....", 6 to "...XX..XX...", 7 to "..XX....XX..")
)

private val LEAF = listOf(
    "...XX...",
    "..XXXX..",
    ".XXXXXX.",
    "XXXXXXXX",
    ".XXXXXX.",
    "..XXXX..",
    "...XX...",
    "....X...",
)

/** Trims empty rows and columns so a symbol can be drawn on its own in a signal tile. */
private fun trimmed(rows: List<String>): List<String> {
    val used = rows.filter { it.contains('X') }
    val first = used.minOf { it.indexOf('X') }
    val last = used.maxOf { it.lastIndexOf('X') }
    return used.map { it.substring(first, last + 1) }
}

private val CHECK_ONLY = trimmed(CHECK)
private val EXCLAMATION_ONLY = trimmed(EXCLAMATION)
private val CROSS_ONLY = trimmed(CROSS)

private fun symbolFor(verdict: Verdict) = when (verdict) {
    Verdict.Safe -> CHECK
    Verdict.Suspicious -> EXCLAMATION
    Verdict.Dangerous -> CROSS
}

/**
 * Draws ASCII grids as whole-pixel squares, layered in order. The cell size is
 * floored to whole pixels so there is no anti-aliasing between cells.
 */
@Composable
fun PixelSprite(layers: List<List<String>>, colors: Map<Char, Color>, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        val cols = layers.first().first().length
        val rowsCount = layers.first().size
        val u = max(1f, floor(minOf(size.width / cols, size.height / rowsCount)))
        val ox = floor((size.width - u * cols) / 2)
        val oy = floor((size.height - u * rowsCount) / 2)
        layers.forEach { rows ->
            rows.forEachIndexed { y, row ->
                row.forEachIndexed { x, ch ->
                    colors[ch]?.let { drawRect(it, Offset(ox + x * u, oy + y * u), Size(u, u)) }
                }
            }
        }
    }
}

/** The 12 x 13 verdict shield. [scale] is dp per cell: 7 on results, 2 on list tiles. */
@Composable
fun PixelShield(verdict: Verdict, scale: Dp, modifier: Modifier = Modifier) {
    PixelSprite(
        layers = listOf(SHIELD, symbolFor(verdict)),
        colors = mapOf('X' to Fixed.plum, 'c' to Fixed.cream),
        modifier = modifier.size(scale * 12, scale * 13),
    )
}

/** Just the symbol, for 32 dp signal tiles. */
@Composable
fun PixelSymbol(level: SignalLevel, modifier: Modifier = Modifier, scale: Dp = 2.dp) {
    val grid = when (level) {
        SignalLevel.Good -> CHECK_ONLY
        SignalLevel.Warn -> EXCLAMATION_ONLY
        SignalLevel.Bad -> CROSS_ONLY
    }
    PixelSprite(
        layers = listOf(grid),
        colors = mapOf('X' to Fixed.plum),
        modifier = modifier.size(scale * grid.first().length, scale * grid.size),
    )
}

/** An 8 x 8 autumn leaf. Decoration only. */
@Composable
fun PixelLeaf(color: Color, size: Dp, modifier: Modifier = Modifier) {
    PixelSprite(layers = listOf(LEAF), colors = mapOf('X' to color), modifier = modifier.size(size))
}
