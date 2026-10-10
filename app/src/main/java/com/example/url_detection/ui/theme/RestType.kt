@file:OptIn(ExperimentalTextApi::class)

package com.example.url_detection.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.example.url_detection.R

// The three fonts are bundled variable fonts (res/font), so they work offline on first launch.
private fun bricolage(weight: Int) = Font(
    R.font.bricolage_grotesque, FontWeight(weight),
    variationSettings = FontVariation.Settings(FontVariation.weight(weight), FontVariation.Setting("opsz", 32f)),
)

private fun dmSans(weight: Int) = Font(
    R.font.dm_sans, FontWeight(weight),
    variationSettings = FontVariation.Settings(FontVariation.weight(weight)),
)

private fun jetBrains(weight: Int) = Font(
    R.font.jetbrains_mono, FontWeight(weight),
    variationSettings = FontVariation.Settings(FontVariation.weight(weight)),
)

val Bricolage = FontFamily(bricolage(600), bricolage(700))
val DmSans = FontFamily(dmSans(400), dmSans(500), dmSans(600), dmSans(700))
val JetBrainsMono = FontFamily(jetBrains(400), jetBrains(500), jetBrains(700))

/** The type scale from the Design tokens board. */
object RestType {
    val wordmark = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 28.sp, lineHeight = 28.sp, letterSpacing = 0.06.em)
    val greeting = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 40.sp, lineHeight = 42.sp)
    val verdict = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 34.sp, lineHeight = 36.sp)
    val screenTitle = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 24.sp, lineHeight = 28.sp)
    val bigTitle = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 32.sp, lineHeight = 36.sp)
    val section = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 20.sp, lineHeight = 28.sp)
    val tileTitle = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 22.sp, lineHeight = 26.sp)
    val buttonPrimary = TextStyle(fontFamily = Bricolage, fontWeight = FontWeight.Bold, fontSize = 21.sp, lineHeight = 22.sp)
    val buttonSecondary = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Bold, fontSize = 16.sp, lineHeight = 20.sp)
    val body = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Medium, fontSize = 14.sp, lineHeight = 20.sp)
    val bodyLarge = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Medium, fontSize = 16.sp, lineHeight = 22.sp)
    val label = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.SemiBold, fontSize = 15.sp, lineHeight = 20.sp)
    val caption = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Medium, fontSize = 12.sp, lineHeight = 16.sp)
    val overline = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Bold, fontSize = 12.sp, lineHeight = 16.sp, letterSpacing = 0.1.em)
    val chip = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Bold, fontSize = 12.sp, lineHeight = 16.sp)
    val textAction = TextStyle(fontFamily = DmSans, fontWeight = FontWeight.Bold, fontSize = 14.sp, lineHeight = 20.sp)
    val urlDetail = TextStyle(fontFamily = JetBrainsMono, fontWeight = FontWeight.Normal, fontSize = 15.sp, lineHeight = 24.sp)
    val urlList = TextStyle(fontFamily = JetBrainsMono, fontWeight = FontWeight.Medium, fontSize = 14.sp, lineHeight = 20.sp)
}

val RestTypography = Typography(
    bodyLarge = RestType.bodyLarge,
    bodyMedium = RestType.body,
    bodySmall = RestType.caption,
    labelLarge = RestType.buttonSecondary,
    titleLarge = RestType.screenTitle,
    titleMedium = RestType.section,
)
