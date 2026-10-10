package com.example.url_detection.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.error
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.url_detection.data.SignalLevel
import com.example.url_detection.data.Verdict
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType

/** Link input with a trailing Paste button. Paste reads the clipboard on tap only. */
@Composable
fun LinkField(
    value: String,
    onValueChange: (String) -> Unit,
    onPaste: () -> Unit,
    onGo: () -> Unit,
    error: String?,
    modifier: Modifier = Modifier,
) {
    val c = Rest.colors
    val shape = RoundedCornerShape(14.dp)
    Column(modifier) {
        Row(
            Modifier
                .fillMaxWidth()
                .height(56.dp)
                .clip(shape)
                .background(c.surface2, shape)
                .border(3.dp, if (error != null) c.danger else c.line, shape)
                .padding(start = 14.dp, end = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(RestIcons.Link, contentDescription = null, tint = c.ink2, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Box(Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
                if (value.isEmpty()) {
                    Text("Paste or type a link", style = RestType.urlList.copy(fontWeight = null), color = c.ink2, maxLines = 1)
                }
                BasicTextField(
                    value = value,
                    onValueChange = onValueChange,
                    singleLine = true,
                    textStyle = RestType.urlList.copy(color = c.ink),
                    cursorBrush = SolidColor(c.ink),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Uri,
                        imeAction = ImeAction.Go,
                        autoCorrectEnabled = false,
                        capitalization = KeyboardCapitalization.None,
                    ),
                    keyboardActions = KeyboardActions(onGo = { onGo() }),
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics {
                            contentDescription = "Link to check"
                            if (error != null) error(error)
                        },
                )
            }
            Spacer(Modifier.width(8.dp))
            val pasteShape = RoundedCornerShape(10.dp)
            Box(
                Modifier
                    .height(42.dp)
                    .clip(pasteShape)
                    .background(c.sun, pasteShape)
                    .border(2.dp, Fixed.plum, pasteShape)
                    .clickable(role = Role.Button, onClick = onPaste)
                    .padding(horizontal = 14.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text("Paste", style = RestType.buttonSecondary.copy(fontSize = 13.sp), color = c.onVerdict)
            }
        }
        if (error != null) {
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(20.dp).clip(RoundedCornerShape(6.dp)).background(c.danger).border(2.dp, Fixed.plum, RoundedCornerShape(6.dp)),
                    contentAlignment = Alignment.Center,
                ) { PixelSymbol(SignalLevel.Warn, scale = 1.5.dp) }
                Spacer(Modifier.width(8.dp))
                Text(error, style = RestType.label.copy(fontSize = 14.sp), color = c.ink)
            }
        }
    }
}

/** Navy tile that opens the scanner, with three pixel leaves as decoration. */
@Composable
fun ScanTile(onClick: () -> Unit, modifier: Modifier = Modifier) {
    val c = Rest.colors
    PressableArcade(
        onClick = onClick,
        fill = c.cool,
        line = c.line,
        radius = 20.dp,
        modifier = modifier.fillMaxWidth(),
        innerModifier = Modifier.fillMaxWidth(),
        contentAlignment = Alignment.CenterStart,
    ) {
        Row(Modifier.padding(start = 16.dp, end = 12.dp, top = 16.dp, bottom = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(54.dp).clip(RoundedCornerShape(14.dp)).background(c.onCool),
                contentAlignment = Alignment.Center,
            ) {
                Icon(RestIcons.Qr, contentDescription = null, tint = c.cool, modifier = Modifier.size(32.dp))
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text("Scan a QR code", style = RestType.tileTitle, color = c.onCool)
                Spacer(Modifier.height(4.dp))
                Text("Point, scan, know before you tap.", style = RestType.bodyLarge, color = c.onCool)
            }
            Spacer(Modifier.width(36.dp))
            Icon(RestIcons.Chevron, contentDescription = null, tint = c.onCool, modifier = Modifier.size(28.dp))
        }
        // Leaves: the largest sits on and past the top edge. Decoration only.
        Box(Modifier.matchParentSize()) {
            PixelLeaf(Fixed.pumpkin, 24.dp, Modifier.align(Alignment.TopEnd).offset(x = (-58).dp, y = (-6).dp))
            PixelLeaf(Fixed.mustard, 16.dp, Modifier.align(Alignment.TopEnd).offset(x = (-36).dp, y = 22.dp))
            PixelLeaf(Fixed.maple, 14.dp, Modifier.align(Alignment.BottomEnd).offset(x = (-18).dp, y = (-12).dp))
        }
    }
}

/**
 * A themed confirmation dialog. The safe choice is the primary button.
 */
@Composable
fun ArcadeDialog(
    title: String,
    message: String,
    confirmText: String,
    onConfirm: () -> Unit,
    dismissText: String,
    onDismiss: () -> Unit,
    verdict: Verdict? = null,
    confirmIsDestructive: Boolean = false,
) {
    val c = Rest.colors
    Dialog(onDismissRequest = onDismiss) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(end = 4.dp, bottom = 4.dp)
                .arcadeBox(c.surface, c.line, radius = 20.dp)
                .padding(20.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (verdict != null) {
                    VerdictTile(verdict)
                    Spacer(Modifier.width(12.dp))
                }
                Text(title, style = RestType.screenTitle, color = c.ink)
            }
            Spacer(Modifier.height(12.dp))
            Text(message, style = RestType.bodyLarge, color = c.ink2)
            Spacer(Modifier.height(20.dp))
            if (confirmIsDestructive) {
                // Destructive confirm: danger fill, cancel stays plain.
                ArcadeButton(confirmText, onConfirm, fill = c.danger, contentColor = c.onVerdict)
                Spacer(Modifier.height(10.dp))
                SecondaryButton(dismissText, onDismiss, modifier = Modifier.fillMaxWidth().padding(end = 4.dp))
            } else {
                // Risky confirm: the safe choice is the big button; opening stays secondary.
                ArcadeButton(dismissText, onDismiss)
                Spacer(Modifier.height(10.dp))
                SecondaryButton(confirmText, onConfirm, modifier = Modifier.fillMaxWidth().padding(end = 4.dp))
            }
        }
    }
}
