package com.example.url_detection.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.url_detection.data.CheckEntity
import com.example.url_detection.data.Verdict
import com.example.url_detection.data.displayHost
import com.example.url_detection.ui.components.ArcadeButton
import com.example.url_detection.ui.components.CheckRow
import com.example.url_detection.ui.components.Duck
import com.example.url_detection.ui.components.ListCard
import com.example.url_detection.ui.components.PressableArcade
import com.example.url_detection.ui.components.ScreenHeader
import com.example.url_detection.ui.components.SectionTitle
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType

@Composable
fun HistoryScreen(
    checks: List<CheckEntity>,
    onBack: () -> Unit,
    onOpenCheck: (Long) -> Unit,
) {
    val c = Rest.colors
    // null means All
    var filterName by rememberSaveable { mutableStateOf<String?>(null) }
    val filter = filterName?.let { Verdict.valueOf(it) }
    val groups = remember(checks, filter) {
        checks.filter { filter == null || it.verdict == filter }
            .groupBy { localDateOf(it.checkedAt) }
            .toList()
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(c.bg)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(top = 16.dp),
    ) {
        Column(Modifier.padding(horizontal = 20.dp)) {
            ScreenHeader("History", onBack, big = true)
        }
        Spacer(Modifier.height(18.dp))

        if (checks.isEmpty()) {
            EmptyHistory(onBack)
            return@Column
        }

        Row(
            Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            FilterChip("All", filter == null) { filterName = null }
            Verdict.entries.forEach { v ->
                FilterChip(v.label, filter == v) { filterName = v.name }
            }
        }

        LazyColumn(
            contentPadding = PaddingValues(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (groups.isEmpty()) {
                item {
                    Text(
                        "No ${filter?.label?.lowercase()} checks yet.",
                        style = RestType.bodyLarge,
                        color = c.ink2,
                        modifier = Modifier.padding(top = 12.dp),
                    )
                }
            }
            groups.forEach { (day, dayChecks) ->
                item(key = "day-$day") {
                    SectionTitle(dayLabel(day), Modifier.padding(top = 10.dp).semantics { heading() })
                }
                items(dayChecks, key = { it.id }) { check ->
                    ListCard(radius = 16.dp, padding = 14.dp) {
                        CheckRow(
                            host = displayHost(check.url),
                            caption = checkCaption(check),
                            verdict = check.verdict,
                            onClick = { onOpenCheck(check.id) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun FilterChip(label: String, selected: Boolean, onClick: () -> Unit) {
    val c = Rest.colors
    PressableArcade(
        onClick = onClick,
        fill = if (selected) c.accent else c.surface,
        line = if (selected) c.line else c.lineSoft,
        radius = 14.dp,
        shadow = if (selected) 4.dp else 0.dp,
        room = 4.dp,
        role = Role.Tab,
        modifier = Modifier.semantics { this.selected = selected },
        innerModifier = Modifier.height(46.dp),
    ) {
        Text(
            label,
            style = RestType.buttonSecondary,
            color = if (selected) c.onAccent else c.ink,
            modifier = Modifier.padding(horizontal = 12.dp),
        )
    }
}

@Composable
private fun EmptyHistory(onCheckLink: () -> Unit) {
    val c = Rest.colors
    Column(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Duck(Modifier.width(120.dp))
        Spacer(Modifier.height(18.dp))
        Text("No checks yet", style = RestType.screenTitle, color = c.ink)
        Spacer(Modifier.height(8.dp))
        Text(
            "Links you check and QR codes you scan show up here. They stay on this phone.",
            style = RestType.bodyLarge,
            color = c.ink2,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(22.dp))
        ArcadeButton("Check a link", onCheckLink)
    }
}
