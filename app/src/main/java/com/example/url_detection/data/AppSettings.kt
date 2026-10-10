package com.example.url_detection.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

enum class ThemeMode { System, Light, Dark }

private val Context.settingsStore by preferencesDataStore(name = "settings")
private val themeKey = stringPreferencesKey("theme_mode")
private val onlineCheckKey = booleanPreferencesKey("online_check")

class AppSettings(private val context: Context) {
    val mode: Flow<ThemeMode> = context.settingsStore.data.map { prefs ->
        prefs[themeKey]?.let { saved -> ThemeMode.entries.firstOrNull { it.name == saved } } ?: ThemeMode.System
    }

    /** Whether links are also sent to Google Safe Browsing. On unless the user turns it off. */
    val onlineCheck: Flow<Boolean> = context.settingsStore.data.map { prefs -> prefs[onlineCheckKey] ?: true }

    suspend fun setMode(mode: ThemeMode) {
        context.settingsStore.edit { it[themeKey] = mode.name }
    }

    suspend fun setOnlineCheck(on: Boolean) {
        context.settingsStore.edit { it[onlineCheckKey] = on }
    }
}
