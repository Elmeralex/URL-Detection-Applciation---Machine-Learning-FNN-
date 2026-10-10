package com.example.url_detection

import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.url_detection.data.AppSettings
import com.example.url_detection.data.CheckDao
import com.example.url_detection.data.CheckEntity
import com.example.url_detection.data.CheckSource
import com.example.url_detection.data.RestDatabase
import com.example.url_detection.data.SafeBrowsingClient
import com.example.url_detection.data.ThemeMode
import com.example.url_detection.data.encodeThreats
import com.example.url_detection.ui.screens.AnalyzingScreen
import com.example.url_detection.ui.screens.HistoryScreen
import com.example.url_detection.ui.screens.HomeScreen
import com.example.url_detection.ui.screens.ProfileScreen
import com.example.url_detection.ui.screens.QrScannerScreen
import com.example.url_detection.ui.screens.ResultScreen
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    private lateinit var urlPredictor: UrlPredictor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        urlPredictor = UrlPredictor.create(this)
        val safeBrowsing = SafeBrowsingClient(applicationContext)
        val checks = RestDatabase.get(this).checks()
        val settings = AppSettings(applicationContext)

        setContent {
            // Wait for the saved theme so the first frame is not the wrong one, and for the
            // online switch so a link is never sent to Google after the user turned it off.
            val mode by settings.mode.collectAsState(initial = null)
            val online by settings.onlineCheck.collectAsState(initial = null)
            val themeMode = mode ?: return@setContent
            val onlineCheck = online ?: return@setContent
            RestTheme(themeMode) {
                RestApp(urlPredictor, safeBrowsing, checks, settings, themeMode, onlineCheck)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        urlPredictor.close()
    }
}

private object Routes {
    const val HOME = "home"
    const val SCAN = "scan"
    const val HISTORY = "history"
    const val PROFILE = "profile"
    const val ANALYZING = "analyzing?url={url}&source={source}&replace={replace}"
    const val RESULT = "result/{id}"

    /** [replace] is the id of an existing check to overwrite on Rescan, or -1 for a new check. */
    fun analyzing(url: String, source: CheckSource, replace: Long = -1) =
        "analyzing?url=${Uri.encode(url)}&source=${source.name}&replace=$replace"

    fun result(id: Long) = "result/$id"
}

@Composable
private fun RestApp(
    predictor: UrlPredictor,
    safeBrowsing: SafeBrowsingClient,
    checks: CheckDao,
    settings: AppSettings,
    themeMode: ThemeMode,
    onlineCheck: Boolean,
) {
    val nav = rememberNavController()
    val scope = rememberCoroutineScope()
    val route = nav.currentBackStackEntryAsState().value?.destination?.route
    val c = Rest.colors

    // Light status and navigation bar icons on dark backgrounds, including the camera.
    SystemBarIcons(lightIcons = c.isDark || route == Routes.SCAN)

    val reduceMotion = Rest.reduceMotion
    val enter: EnterTransition = if (reduceMotion) EnterTransition.None else fadeIn(tween(180))
    val exit: ExitTransition = if (reduceMotion) ExitTransition.None else fadeOut(tween(180))

    NavHost(
        navController = nav,
        startDestination = Routes.HOME,
        modifier = Modifier.fillMaxSize().background(c.bg),
        enterTransition = { enter },
        exitTransition = { exit },
        popEnterTransition = { enter },
        popExitTransition = { exit },
    ) {
        composable(Routes.HOME) {
            val recent by remember { checks.recent(2) }.collectAsState(initial = emptyList())
            HomeScreen(
                recent = recent,
                onCheck = { url -> nav.navigate(Routes.analyzing(url, CheckSource.Link)) },
                onScan = { nav.navigate(Routes.SCAN) },
                onOpenCheck = { id -> nav.navigate(Routes.result(id)) },
                onSeeAll = { nav.navigate(Routes.HISTORY) },
                onProfile = { nav.navigate(Routes.PROFILE) },
            )
        }

        composable(Routes.SCAN) {
            QrScannerScreen(
                onClose = { nav.popBackStack() },
                onQrCodeScanned = { url ->
                    nav.navigate(Routes.analyzing(url, CheckSource.Qr)) {
                        popUpTo(Routes.SCAN) { inclusive = true }
                    }
                },
            )
        }

        composable(
            Routes.ANALYZING,
            arguments = listOf(
                navArgument("url") { type = NavType.StringType; defaultValue = "" },
                navArgument("source") { type = NavType.StringType; defaultValue = CheckSource.Link.name },
                navArgument("replace") { type = NavType.LongType; defaultValue = -1L },
            ),
        ) { entry ->
            val args = requireNotNull(entry.arguments)
            val url = args.getString("url").orEmpty()
            val source = CheckSource.entries.firstOrNull { it.name == args.getString("source") } ?: CheckSource.Link
            val replace = args.getLong("replace")
            AnalyzingScreen(
                url = url,
                onlineCheck = onlineCheck,
                runModel = { withContext(Dispatchers.Default) { predictor.maliciousProbability(it) } },
                checkOnline = { safeBrowsing.check(it) },
                onDone = { p, online ->
                    val check = CheckEntity(
                        url = url,
                        source = source,
                        maliciousProbability = p,
                        checkedAt = System.currentTimeMillis(),
                        safeBrowsing = online.status,
                        threatTypes = encodeThreats(online.threats),
                    )
                    val id = if (replace > 0) {
                        checks.update(check.copy(id = replace))
                        replace
                    } else {
                        checks.insert(check)
                    }
                    nav.navigate(Routes.result(id)) {
                        popUpTo(Routes.ANALYZING) { inclusive = true }
                    }
                },
                onCancel = { nav.popBackStack(Routes.HOME, inclusive = false) },
            )
        }

        composable(Routes.RESULT, arguments = listOf(navArgument("id") { type = NavType.LongType })) { entry ->
            val id = requireNotNull(entry.arguments).getLong("id")
            val check by remember(id) { checks.byId(id) }.collectAsState(initial = null)
            val current = check
            if (current == null) {
                Box(Modifier.fillMaxSize().background(c.bg))
            } else {
                ResultScreen(
                    check = current,
                    onBack = { nav.popBackStack() },
                    onGoBack = { nav.popBackStack() },
                    onRescan = {
                        // Re-runs the model and Safe Browsing on the same link through Analyzing and updates this check.
                        nav.navigate(Routes.analyzing(current.url, current.source, current.id)) {
                            popUpTo(Routes.RESULT) { inclusive = true }
                        }
                    },
                )
            }
        }

        composable(Routes.HISTORY) {
            val all by remember { checks.all() }.collectAsState(initial = emptyList())
            HistoryScreen(
                checks = all,
                onBack = { nav.popBackStack() },
                onOpenCheck = { id -> nav.navigate(Routes.result(id)) },
            )
        }

        composable(Routes.PROFILE) {
            val all by remember { checks.all() }.collectAsState(initial = emptyList())
            ProfileScreen(
                themeMode = themeMode,
                onlineCheck = onlineCheck,
                historyCount = all.size,
                onThemeChange = { mode -> scope.launch { settings.setMode(mode) } },
                onOnlineCheckChange = { on -> scope.launch { settings.setOnlineCheck(on) } },
                onClearHistory = { checks.clear() },
                onBack = { nav.popBackStack() },
            )
        }
    }
}

@Composable
private fun SystemBarIcons(lightIcons: Boolean) {
    val activity = LocalContext.current as ComponentActivity
    DisposableEffect(lightIcons) {
        activity.enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT) { lightIcons },
            navigationBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT) { lightIcons },
        )
        onDispose { }
    }
}
