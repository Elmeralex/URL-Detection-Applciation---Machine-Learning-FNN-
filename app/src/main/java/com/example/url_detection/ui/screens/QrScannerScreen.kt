package com.example.url_detection.ui.screens

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.url_detection.data.looksLikeUrl
import com.example.url_detection.ui.components.ArcadeButton
import com.example.url_detection.ui.components.BobbingDuck
import com.example.url_detection.ui.components.RestIcons
import com.example.url_detection.ui.components.ScannerButton
import com.example.url_detection.ui.components.SecondaryButton
import com.example.url_detection.ui.theme.Fixed
import com.example.url_detection.ui.theme.Rest
import com.example.url_detection.ui.theme.RestType
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.RGBLuminanceSource
import com.google.zxing.common.HybridBinarizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors

private val qrHints = mapOf(
    DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE),
    DecodeHintType.TRY_HARDER to true,
)

/**
 * Full-screen camera with a themed bottom sheet. The camera layer, corner brackets
 * and scan line are the same in both themes. When a code is found it goes straight on.
 */
@Composable
fun QrScannerScreen(
    onClose: () -> Unit,
    onQrCodeScanned: (String) -> Unit,
) {
    val c = Rest.colors
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    val latestOnScanned by rememberUpdatedState(onQrCodeScanned)

    var hasPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    // Android stops showing the system prompt after two denials; then point to Settings.
    var denials by remember { mutableIntStateOf(0) }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        hasPermission = granted
        if (!granted) denials++
    }
    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    var camera by remember { mutableStateOf<Camera?>(null) }
    var torchOn by remember { mutableStateOf(false) }
    var found by remember { mutableStateOf(false) }

    var lastRejected by remember { mutableStateOf<String?>(null) }

    // Called for every decoded frame; only the first real link moves on.
    fun deliver(text: String) {
        if (found) return
        val t = text.trim()
        if (!looksLikeUrl(t)) {
            if (t != lastRejected) {
                lastRejected = t
                scope.launch { snackbar.showSnackbar("This QR code doesn't contain a link.") }
            }
            return
        }
        found = true
        latestOnScanned(t)
    }

    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            val text = withContext(Dispatchers.Default) { decodeQrFromImage(context, uri) }
            if (text != null) deliver(text) else snackbar.showSnackbar("No QR code found in that image.")
        }
    }

    Scaffold(
        containerColor = Fixed.camera,
        contentWindowInsets = WindowInsets(0),
        snackbarHost = { SnackbarHost(snackbar) },
    ) { inner ->
        Box(Modifier.fillMaxSize().padding(inner).background(Fixed.camera)) {
            if (hasPermission) {
                CameraPreview(
                    onCamera = { camera = it },
                    onCode = ::deliver,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            Column(Modifier.fillMaxSize()) {
                // Top bar over the camera.
                Row(
                    Modifier
                        .fillMaxWidth()
                        .windowInsetsPadding(WindowInsets.statusBars)
                        .padding(horizontal = 20.dp, vertical = 16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    ScannerButton(RestIcons.Close, "Close scanner", onClose, tint = Fixed.creamTile)
                    Text(
                        "Scan QR code",
                        style = RestType.screenTitle,
                        color = Fixed.creamTile,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.weight(1f).semantics { heading() },
                    )
                    val hasFlash = camera?.cameraInfo?.hasFlashUnit() == true
                    ScannerButton(
                        if (torchOn) RestIcons.FlashOff else RestIcons.FlashOn,
                        if (torchOn) "Turn flashlight off" else "Flashlight",
                        onClick = {
                            if (hasFlash) {
                                torchOn = !torchOn
                                camera?.cameraControl?.enableTorch(torchOn)
                            } else {
                                scope.launch { snackbar.showSnackbar("This camera has no flashlight.") }
                            }
                        },
                        tint = Fixed.mustard,
                    )
                }

                Column(
                    Modifier.weight(1f).fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                ) {
                    if (hasPermission) {
                        Viewfinder()
                        Spacer(Modifier.height(28.dp))
                        Text("Point at a QR code", style = RestType.screenTitle, color = Fixed.creamTile)
                        Spacer(Modifier.height(6.dp))
                        Text("I'll check the link before you open it.", style = RestType.bodyLarge, color = Fixed.creamTile.copy(alpha = 0.85f))
                    } else {
                        PermissionPrompt(
                            permanentlyDenied = denials >= 2,
                            onAllow = { permissionLauncher.launch(Manifest.permission.CAMERA) },
                            onOpenSettings = {
                                context.startActivity(
                                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null)),
                                )
                            },
                        )
                    }
                }

                // Themed bottom sheet with the duck standing on its edge.
                Box {
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .padding(top = 96.dp)
                            .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
                            .background(c.surface)
                            .border(3.dp, c.line, RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
                            .windowInsetsPadding(WindowInsets.navigationBars)
                            .padding(start = 20.dp, end = 20.dp, top = 24.dp, bottom = 20.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            BlinkingSquare()
                            Spacer(Modifier.width(12.dp))
                            Text(
                                if (hasPermission) "Looking for a QR code…" else "Camera is off",
                                style = RestType.section,
                                color = c.ink,
                            )
                        }
                        Spacer(Modifier.height(16.dp))
                        SecondaryButton(
                            "Choose from gallery",
                            onClick = {
                                galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                            },
                            icon = RestIcons.Gallery,
                            fill = c.surface2,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                    BobbingDuck(
                        Modifier
                            .align(Alignment.TopEnd)
                            .padding(end = 52.dp)
                            .width(96.dp)
                            .offset(y = 3.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun PermissionPrompt(permanentlyDenied: Boolean, onAllow: () -> Unit, onOpenSettings: () -> Unit) {
    Column(Modifier.padding(horizontal = 32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Camera access is off", style = RestType.screenTitle, color = Fixed.creamTile)
        Spacer(Modifier.height(8.dp))
        Text(
            "Allow the camera to scan QR codes, or choose a picture from your gallery.",
            style = RestType.bodyLarge,
            color = Fixed.creamTile.copy(alpha = 0.85f),
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(20.dp))
        if (permanentlyDenied) {
            ArcadeButton("Open settings", onOpenSettings)
        } else {
            ArcadeButton("Allow camera", onAllow)
        }
    }
}

/** 260 dp square: four 52 dp sun corners and a scan line stepping up and down. */
@Composable
private fun Viewfinder() {
    val reduceMotion = Rest.reduceMotion
    var step by remember { mutableIntStateOf(7) }
    LaunchedEffect(reduceMotion) {
        if (reduceMotion) { step = 7; return@LaunchedEffect }
        // 2 s per cycle in 14 steps: 7 down, 7 up.
        var i = 0
        while (true) {
            delay(2000L / 14)
            i = (i + 1) % 14
            step = if (i <= 7) i else 14 - i
        }
    }
    Canvas(Modifier.size(260.dp).clearAndSetSemantics { }) {
        val corner = 52.dp.toPx()
        val stroke = 8.dp.toPx()
        val w = size.width
        val h = size.height
        val sun = Fixed.mustard
        // top-left
        drawRect(sun, Offset(0f, 0f), Size(corner, stroke))
        drawRect(sun, Offset(0f, 0f), Size(stroke, corner))
        // top-right
        drawRect(sun, Offset(w - corner, 0f), Size(corner, stroke))
        drawRect(sun, Offset(w - stroke, 0f), Size(stroke, corner))
        // bottom-left
        drawRect(sun, Offset(0f, h - stroke), Size(corner, stroke))
        drawRect(sun, Offset(0f, h - corner), Size(stroke, corner))
        // bottom-right
        drawRect(sun, Offset(w - corner, h - stroke), Size(corner, stroke))
        drawRect(sun, Offset(w - stroke, h - corner), Size(stroke, corner))
        // scan line
        val lineH = 6.dp.toPx()
        val inset = 40.dp.toPx()
        val top = stroke + 12.dp.toPx()
        val travel = h - 2 * top - lineH
        drawRect(Fixed.scanLine, Offset(inset, top + travel * step / 7f), Size(w - 2 * inset, lineH))
    }
}

@Composable
private fun BlinkingSquare() {
    val reduceMotion = Rest.reduceMotion
    var on by remember { mutableStateOf(true) }
    LaunchedEffect(reduceMotion) {
        if (reduceMotion) return@LaunchedEffect
        while (true) {
            delay(500)
            on = !on
        }
    }
    Box(
        Modifier
            .size(16.dp)
            .background(if (on) Rest.colors.accent else Rest.colors.accent.copy(alpha = 0.25f))
            .border(2.dp, Fixed.plum),
    )
}

@Composable
private fun CameraPreview(onCamera: (Camera) -> Unit, onCode: (String) -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val latestOnCode by rememberUpdatedState(onCode)
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) {
        onDispose {
            runCatching { ProcessCameraProvider.getInstance(context).get().unbindAll() }
            analysisExecutor.shutdown()
        }
    }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            val providerFuture = ProcessCameraProvider.getInstance(ctx)
            providerFuture.addListener({
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, QrCodeAnalyzer(ContextCompat.getMainExecutor(ctx)) { text -> latestOnCode(text) }) }
                try {
                    provider.unbindAll()
                    onCamera(provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis))
                } catch (e: Exception) {
                    Log.e("QrScannerScreen", "Camera binding failed", e)
                }
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        },
    )
}

private class QrCodeAnalyzer(
    private val mainExecutor: Executor,
    private val onQrCodeScanned: (String) -> Unit,
) : ImageAnalysis.Analyzer {

    private val reader = MultiFormatReader().apply { setHints(qrHints) }

    override fun analyze(imageProxy: ImageProxy) {
        try {
            // The Y plane can have row padding, so use the row stride as the data width.
            val plane = imageProxy.planes[0]
            val buffer = plane.buffer
            val data = ByteArray(buffer.remaining()).also { buffer.get(it) }
            val source = PlanarYUVLuminanceSource(
                data, plane.rowStride, imageProxy.height, 0, 0, imageProxy.width, imageProxy.height, false,
            )
            val result = reader.decodeWithState(BinaryBitmap(HybridBinarizer(source)))
            mainExecutor.execute { onQrCodeScanned(result.text) }
        } catch (e: Exception) {
            // No QR code in this frame.
        } finally {
            reader.reset()
            imageProxy.close()
        }
    }
}

/** Decodes a QR code from a picked image, scaled down so large photos stay fast. */
private fun decodeQrFromImage(context: Context, uri: Uri): String? = try {
    val original: Bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val longest = maxOf(info.size.width, info.size.height)
            if (longest > 1600) {
                val s = 1600f / longest
                decoder.setTargetSize((info.size.width * s).toInt(), (info.size.height * s).toInt())
            }
        }
    } else {
        @Suppress("DEPRECATION")
        MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
    }
    val bmp = original.copy(Bitmap.Config.ARGB_8888, false)
    val pixels = IntArray(bmp.width * bmp.height)
    bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
    val source = RGBLuminanceSource(bmp.width, bmp.height, pixels)
    MultiFormatReader().apply { setHints(qrHints) }.decodeWithState(BinaryBitmap(HybridBinarizer(source))).text
} catch (e: Exception) {
    null
}
