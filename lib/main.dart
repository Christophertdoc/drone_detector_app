import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'dart:math' as math;
import 'dart:async';
import 'detector_service.dart';
import 'models.dart';

class DetectionBoxPainter extends CustomPainter {
  final List<Detection> detections;

  DetectionBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.redAccent;

    final fillPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.redAccent.withOpacity(0.2);

    // Since both the preview and model input are square,
    // we can directly map normalized coordinates (0-1) to the preview size
    for (final detection in detections) {
      final left = detection.boundingBox.left * size.width;
      final top = detection.boundingBox.top * size.height;
      final right = detection.boundingBox.right * size.width;
      final bottom = detection.boundingBox.bottom * size.height;

      final rect = Rect.fromLTRB(left, top, right, bottom);

      // Draw filled box
      canvas.drawRect(rect, fillPaint);
      // Draw box outline
      canvas.drawRect(rect, boxPaint);

      // Draw confidence label with background
      final textSpan = TextSpan(
        text: '${(detection.confidence * 100).toStringAsFixed(1)}%',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final labelTop = math.max(0.0, top - textPainter.height - 4);
      final labelRect = Rect.fromLTWH(
        left,
        labelTop,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      // Draw label background
      final labelBgPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = Colors.black.withOpacity(0.7);
      canvas.drawRect(labelRect, labelBgPaint);

      // Draw label text
      textPainter.paint(canvas, Offset(labelRect.left + 4, labelRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(DetectionBoxPainter oldDelegate) {
    return detections != oldDelegate.detections;
  }
}

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  await DroneDetector.enableTflite();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Drone Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Drone Detector'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool isDetecting = false;
  List<Detection>? _detections;
  CameraImage? _latestImage;
  Timer? _inferenceTimer;
  bool _isContinuousInference = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final camera = cameras[0]; // Use the first available camera
    _controller = CameraController(
      camera,
      ResolutionPreset.low, // Lower resolution = faster preprocessing
      enableAudio: false,
    );

    _initializeControllerFuture = _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      _startImageStream();
      _startContinuousInference(); // Start continuous inference at 1 FPS
    });
  }

  void _startImageStream() {
    _controller.startImageStream((CameraImage image) {
      // Keep latest image for continuous inference
      _latestImage = image;
    });
  }

  void _startContinuousInference() {
    debugPrint('Starting continuous inference...');
    _isContinuousInference = true;
    _runInferenceLoop();
  }

  Future<void> _runInferenceLoop() async {
    debugPrint('Inference loop started');
    while (_isContinuousInference && mounted) {
      if (_latestImage != null) {
        debugPrint('Running detection...');
        final detections = await DroneDetector.detectDrones(_latestImage!);
        debugPrint('Got ${detections.length} detections');
        if (mounted) {
          setState(() {
            _detections = detections;
          });
        }
      } else {
        debugPrint('No image available yet');
      }
      // Small delay to prevent tight loop, but inference runs as fast as possible
      await Future.delayed(const Duration(milliseconds: 100));
    }
    debugPrint('Inference loop stopped');
  }

  void _stopContinuousInference() {
    _isContinuousInference = false;
    _inferenceTimer?.cancel();
    _inferenceTimer = null;
    setState(() {
      _detections = [];
    });
  }

  @override
  void dispose() {
    _inferenceTimer?.cancel();
    _controller.dispose();
    DroneDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // Use a square preview to match the model's 640x640 input
            // This eliminates aspect ratio mapping issues
            return Center(
              child: AspectRatio(
                aspectRatio: 1.0, // Square preview
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_controller),
                    CustomPaint(
                      painter: DetectionBoxPainter(_detections ?? []),
                    ),
                  ],
                ),
              ),
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: kDebugMode
          ? FloatingActionButton.extended(
              heroTag: 'toggle_inference',
              onPressed: () {
                if (_isContinuousInference) {
                  _stopContinuousInference();
                } else {
                  _startContinuousInference();
                }
              },
              icon: Icon(
                _isContinuousInference ? Icons.stop : Icons.play_arrow,
              ),
              label: Text(_isContinuousInference ? 'Stop' : 'Start'),
              tooltip: _isContinuousInference
                  ? 'Stop continuous inference'
                  : 'Start continuous inference',
            )
          : null,
    );
  }
}
