import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'dart:math' as math;
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
  await DroneDetector.initialize();
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

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final camera = cameras[0]; // Use the first available camera
    _controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      _startImageStream();
    });
  }

  void _startImageStream() {
    _controller.startImageStream((CameraImage image) {
      // Keep latest image for debug inference
      _latestImage = image;
    });
  }

  @override
  void dispose() {
    _controller.dispose();
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
          ? Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                FloatingActionButton(
                  heroTag: 'enable_tflite',
                  onPressed: () async {
                    final msg = await DroneDetector.enableTflite();
                    if (!mounted) return;
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text(msg)));
                  },
                  child: const Icon(Icons.bug_report),
                  tooltip: 'Enable tflite (debug)',
                ),
                const SizedBox(height: 8),
                FloatingActionButton(
                  heroTag: 'run_inference',
                  onPressed: () async {
                    if (_latestImage == null) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('No camera frame available yet'),
                        ),
                      );
                      return;
                    }

                    // Clear any existing detections before running new inference
                    setState(() {
                      _detections = [];
                    });

                    // Run single inference and update detections
                    final detections = await DroneDetector.detectDrones(
                      _latestImage!,
                    );
                    setState(() {
                      _detections = detections;
                    });

                    // Show debug dialog with shapes and log
                    if (!mounted) return;
                    final inputShape = DroneDetector.inputShape;
                    final outputShape = DroneDetector.outputShape;
                    final debugLog = DroneDetector.getDebugLog();

                    return;

                    // Show debug output in scrollable dialog
                    showDialog(
                      context: context,
                      builder: (context) => AlertDialog(
                        title: const Text('Inference Debug Output'),
                        content: SizedBox(
                          width: double.maxFinite,
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              if (inputShape != null)
                                Text('Input shape: $inputShape'),
                              if (outputShape != null)
                                Text('Output shape: $outputShape'),
                              const SizedBox(height: 8),
                              Text('Detections: ${_detections?.length ?? 0}'),
                              if (_detections?.isNotEmpty ?? false) ...[
                                const SizedBox(height: 8),
                                Text(
                                  'Confidence scores: ${_detections!.map((d) => d.confidence.toStringAsFixed(2)).join(", ")}',
                                ),
                              ],
                              const SizedBox(height: 8),
                              const Text('Full debug log:'),
                              Flexible(
                                child: SingleChildScrollView(
                                  child: SelectableText(debugLog),
                                ),
                              ),
                              const SizedBox(height: 16),
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  TextButton(
                                    onPressed: () {
                                      final data = ClipboardData(
                                        text: debugLog,
                                      );
                                      Clipboard.setData(data);
                                      ScaffoldMessenger.of(
                                        context,
                                      ).showSnackBar(
                                        const SnackBar(
                                          content: Text(
                                            'Debug log copied to clipboard',
                                          ),
                                        ),
                                      );
                                    },
                                    child: const Text('Copy Log'),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text('Close'),
                          ),
                        ],
                      ),
                    );
                  },
                  child: const Icon(Icons.play_arrow),
                  tooltip: 'Run single inference (debug)',
                ),
              ],
            )
          : null,
    );
  }
}
