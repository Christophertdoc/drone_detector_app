import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'detector_service.dart';
import 'models.dart';

class DetectionBoxPainter extends CustomPainter {
  final List<Detection> detections;

  DetectionBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.red;

    for (final detection in detections) {
      final rect = Rect.fromLTRB(
        detection.boundingBox.left * size.width,
        detection.boundingBox.top * size.height,
        detection.boundingBox.right * size.width,
        detection.boundingBox.bottom * size.height,
      );

      canvas.drawRect(rect, paint);

      // Draw confidence score
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${(detection.confidence * 100).toStringAsFixed(0)}%',
          style: const TextStyle(color: Colors.red, fontSize: 16),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      textPainter.paint(canvas, Offset(rect.left, rect.top - 20));
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
      // keep latest image for debug inference
      _latestImage = image;
      if (!isDetecting) {
        isDetecting = true;
        _detectDrones(image).then((_) {
          isDetecting = false;
        });
      }
    });
  }

  Future<void> _detectDrones(CameraImage image) async {
    try {
      final results = await DroneDetector.detectDrones(image);
      if (results != null) {
        setState(() {
          _detections = results;
        });
      }
    } catch (e) {
      print('Error during detection: $e');
    }
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
            return Stack(
              children: [
                CameraPreview(_controller),
                CustomPaint(
                  painter: DetectionBoxPainter(_detections ?? []),
                  child: Container(),
                ),
              ],
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
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(msg)),
                    );
                  },
                  child: const Icon(Icons.bug_report),
                  tooltip: 'Enable tflite (debug)',
                ),
                const SizedBox(height: 8),
                FloatingActionButton(
                  heroTag: 'run_inference',
                  onPressed: () async {
                    if (_latestImage == null) {
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('No camera frame available yet')));
                      return;
                    }
                    final msg = await DroneDetector.testInference(_latestImage!);
                    if (!mounted) return;
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(msg)),
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
