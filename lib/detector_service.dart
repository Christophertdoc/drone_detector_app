import 'dart:typed_data';

import 'package:camera/camera.dart';
// Note: we avoid using the `image` package here for preprocessing to keep
// conversion and resizing explicit and robust across package API changes.
import 'package:tflite_flutter/tflite_flutter.dart';

import 'models.dart';

/// DroneDetector with optional TensorFlow Lite support.
///
/// We load TFLite only when `enableTflite()` is called so the app can run
/// with camera-only functionality while we incrementally enable ML.
class DroneDetector {
  static Interpreter? _interpreter;
  static bool _tfliteEnabled = false;

  // Model / tensor shapes (adjust if your model differs)
  static const int inputSize = 640;
  static const int numChannels = 3;
  static const int batchSize = 1;
  static const int outputElements = 8400; // model-specific
  static const int outputBoxes = 5; // x,y,w,h,confidence

  /// Initialize non-ML parts (camera) — keeps lightweight.
  static Future<void> initialize() async {
    print('DroneDetector initialized (non-ML)');
  }

  /// Enable and load TensorFlow Lite interpreter. Returns a short status
  /// message describing success or the error encountered.
  static Future<String> enableTflite({String modelAsset = 'models/drone-detection-yolov11_float16.tflite'}) async {
    if (_interpreter != null) return 'Interpreter already loaded';
    try {
      _interpreter = await Interpreter.fromAsset(modelAsset, options: InterpreterOptions()..threads = 2);
      _tfliteEnabled = true;
      return 'Interpreter loaded successfully';
    } catch (e, st) {
      return 'Failed to load interpreter: $e\n$st';
    }
  }

  /// Run a single inference on the provided [image] and return a short
  /// raw-tensor dump string. This is a debug helper to show the interpreter
  /// output quickly without implementing the full detection pipeline.
  static Future<String> testInference(CameraImage image) async {
    if (!_tfliteEnabled || _interpreter == null) return 'TFLite not enabled';

    try {
      // Preprocess image -> Float32List
      final input = _preprocessToFloat32(image);

      // Prepare output buffer as a flat list then reshape via extension
      final outLen = batchSize * outputBoxes * outputElements;
      var output = List.filled(outLen, 0.0);

      // Run inference. The interpreter accepts nested lists or typed buffers.
      _interpreter!.run(input.toList().reshape([batchSize, numChannels, inputSize, inputSize]),
          output.reshape([batchSize, outputBoxes, outputElements]));

      // Provide a short dump: length + first 100 values
      final flat = output.cast<double>();
      final firstN = flat.length < 100 ? flat : flat.sublist(0, 100);
      return 'output_len=${flat.length}; sample=${firstN.join(', ')}';
    } catch (e, st) {
      return 'Inference failed: $e\n$st';
    }
  }

  /// Detect drones on a camera image. If TFLite is not enabled, returns
  /// an empty list so the app can continue to function without ML.
  static Future<List<Detection>?> detectDrones(CameraImage image) async {
    if (!_tfliteEnabled || _interpreter == null) return [];

    // Full ML pipeline not implemented yet — placeholder to add inference later.
    return [];
  }

  /// Convert CameraImage (YUV420) to a normalized Float32List matching
  /// model input. Returns a Float32List in (flattened) [R,G,B,...] order.
  static Float32List _preprocessToFloat32(CameraImage image) {
    // We'll produce a channels-last flattened buffer [R,G,B,R,G,B,...]
    // mapped to the model's expected memory layout via reshape later.
    final width = image.width;
    final height = image.height;

    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;

    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;

    final buffer = Float32List(batchSize * numChannels * inputSize * inputSize);

    for (var y = 0; y < inputSize; y++) {
      final srcY = (y * height / inputSize).floor().clamp(0, height - 1);
      for (var x = 0; x < inputSize; x++) {
        final srcX = (x * width / inputSize).floor().clamp(0, width - 1);

        final int yIndex = srcY * width + srcX;
        final yp = yPlane[yIndex];

        final uvIndex = uvPixelStride * (srcX / 2).floor() + uvRowStride * (srcY / 2).floor();
        final up = uPlane[uvIndex];
        final vp = vPlane[uvIndex];

        // Convert YUV to RGB
        var r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
        var g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).round().clamp(0, 255);
        var b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);

        final base = (y * inputSize + x) * 3;
        buffer[base] = r / 255.0;
        buffer[base + 1] = g / 255.0;
        buffer[base + 2] = b / 255.0;
      }
    }

    return buffer;
  }

  /// Dispose interpreter when done.
  static void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _tfliteEnabled = false;
  }
}
