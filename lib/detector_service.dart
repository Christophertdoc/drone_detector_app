import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
// Note: we avoid using the `image` package here for preprocessing to keep
// conversion and resizing explicit and robust across package API changes.
import 'package:tflite_flutter/tflite_flutter.dart';

import 'models.dart';

// Top-level isolate-safe preprocess function. It receives a Map with
// camera frame metadata and plane bytes, and returns a Float32List
// normalized [R,G,B,...] buffer sized for `inputSize` x `inputSize`.
// This allows expensive pixel conversions to run off the UI isolate.
Future<Float32List> _preprocessToFloat32Isolate(
  Map<String, dynamic> req,
) async {
  final int width = req['width'] as int;
  final int height = req['height'] as int;
  final int inputSizeLocal = req['inputSize'] as int;
  final int numChannelsLocal = req['numChannels'] as int;
  final bool isBGRA = req['isBGRA'] as bool;

  final List planes = req['planes'] as List;

  final buffer = Float32List(
    1 * numChannelsLocal * inputSizeLocal * inputSizeLocal,
  );

  if (isBGRA) {
    final Uint8List planeBytes = planes[0]['bytes'] as Uint8List;
    final int rowStride = planes[0]['bytesPerRow'] as int;

    for (var y = 0; y < inputSizeLocal; y++) {
      final srcY = (y * height / inputSizeLocal).floor().clamp(0, height - 1);
      for (var x = 0; x < inputSizeLocal; x++) {
        final srcX = (x * width / inputSizeLocal).floor().clamp(0, width - 1);
        final srcIndex = srcY * rowStride + srcX * 4;

        int b = 0, g = 0, r = 0;
        if (srcIndex + 2 < planeBytes.length) {
          b = planeBytes[srcIndex];
          g = planeBytes[srcIndex + 1];
          r = planeBytes[srcIndex + 2];
        }

        final base = (y * inputSizeLocal + x) * 3;
        buffer[base] = r / 255.0;
        buffer[base + 1] = g / 255.0;
        buffer[base + 2] = b / 255.0;
      }
    }
  } else {
    final Uint8List yPlane = planes[0]['bytes'] as Uint8List;
    final Uint8List uPlane = planes.length > 1
        ? planes[1]['bytes'] as Uint8List
        : Uint8List(0);
    final Uint8List vPlane = planes.length > 2
        ? planes[2]['bytes'] as Uint8List
        : Uint8List(0);
    final int uvRowStride = planes.length > 1
        ? planes[1]['bytesPerRow'] as int
        : 0;
    final int uvPixelStride =
        planes.length > 1 && planes[1]['bytesPerPixel'] != null
        ? planes[1]['bytesPerPixel'] as int
        : 1;

    for (var y = 0; y < inputSizeLocal; y++) {
      final srcY = (y * height / inputSizeLocal).floor().clamp(0, height - 1);
      for (var x = 0; x < inputSizeLocal; x++) {
        final srcX = (x * width / inputSizeLocal).floor().clamp(0, width - 1);

        final int yIndex = srcY * width + srcX;
        final yp = (yIndex < yPlane.length) ? yPlane[yIndex] : 0;

        final int uvX = (srcX / 2).floor();
        final int uvY = (srcY / 2).floor();
        final int uvIndex = uvX * uvPixelStride + uvY * uvRowStride;

        final up = (uvIndex < uPlane.length) ? uPlane[uvIndex] : 128;
        final vp = (uvIndex < vPlane.length) ? vPlane[uvIndex] : 128;

        // Convert YUV to RGB
        var r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
        var g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
            .round()
            .clamp(0, 255);
        var b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);

        final base = (y * inputSizeLocal + x) * 3;
        buffer[base] = r / 255.0;
        buffer[base + 1] = g / 255.0;
        buffer[base + 2] = b / 255.0;
      }
    }
  }

  return buffer;
}

// (No top-level InferenceData anymore — we keep inference synchronous for now.)

/// DroneDetector with optional TensorFlow Lite support.
///
/// We load TFLite only when `enableTflite()` is called so the app can run
/// with camera-only functionality while we incrementally enable ML.
class DroneDetector {
  static Interpreter? _interpreter;

  /// Public: get model input shape, or null if not loaded
  static List<int>? get inputShape {
    try {
      return _interpreter?.getInputTensor(0).shape;
    } catch (_) {
      return null;
    }
  }

  /// Public: get model output shape, or null if not loaded
  static List<int>? get outputShape {
    try {
      return _interpreter?.getOutputTensor(0).shape;
    } catch (_) {
      return null;
    }
  }

  // Model / tensor shapes (adjust if your model differs)
  static const int inputSize = 640;
  static const int numChannels = 3;
  static const int batchSize = 1;
  static const int outputElements = 8400; // model-specific
  static const int outputBoxes = 5; // x,y,w,h,confidence

  /// Build input tensor in NHWC format (batch, height, width, channels)
  static Object _buildNHWCInput(Float32List input) {
    return List.generate(batchSize, (_) {
      return List.generate(inputSize, (y) {
        return List.generate(inputSize, (x) {
          final base = (y * inputSize + x) * 3;
          return [input[base], input[base + 1], input[base + 2]];
        });
      });
    });
  }

  /// Enable and load TensorFlow Lite interpreter. Returns a short status
  /// message describing success or the error encountered.
  static Future<String> enableTflite({
    String modelAsset = 'models/drone-detection-yolov11_float16.tflite',
  }) async {
    if (_interpreter != null) {
      return 'Interpreter already loaded';
    }
    try {
      _interpreter = await Interpreter.fromAsset(
        modelAsset,
        options: InterpreterOptions()..threads = 2,
      );

      return 'Interpreter loaded successfully';
    } catch (e, st) {
      final error = 'Failed to load interpreter: $e\n$st';
      debugPrint(error);
      return error;
    }
  }

  /// Run inference and return the raw output tensor in shape (1, 5, 8400)
  static Future<List<List<List<double>>>?> runInference(
    CameraImage image,
  ) async {
    try {
      if (_interpreter == null) {
        debugPrint('TFLite not enabled');
        return null;
      }

      debugPrint('Starting inference...');

      // Preprocess image in isolate
      final req = {
        'width': image.width,
        'height': image.height,
        'inputSize': inputSize,
        'numChannels': numChannels,
        'isBGRA':
            image.format.group == ImageFormatGroup.bgra8888 ||
            (image.planes.length == 1 && image.planes[0].bytesPerPixel == 4),
        'planes': image.planes
            .map(
              (p) => {
                'bytes': p.bytes,
                'bytesPerRow': p.bytesPerRow,
                'bytesPerPixel': p.bytesPerPixel,
              },
            )
            .toList(),
      };

      final input = await compute(_preprocessToFloat32Isolate, req);

      // For NHWC format (1, 640, 640, 3)
      final inputForInterpreter = _buildNHWCInput(input);

      // Prepare output tensor for shape (1, 5, 8400)
      final outputForInterpreter = List.generate(
        1,
        (_) => List.generate(5, (_) => List.filled(8400, 0.0)),
      );

      // Run inference
      _interpreter!.run(inputForInterpreter, outputForInterpreter);

      debugPrint('Inference complete');
      return outputForInterpreter;
    } catch (error) {
      debugPrint('Error during inference: $error');
      return null;
    }
  }

  /// Calculate Intersection over Union (IoU) between two bounding boxes
  static double _calculateIoU(Detection a, Detection b) {
    // Calculate intersection
    final intersectLeft = math.max(a.boundingBox.left, b.boundingBox.left);
    final intersectTop = math.max(a.boundingBox.top, b.boundingBox.top);
    final intersectRight = math.min(a.boundingBox.right, b.boundingBox.right);
    final intersectBottom = math.min(
      a.boundingBox.bottom,
      b.boundingBox.bottom,
    );

    if (intersectRight <= intersectLeft || intersectBottom <= intersectTop) {
      return 0.0;
    }

    final intersectionArea =
        (intersectRight - intersectLeft) * (intersectBottom - intersectTop);

    // Calculate union
    final box1Area =
        (a.boundingBox.right - a.boundingBox.left) *
        (a.boundingBox.bottom - a.boundingBox.top);
    final box2Area =
        (b.boundingBox.right - b.boundingBox.left) *
        (b.boundingBox.bottom - b.boundingBox.top);
    final unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
  }

  /// Apply Non-Maximum Suppression to filter overlapping detections
  static List<Detection> _applyNMS(
    List<Detection> detections,
    double iouThreshold,
  ) {
    if (detections.isEmpty) return [];

    final List<Detection> selected = [];
    // Detections should already be sorted by confidence (highest first)

    while (detections.isNotEmpty) {
      final current = detections.removeAt(
        0,
      ); // Take highest confidence detection
      selected.add(current);

      // Filter out detections that overlap too much with the current one
      detections.removeWhere((detection) {
        final iou = _calculateIoU(current, detection);
        return iou > iouThreshold;
      });
    }

    return selected;
  }

  /// Detect drones on a camera image. If TFLite is not enabled, returns
  /// an empty list so the app can continue to function without ML.
  static Future<List<Detection>> detectDrones(CameraImage image) async {
    if (_interpreter == null) return [];

    final output = await runInference(image);
    if (output == null || output.isEmpty) return [];

    final List<Detection> rawDetections = [];
    const confidenceThreshold = 0.50; // Adjustable confidence threshold
    const iouThreshold =
        0.3; // IoU threshold for NMS (lower = stricter filtering)
    const minBoxArea =
        0.005; // Minimum box area (0.5% of image) to filter tiny detections

    try {
      // Process output tensor (1, 5, 8400)
      final boxes = output[0]; // shape is [5, 8400]

      // For each grid cell
      for (var i = 0; i < 8400; i++) {
        final confidence = boxes[4][i];
        if (confidence > confidenceThreshold) {
          // Extract box coordinates (centerX, centerY, width, height)
          final centerX = boxes[0][i];
          final centerY = boxes[1][i];
          final width = boxes[2][i];
          final height = boxes[3][i];

          // Convert to corners (relative coordinates 0-1)
          final x1 = (centerX - width / 2).clamp(0.0, 1.0);
          final y1 = (centerY - height / 2).clamp(0.0, 1.0);
          final x2 = (centerX + width / 2).clamp(0.0, 1.0);
          final y2 = (centerY + height / 2).clamp(0.0, 1.0);

          // Calculate box area and filter out tiny detections
          final boxWidth = x2 - x1;
          final boxHeight = y2 - y1;
          final boxArea = boxWidth * boxHeight;

          if (boxArea < minBoxArea) {
            continue; // Skip tiny detections (likely noise)
          }

          rawDetections.add(
            Detection(
              boundingBox: BoundingBox(
                left: x1,
                top: y1,
                right: x2,
                bottom: y2,
              ),
              confidence: confidence,
            ),
          );
        }
      }

      // Sort by confidence (highest first) before NMS
      rawDetections.sort((a, b) => b.confidence.compareTo(a.confidence));

      // Apply Non-Maximum Suppression
      // NMS (Non-Maximum Suppression) removes duplicate/overlapping bounding
      // boxes of the same object by keeping only the highest-confidence detection
      // and discarding others that overlap with it by more than a threshold (iou)
      final filteredDetections = _applyNMS(
        List.from(rawDetections),
        iouThreshold,
      );

      return filteredDetections;
    } catch (e) {
      debugPrint('[ Error processing detections ]: $e');
      return [];
    }
  }
}
