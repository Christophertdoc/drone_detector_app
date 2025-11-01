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

      // Inspect the interpreter's expected input and output shapes so we can
      // feed tensors in the correct memory layout (NHWC vs NCHW etc.). This
      // prevents channel/shape mismatch errors like the conv.cc failure.
      List<int> inShape;
      try {
        final inTensor = _interpreter!.getInputTensor(0);
        inShape = inTensor.shape;
      } catch (e) {
        // Fallback: common NHWC layout [1, H, W, C]
        inShape = [batchSize, inputSize, inputSize, numChannels];
      }

      // Build input nested list matching the interpreter's input shape.
      Object inputForInterpreter;
      if (inShape.length == 4 && inShape[1] == numChannels) {
        // Detected something like [1, C, H, W] (NCHW)
        inputForInterpreter = List.generate(batchSize, (_) {
          return List.generate(numChannels, (c) {
            return List.generate(inputSize, (y) {
              return List.generate(inputSize, (x) {
                final base = (y * inputSize + x) * 3;
                switch (c) {
                  case 0:
                    return input[base];
                  case 1:
                    return input[base + 1];
                  default:
                    return input[base + 2];
                }
              });
            });
          });
        });
      } else {
        // Default to NHWC [1, H, W, C]
        inputForInterpreter = List.generate(batchSize, (_) {
          return List.generate(inputSize, (y) {
            return List.generate(inputSize, (x) {
              final base = (y * inputSize + x) * 3;
              return [input[base], input[base + 1], input[base + 2]];
            });
          });
        });
      }

      // Prepare an output buffer that matches the interpreter's output shape.
      List<int> outShape;
      try {
        final outTensor = _interpreter!.getOutputTensor(0);
        outShape = outTensor.shape;
      } catch (e) {
        // Fallback conservative shape used previously: [1, outputBoxes, outputElements]
        outShape = [batchSize, outputBoxes, outputElements];
      }

      Object outputForInterpreter = _makeNestedListFromShape(outShape);

      // Run inference.
      _interpreter!.run(inputForInterpreter, outputForInterpreter);

      // Flatten the nested output into a simple List<double> for a short dump.
      final flat = <double>[];
      _flattenNested(outputForInterpreter, flat);
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
    final buffer = Float32List(batchSize * numChannels * inputSize * inputSize);

    // Detect input format: some platforms (iOS simulators / certain devices)
    // may deliver BGRA8888 instead of YUV420. Handle both safely.
    final formatGroup = image.format.group;

    if (formatGroup == ImageFormatGroup.bgra8888 || (image.planes.length == 1 && image.planes[0].bytesPerPixel == 4)) {
      // BGRA path: plane[0].bytes contains BGRA bytes per pixel
      final plane = image.planes[0].bytes;
      final rowStride = image.planes[0].bytesPerRow;

      for (var y = 0; y < inputSize; y++) {
        final srcY = (y * height / inputSize).floor().clamp(0, height - 1);
        for (var x = 0; x < inputSize; x++) {
          final srcX = (x * width / inputSize).floor().clamp(0, width - 1);
          final srcIndex = srcY * rowStride + srcX * 4;

          int b = 0, g = 0, r = 0;
          if (srcIndex + 2 < plane.length) {
            b = plane[srcIndex];
            g = plane[srcIndex + 1];
            r = plane[srcIndex + 2];
          }

          final base = (y * inputSize + x) * 3;
          buffer[base] = r / 255.0;
          buffer[base + 1] = g / 255.0;
          buffer[base + 2] = b / 255.0;
        }
      }
    } else {
      // YUV420 path (default for many Android devices). Safely handle
      // cases where U/V plane strides or lengths may be unexpected.
      final yPlane = image.planes[0].bytes;
      final uPlane = image.planes.length > 1 ? image.planes[1].bytes : Uint8List(0);
      final vPlane = image.planes.length > 2 ? image.planes[2].bytes : Uint8List(0);
      final uvRowStride = image.planes.length > 1 ? image.planes[1].bytesPerRow : 0;
      final uvPixelStride = image.planes.length > 1 && image.planes[1].bytesPerPixel != null ? image.planes[1].bytesPerPixel! : 1;

      for (var y = 0; y < inputSize; y++) {
        final srcY = (y * height / inputSize).floor().clamp(0, height - 1);
        for (var x = 0; x < inputSize; x++) {
          final srcX = (x * width / inputSize).floor().clamp(0, width - 1);

          final int yIndex = srcY * width + srcX;
          final yp = (yIndex < yPlane.length) ? yPlane[yIndex] : 0;

          final int uvX = (srcX / 2).floor();
          final int uvY = (srcY / 2).floor();
          final int uvIndex = uvX * uvPixelStride + uvY * uvRowStride;

          final up = (uvIndex < uPlane.length) ? uPlane[uvIndex] : 128;
          final vp = (uvIndex < vPlane.length) ? vPlane[uvIndex] : 128;

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
    }

    return buffer;
  }

  // Helper: build a nested List filled with 0.0 that matches the provided shape.
  // Example: shape=[1,3,640,640] -> returns List of depth 4 with doubles.
  static Object _makeNestedListFromShape(List<int> shape) {
    Object build(int dim) {
      if (dim == shape.length - 1) {
        return List.filled(shape[dim], 0.0);
      }
      return List.generate(shape[dim], (_) => build(dim + 1));
    }

    return build(0);
  }

  // Helper: flatten arbitrarily nested Lists of doubles into [out].
  static void _flattenNested(Object node, List<double> out) {
    if (node is double) {
      out.add(node);
    } else if (node is List) {
      for (var e in node) {
        _flattenNested(e, out);
      }
    } else if (node is num) {
      out.add(node.toDouble());
    } else {
      // unknown node type; ignore
    }
  }

  /// Dispose interpreter when done.
  static void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _tfliteEnabled = false;
  }
}
