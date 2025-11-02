import 'package:tflite_flutter/tflite_flutter.dart';

/// Lightweight test harness to load the TFLite interpreter and immediately close it.
///
/// This file is intentionally isolated so we can enable and debug TensorFlow Lite
/// loading without changing app startup behavior. Call `TfliteTest.loadAndClose()`
/// from a debug-only button or direct call when you want to test the native bindings.
class TfliteTest {
  /// Attempts to load the model from assets and close the interpreter.
  /// Returns a short message describing success or the caught error.
  static Future<String> loadAndClose({
    String assetPath = 'models/drone-detection-yolov11_float16.tflite',
  }) async {
    try {
      final interpreter = await Interpreter.fromAsset(
        assetPath,
        options: InterpreterOptions()..threads = 2,
      );
      final address = interpreter.address;
      interpreter.close();
      return 'Interpreter loaded and closed successfully (address=$address)';
    } catch (e, st) {
      return 'Interpreter load failed: $e\n$st';
    }
  }
}
