import 'package:camera/camera.dart';
import 'models.dart';

// Temporary stub implementation for testing camera
class DroneDetector {
  // Initialize method that does nothing for now
  static Future<void> initialize() async {
    print('DroneDetector initialized (stub implementation)');
  }

  // Return empty detections list for now
  static Future<List<Detection>?> detectDrones(CameraImage image) async {
    // Just return an empty list for now to test camera
    return [];
  }

  // No cleanup needed in stub implementation
  static void dispose() {}
}
