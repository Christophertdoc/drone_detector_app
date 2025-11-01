# Drone Detector App - AI Assistant Instructions

This document provides essential knowledge for AI coding assistants working with the Drone Detector Flutter application.

## Project Overview
- Cross-platform Flutter application for drone detection
- Uses TensorFlow Lite for ML-based drone detection (`models/drone-detection-yolov11_float16.tflite`)
- Supports multiple platforms: iOS, Android, macOS, Linux, Windows, and web

## Key Project Structure
```
lib/            # Core Flutter application code
models/         # ML model files
test/          # Widget tests
```

## Development Environment
- Flutter SDK version: ^3.9.2
- Dart environment configuration in `pubspec.yaml`
- Uses Material Design (`uses-material-design: true`)

## Development Workflows

### Testing
- Widget tests are in `test/widget_test.dart`
- Tests use `flutter_test` package
- Run tests with: `flutter test`

### Build & Run
- Debug mode: `flutter run`
- Hot reload supported (press 'r' in terminal or use IDE button)
- Hot restart for state reset (when hot reload isn't sufficient)

### ML Model Integration
- TensorFlow Lite model: `models/drone-detection-yolov11_float16.tflite`
- Model format: YOLOv11 (float16 quantization)
- Note: TensorFlow Lite integration needs to be implemented in the app

## Code Conventions
1. State Management
   - Uses Flutter's built-in `setState` for simple state changes
   - Stateful widgets extend `StatefulWidget` with separate `State` classes

2. Widget Structure
   - Main app widget: `MyApp` (stateless)
   - Home page: `MyHomePage` (stateful)
   - Follow Material Design patterns

## Key Files
- `lib/main.dart`: Application entry point and core UI setup
- `pubspec.yaml`: Project configuration and dependencies
- `test/widget_test.dart`: Example widget test structure

## TODO
1. Implement TensorFlow Lite integration
2. Add camera input handling
3. Create drone detection UI components
4. Implement real-time detection pipeline