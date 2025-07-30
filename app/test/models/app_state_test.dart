import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:app/models/app_state.dart';

// Mocks are needed for types from dart:io
class MockFile extends Mock implements File {}

void main() {
  group('AppState', () {
    late AppState appState;

    setUp(() {
      appState = AppState();
    });

    test('Initial values are correct', () {
      expect(appState.selectedImage, isNull);
      expect(appState.depthMapImageBytes, isNull);
      expect(appState.isProcessing, isFalse);
      expect(appState.isImageLoading, isFalse);
      expect(appState.inferenceTime, isEmpty);
      expect(appState.selectedColorMap, 'Grayscale');
      expect(appState.themeMode, ThemeMode.system);
      expect(appState.rawDepthMap, isNull);
    });

    test('setSelectedImage resets related states', () {
      // Setup listener to verify it's called
      bool listenerCalled = false;
      appState.addListener(() => listenerCalled = true);

      final mockImage = MockFile();
      appState.setSelectedImage(mockImage);

      expect(appState.selectedImage, mockImage);
      expect(appState.depthMapImageBytes, isNull);
      expect(appState.rawDepthMap, isNull);
      expect(appState.inferenceTime, '');
      expect(listenerCalled, isTrue);
    });

    test('setDepthMap updates depth map bytes', () {
      bool listenerCalled = false;
      appState.addListener(() => listenerCalled = true);

      final bytes = Uint8List.fromList([1, 2, 3]);
      appState.setDepthMap(bytes);

      expect(appState.depthMapImageBytes, bytes);
      expect(listenerCalled, isTrue);
    });

    test('setProcessing updates processing state', () {
      bool listenerCalled = false;
      appState.addListener(() => listenerCalled = true);

      appState.setProcessing(true);

      expect(appState.isProcessing, isTrue);
      expect(listenerCalled, isTrue);
    });

    test('toggleTheme switches between light and dark', () {
      bool listenerCalled = false;
      appState.addListener(() => listenerCalled = true);

      // Initial state is system, let's assume it resolves to light first
      // Toggle to dark
      appState.toggleTheme();
      expect(appState.themeMode, ThemeMode.dark);

      // Toggle back to light
      appState.toggleTheme();
      expect(appState.themeMode, ThemeMode.light);
      
      // Should be called twice
      expect(listenerCalled, isTrue);
    });
    
    test('setSelectedColorMap updates the color map', () {
      bool listenerCalled = false;
      appState.addListener(() => listenerCalled = true);

      appState.setSelectedColorMap('Viridis');

      expect(appState.selectedColorMap, 'Viridis');
      expect(listenerCalled, isTrue);
    });
  });
}