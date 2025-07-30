import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:app/utils/color_maps.dart';

void main() {
  group('getViridisColor', () {
    test('returns the first color for value 0.0', () {
      final color = getViridisColor(0.0);
      expect(color, const Color(0xFF440154));
    });

    test('returns the last color for value 1.0', () {
      final color = getViridisColor(1.0);
      expect(color, const Color(0xFFFDE725));
    });

    test('returns the last color for values greater than 1.0', () {
      final color = getViridisColor(1.5);
      expect(color, const Color(0xFFFDE725));
    });

    test('returns the first color for values less than 0.0', () {
      final color = getViridisColor(-0.5);
      expect(color, const Color(0xFF440154));
    });

    test('correctly interpolates a color for value 0.5', () {
      // 0.5 should fall between index 5 and 6 of the 12-color list
      final color1 = const Color(0xFF25858E); // index 5
      final color2 = const Color(0xFF1E9B8A); // index 6
      
      // Manually calculate the expected interpolated color
      // The function calculates the index as t * (length - 1) = 0.5 * 11 = 5.5
      // So it will lerp between index 5 and 6 with a t-value of 0.5
      final expectedColor = Color.lerp(color1, color2, 0.5);

      final actualColor = getViridisColor(0.5);
      
      expect(actualColor, expectedColor);
    });
  });
}