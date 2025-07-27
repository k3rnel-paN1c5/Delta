import 'package:flutter/material.dart';

Color getViridisColor(double value) {
  final List<Color> viridisColors = [
    const Color(0xFF440154), const Color(0xFF482173), const Color(0xFF433E85),
    const Color(0xFF38598C), const Color(0xFF2D708E), const Color(0xFF25858E),
    const Color(0xFF1E9B8A), const Color(0xFF2BB07F), const Color(0xFF51C56A),
    const Color(0xFF85D54A), const Color(0xFFC2E02A), const Color(0xFFFDE725),
  ];

  // Clamp the value to the valid range [0, 1]
  final t = value.clamp(0.0, 1.0);

  if (t == 1.0) {
    return viridisColors.last;
  }

  // Calculate the position in the color list
  final double rawIndex = t * (viridisColors.length - 1);
  final int index1 = rawIndex.floor();
  final double t2 = rawIndex - index1;

  // Linearly interpolate between the two closest colors
  return Color.lerp(viridisColors[index1], viridisColors[index1 + 1], t2)!;
}