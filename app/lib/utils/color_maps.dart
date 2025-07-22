import 'package:flutter/material.dart';

Color getViridisColor(double value) {
  final List<Color> viridisColors = [
    const Color(0xFF440154),
    const Color(0xFF482173),
    const Color(0xFF433E85),
    const Color(0xFF38598C),
    const Color(0xFF2D708E),
    const Color(0xFF25858E),
    const Color(0xFF1E9B8A),
    const Color(0xFF2BB07F),
    const Color(0xFF51C56A),
    const Color(0xFF85D54A),
    const Color(0xFFC2E02A),
    const Color(0xFFFDE725),
  ];

  final index = (value * (viridisColors.length - 1)).round().clamp(0, viridisColors.length - 1);
  return viridisColors[index];
}