import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';

/// A dropdown to select the color map for the depth map visualization.
class ColorMapDropdown extends StatelessWidget {
  final Function(String)? onColorMapChanged;
  const ColorMapDropdown({super.key, this.onColorMapChanged});

  @override
  Widget build(BuildContext context) {
    final appState = Provider.of<AppState>(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12.0),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(10.0),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: appState.selectedColorMap,
          isExpanded: true,
          onChanged: (String? newValue) {
            if (newValue != null) {
              appState.setSelectedColorMap(newValue);
              onColorMapChanged?.call(newValue);
            }
          },
          items: <String>['Grayscale', 'Viridis'].map<DropdownMenuItem<String>>(
            (String value) {
              return DropdownMenuItem<String>(value: value, child: Text(value));
            },
          ).toList(),
        ),
      ),
    );
  }
}
