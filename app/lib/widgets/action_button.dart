import 'package:flutter/material.dart';

/// A reusable, styled action button for the application.
///
/// It can display a loading indicator and can be disabled via the [onPressed] callback.
class ActionButton extends StatelessWidget {
  /// The text label to display on the button.
  final String label;

  /// The text label to display when the button is in a loading state.
  final String? loadingLabel;

  /// The icon to display on the button.
  final IconData icon;

  /// The callback that is called when the button is tapped. If null, the button is disabled.
  final VoidCallback? onPressed;

  /// The background color of the button.
  final Color backgroundColor;

  /// If true, a [CircularProgressIndicator] is shown instead of the icon.
  final bool isLoading;

  const ActionButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
    this.backgroundColor = Colors.blue,
    this.isLoading = false,
    this.loadingLabel,
  });

  @override
  Widget build(BuildContext context) {
    // The button is functionally disabled if isLoading is true or onPressed is null.
    final bool isDisabled = isLoading || onPressed == null;

    return ElevatedButton.icon(
      onPressed: isDisabled ? null : onPressed,
      icon: isLoading
          ? const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(
                color: Colors.white,
                strokeWidth: 2,
              ),
            )
          : Icon(icon),
      label: Text(
        isLoading ? (loadingLabel ?? label) : label,
      ),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 15),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        textStyle: const TextStyle(fontSize: 18),
        backgroundColor: backgroundColor,
        // Use a less opaque color when the button is disabled.
        disabledBackgroundColor: backgroundColor.withAlpha(125),
        foregroundColor: Colors.white,
      ),
    );
  }
}