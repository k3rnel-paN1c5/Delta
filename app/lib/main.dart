import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'models/app_state.dart';
import 'screens/home_page.dart';

/// The main entry point of the application.
void main() {
  runApp(
    // Using ChangeNotifierProvider to make the AppState available to the entire widget tree.
    ChangeNotifierProvider(
      create: (context) => AppState(),
      child: const DepthEstimationApp(),
    ),
  );
}

/// The root widget of the application.
class DepthEstimationApp extends StatelessWidget {
  const DepthEstimationApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Accessing the app's state to get the current theme mode.
    final appState = Provider.of<AppState>(context);

    return MaterialApp(
      title: 'Depth Estimation Demo',
      // Defines the light theme for the app.
      theme: ThemeData(
        brightness: Brightness.light,
        primarySwatch: Colors.blueGrey,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Inter',
        scaffoldBackgroundColor: Colors.grey[100],
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.blueGrey,
          foregroundColor: Colors.white,
        ),
        textTheme: const TextTheme(
          headlineSmall: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
        ),
      ),
      // Defines the dark theme for the app.
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.teal,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Inter',
        scaffoldBackgroundColor: const Color(0xFF121212),
        cardColor: Colors.grey[850],
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.grey[900],
        ),
        textTheme: const TextTheme(
          headlineSmall: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      // Sets the theme mode based on the app's state.
      themeMode: appState.themeMode,
      home: const DepthEstimationHomePage(),
    );
  }
}