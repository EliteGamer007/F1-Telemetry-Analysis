F1-Telemetry-Analysis

A Python-based tool for analyzing Formula 1 telemetry data. This project serves two distinct purposes: visualizing real-world F1 timing data (live and historical) and performing strategy analysis for the video game F1 Manager 2024.

Project Overview

This tool aggregates data from the official F1 timing API and local game save files to provide a centralized dashboard for race strategy and performance analysis. It uses Machine Learning to model tyre degradation in real-world scenarios and database parsing to optimize strategies within the F1 Manager game engine.

Key Features

#1. Real-World Analysis (FastF1)

Dynamic Track Map: Visualizes driver positions and gaps on a 2D circuit map.
Telemetry Traces: Synchronized plotting of Speed, RPM, Gear, Throttle, and Brake data.
Historical Replay: Load and replay race sessions from 2018 to present.

Driver Comparison: Overlay telemetry from different drivers to identify time deltas and driving styles.

#2. Machine Learning Strategy Engine

Fuel-Corrected Pace: Algorithms to decouple fuel burn effects from raw lap times to isolate true tyre performance.
Degradation Modeling: Uses Linear Regression (scikit-learn) to calculate degradation slopes ($s/lap$) for specific tyre compounds based on practice data.
Predictive Simulation: Monte Carlo simulations to predict optimal pit windows and race outcomes.

#3. F1 Manager 2024 Support

Save File Analysis: Parses extracted SQLite save files from F1 Manager 2024.
Hidden Stat Extraction: Retrieves "under the hood" variables not shown in the game UI (e.g., precise tyre wear coefficients, AI decision thresholds).
Game Strategy Optimizer: Calculates optimal pit strategies based on the game's internal math logic rather than real-world physics. 
