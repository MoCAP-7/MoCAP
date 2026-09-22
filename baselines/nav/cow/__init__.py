"""CoW (CLIP on Wheels) ObjectNav baseline on the real YOR robot.

The upstream agent runs unmodified from a pinned external checkout. This package
only renders ZED frames into CoW's camera model, supplies the robot pose, and
executes CoW's discrete actions with YOR's navigation primitives.
"""
