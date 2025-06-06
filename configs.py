# This is the main configuration file used for visual settings and camera placement
# It defines how big the window is and where the 3D camera starts

import raylibpy as pr

# Window resolution and frame rate
window_w = 1500     # Width of the application window (affects how much of the scene is visible horizontally)
window_h = 900      # Height of the application window
fps = 60            # Target frames per second

# Camera setup (used to view the 3D Rubik's cube)
camera = pr.Camera3D(
    pr.Vector3(8.0, 8.0, 8.0),     # Camera position in 3D space
    pr.Vector3(0.0, 0.0, 0.0),     # What the camera is looking at (center of the cube)
    pr.Vector3(0.0, 1.0, 0.0),     # Which direction is "up"
    45.0,                          # Field of view in degrees
    pr.CAMERA_PERSPECTIVE         # Use 3D perspective camera mode
)

# Camera settings for OpenCV capture
cv_camera_index = 0               # Usually 0 is the default webcam. Try 1, 2, etc., if you have multiple cameras.
cv_capture_width = 1280          # Width of captured frames from webcam
cv_capture_height = 720          # Height of captured frames from webcam
