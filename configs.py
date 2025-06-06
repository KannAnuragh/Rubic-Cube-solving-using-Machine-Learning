# configs.py - Configuration settings
import raylibpy as pr

# Window settings
window_w = 1200
window_h = 800
fps = 60

# Camera settings for 3D view
camera = pr.Camera3D(
    pr.Vector3(8.0, 8.0, 8.0),    # position
    pr.Vector3(0.0, 0.0, 0.0),    # target
    pr.Vector3(0.0, 1.0, 0.0),    # up
    45.0,                          # fovy
    pr.CAMERA_PERSPECTIVE          # projection
)

# CV Settings
cv_camera_index = 0
cv_capture_width = 1280
cv_capture_height = 720