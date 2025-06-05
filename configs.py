import pyray as pr

window_w = 1200
window_h = 900
fps = 60
camera = pr.Camera3D()
camera.position = pr.Vector3(10.0, 10.0, 10.0)
camera.target = pr.Vector3(0.0, 0.0, 0.0)
camera.up = pr.Vector3(0.0, 1.0, 0.0)
camera.fovy = 45.00
camera.projection = pr.CAMERA_PERSPECTIVE

#camera = pr.Camera3D([18.0, 16.0, 18.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 0)