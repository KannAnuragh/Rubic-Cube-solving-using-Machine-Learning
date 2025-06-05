import pyray as pr
import configs
from rubik import Rubik

pr.init_window(configs.window_w, configs.window_h, "Rubik's Cube")
rubic_cube = Rubik()
piece = rubic_cube.generate_rubik(size = 1)
pr.set_target_fps(configs.fps)


while not pr.window_should_close():
    pr.update_camera(configs.camera, pr.CameraMode.CAMERA_ORBITAL)
    
    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)
    
    position = pr.Vector3(piece.center[0], piece.center[1], piece.center[2])

    pr.begin_mode_3d(configs.camera)
    pr.draw_model(piece.model, position, 2, piece.face_color)
    pr.draw_grid(20, 1.0)
    pr.end_mode_3d()
    
    pr.end_drawing()

pr.close_window()
