import raylibpy as pr
import configs
from rubik import Rubik

pr.init_window(configs.window_w, configs.window_h, "Rubik's Cube")
rubic_cube = Rubik()

rotation_queue = []

pr.set_target_fps(configs.fps)

pr.set_target_fps(configs.fps)
pr.begin_drawing()
pr.clear_background(pr.RAYWHITE)
pr.draw_text(b"Initializing...", 10, 10, 20, pr.DARKGRAY)
pr.end_drawing()
pr.wait_time(0.1)

while not pr.window_should_close():

    rotation_queue, _ = rubic_cube.handle_rotation(rotation_queue)
    pr.update_camera(configs.camera, pr.CameraMode.CAMERA_THIRD_PERSON)

    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)
    
    pr.begin_mode3d(configs.camera)
    pr.draw_grid(20, 1.0)

    for cube_group in rubic_cube.cubes:
        for cube_part in cube_group:
            pr.draw_model(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.WHITE)
            pr.draw_model_wires(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.DARKGRAY)


    pr.end_mode3d() 
    pr.end_drawing()

pr.close_window()