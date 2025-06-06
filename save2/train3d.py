import raylibpy as pr
import numpy as np
import configs
from rubik import Rubik

# Initialize window
pr.init_window(configs.window_w, configs.window_h, "Rubik's Cube")
rubik_cube = Rubik()
rotation_queue = []

pr.set_target_fps(configs.fps)

# Initial loading screen
pr.begin_drawing()
pr.clear_background(pr.RAYWHITE)
pr.draw_text(b"Initializing...", 10, 10, 20, pr.DARKGRAY)
pr.draw_text(b"Press R to rotate right face", 10, 40, 20, pr.DARKGRAY)
pr.draw_text(b"Press F to rotate front face", 10, 70, 20, pr.DARKGRAY)
pr.draw_text(b"Press U to rotate top face", 10, 100, 20, pr.DARKGRAY)
pr.end_drawing()
pr.wait_time(0.1)

while not pr.window_should_close():
    # Check if shift is held down
    shift_held = pr.is_key_down(pr.KEY_LEFT_SHIFT) or pr.is_key_down(pr.KEY_RIGHT_SHIFT)
    
    # Handle input
    if pr.is_key_pressed(pr.KEY_R):
        # Rotate right face (X-axis, level 2)
        axis = np.array([1, 0, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)
    elif pr.is_key_pressed(pr.KEY_F):
        # Rotate front face (Z-axis, level 2)
        axis = np.array([0, 0, 1])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)
    elif pr.is_key_pressed(pr.KEY_U):
        # Rotate top face (Y-axis, level 2)
        axis = np.array([0, 1, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)
    elif pr.is_key_pressed(pr.KEY_L):
        # Rotate left face (X-axis, level 0)
        axis = np.array([1, 0, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    elif pr.is_key_pressed(pr.KEY_B):
        # Rotate back face (Z-axis, level 0)
        axis = np.array([0, 0, 1])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    elif pr.is_key_pressed(pr.KEY_D):
        # Rotate bottom face (Y-axis, level 0)
        axis = np.array([0, 1, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)

    # Handle rotation animation
    rotation_queue, _ = rubik_cube.handle_rotation(rotation_queue)
    
    # Update camera
    pr.update_camera(configs.camera, pr.CAMERA_THIRD_PERSON)

    # Draw everything
    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)
    
    pr.begin_mode3d(configs.camera)
    pr.draw_grid(20, 1.0)

    # Draw all cube parts
    for cube_group in rubik_cube.cubes:
        for cube_part in cube_group:
            pr.draw_model(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.WHITE)
            pr.draw_model_wires(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.DARKGRAY)

    pr.end_mode3d()
    
    # Draw instructions
    pr.draw_text(b"Controls:", 10, 10, 20, pr.DARKGRAY)
    pr.draw_text(b"R-Right, L-Left, U-Up, D-Down, F-Front, B-Back", 10, 35, 16, pr.DARKGRAY)
    pr.draw_text(b"Hold SHIFT for counter-clockwise rotation", 10, 55, 16, pr.DARKGRAY)
    if rubik_cube.is_rotating:
        pr.draw_text(b"Rotating...", 10, 80, 20, pr.RED)
    
    # Show current modifier state
    if shift_held:
        pr.draw_text(b"SHIFT: Counter-Clockwise Mode", 10, 105, 16, pr.BLUE)
    
    pr.end_drawing()

pr.close_window()