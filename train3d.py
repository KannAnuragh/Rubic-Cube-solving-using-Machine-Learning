import raylibpy as pr
import numpy as np
import configs
from rubik import Rubik
from cv_solver import RubikCVSolver, integrate_cv_solver_with_rubik

# Initialize window
pr.init_window(configs.window_w, configs.window_h, "Rubik's Cube CV Solver")
rubik_cube = Rubik()
cv_solver = RubikCVSolver()
rotation_queue = []

# Solver state
solver_mode = False
auto_solve = False
solution_ready = False
capture_completed = False

pr.set_target_fps(configs.fps)

# Initialize camera (try to initialize, but don't fail if no camera)
try:
    cv_solver.initialize_camera()
    camera_available = True
    print("Camera initialized successfully")
except Exception as e:
    camera_available = False
    print(f"Camera not available: {e}")

# Initial loading screen
pr.begin_drawing()
pr.clear_background(pr.RAYWHITE)
pr.draw_text(b"Rubik's Cube CV Solver", 10, 10, 24, pr.DARKGRAY)
pr.draw_text(b"Initializing...", 10, 50, 16, pr.DARKGRAY)
pr.end_drawing()
pr.wait_time(0.1)

while not pr.window_should_close():
    # Check if shift is held down for counter-clockwise moves
    shift_held = pr.is_key_down(pr.KEY_LEFT_SHIFT) or pr.is_key_down(pr.KEY_RIGHT_SHIFT)
    
    # Manual rotation controls (original functionality)
    if pr.is_key_pressed(pr.KEY_X) and camera_available and capture_completed:
        print("Starting recapture mode...")
        try:
            if cv_solver.recapture_faces():
                print("Recapture completed successfully!")
                rubik_cube.update_colors(cv_solver.cube_state)
            else:
                print("Recapture cancelled or failed")
        except Exception as e:
            print(f"Error during recapture: {e}")

    elif pr.is_key_pressed(pr.KEY_F):
        axis = np.array([0, 0, 1])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)

    elif pr.is_key_pressed(pr.KEY_U):
        axis = np.array([0, 1, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)
    elif pr.is_key_pressed(pr.KEY_L):
        axis = np.array([1, 0, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    elif pr.is_key_pressed(pr.KEY_B):
        axis = np.array([0, 0, 1])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    elif pr.is_key_pressed(pr.KEY_D):
        axis = np.array([0, 1, 0])
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    
    # Computer Vision Solver Controls
    elif pr.is_key_pressed(pr.KEY_C) and camera_available:
        # Initial capture or recapture cube state with camera
        print("Starting cube capture...")
        solver_mode = True
        try:
            if cv_solver.capture_cube_state():
                print("Cube state captured successfully!")
                rubik_cube.update_colors(cv_solver.cube_state)
                capture_completed = True
            else:
                print("Failed to capture cube state")
                capture_completed = False

        except Exception as e:
            print(f"Error capturing cube: {e}")
            solver_mode = False
            capture_completed = False
    
    elif pr.is_key_pressed(pr.KEY_X) and camera_available and capture_completed:
        # Recapture specific faces
        print("Starting recapture mode...")
        try:
            if cv_solver.recapture_faces():
                print("Recapture completed successfully!")
                rubik_cube.update_colors(cv_solver.cube_state)
            else:
                print("Recapture cancelled or failed")

        except Exception as e:
            print(f"Error during recapture: {e}")
    
    elif pr.is_key_pressed(pr.KEY_S) and solver_mode and capture_completed:
        # Generate solution for captured cube
        try:
            if cv_solver.solve_cube():
                solution_ready = True
                print("Solution generated! Press SPACE to step through moves or A for auto-solve.")
            else:
                print("Failed to generate solution")
        except Exception as e:
            print(f"Error generating solution: {e}")
    
    elif pr.is_key_pressed(pr.KEY_SPACE) and solution_ready:
        # Next step in solution
        if not rubik_cube.is_rotating:  # Only advance if not currently rotating
            move_data = integrate_cv_solver_with_rubik(rubik_cube, cv_solver)
            if move_data:
                axis, level, clockwise = move_data
                rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, clockwise)
                
                # Advance to next step
                if not cv_solver.next_step():
                    print("Solution complete! Cube should be solved.")
                    solution_ready = False
                    solver_mode = False
            else:
                print("No more moves or invalid move")
                solution_ready = False
    
    elif pr.is_key_pressed(pr.KEY_A) and solution_ready:
        # Toggle auto-solve mode
        auto_solve = not auto_solve
        print(f"Auto-solve mode: {'ON' if auto_solve else 'OFF'}")
    
    elif pr.is_key_pressed(pr.KEY_ESCAPE):
        # Reset solution
        cv_solver.reset_solution()
        solution_ready = False
        solver_mode = False
        auto_solve = False
        capture_completed = False
        print("Solution reset")
    
    # Auto-solve functionality
    if auto_solve and solution_ready and not rubik_cube.is_rotating:
        move_data = integrate_cv_solver_with_rubik(rubik_cube, cv_solver)
        if move_data:
            axis, level, clockwise = move_data
            rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, clockwise)
            
            if not cv_solver.next_step():
                print("Auto-solve complete!")
                auto_solve = False
                solution_ready = False
                solver_mode = False
        else:
            auto_solve = False
            solution_ready = False

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
    
    # Draw UI
    y_offset = 10
    
    # Manual controls
    pr.draw_text(b"Manual Controls:", 10, y_offset, 16, pr.DARKGRAY)
    y_offset += 25
    pr.draw_text(b"R-Right, L-Left, U-Up, D-Down, F-Front, B-Back", 10, y_offset, 12, pr.DARKGRAY)
    y_offset += 20
    pr.draw_text(b"Hold SHIFT for counter-clockwise", 10, y_offset, 12, pr.DARKGRAY)
    y_offset += 30
    
    # CV Solver controls
    pr.draw_text(b"CV Solver:", 10, y_offset, 16, pr.DARKGRAY)
    y_offset += 25
    
    if camera_available:
        pr.draw_text(b"C-Capture cube | X-Recapture faces | S-Solve | SPACE-Next | A-Auto | ESC-Reset", 10, y_offset, 12, pr.DARKGRAY)
        y_offset += 20
        
        if capture_completed:
            pr.draw_text(b"Capture completed! Use X to recapture individual faces if needed", 10, y_offset, 11, pr.DARKGREEN)
        else:
            pr.draw_text(b"Press C to start capturing cube faces", 10, y_offset, 11, pr.GRAY)
    else:
        pr.draw_text(b"Camera not available", 10, y_offset, 12, pr.RED)
    y_offset += 25
    
    # Current status
    if rubik_cube.is_rotating:
        pr.draw_text(b"Rotating...", 10, y_offset, 16, pr.RED)
        y_offset += 25
    
    if shift_held:
        pr.draw_text(b"SHIFT: Counter-Clockwise Mode", 10, y_offset, 14, pr.BLUE)
        y_offset += 25
    
    if solver_mode:
        pr.draw_text(b"CV Solver Mode Active", 10, y_offset, 16, pr.GREEN)
        y_offset += 25
        
        if solution_ready:
            # Show current move and progress
            current_move = cv_solver.get_current_move()
            if current_move[0]:
                move_text = f"Next: {current_move[0]} - {current_move[1]}".encode()
                pr.draw_text(move_text, 10, y_offset, 14, pr.DARKBLUE)
                y_offset += 20
            
            current_step, total_steps = cv_solver.get_progress()
            progress_text = f"Progress: {current_step}/{total_steps}".encode()
            pr.draw_text(progress_text, 10, y_offset, 14, pr.DARKBLUE)
            y_offset += 20
            
            if auto_solve:
                pr.draw_text(b"AUTO-SOLVE ON", 10, y_offset, 14, pr.ORANGE)
                y_offset += 20
            
            if cv_solver.is_solved:
                pr.draw_text(b"SOLVED!", 10, y_offset, 20, pr.GREEN)
        elif capture_completed:
            pr.draw_text(b"Ready to solve! Press S to generate solution", 10, y_offset, 14, pr.BLUE)
    
    
    if capture_completed:
        y_offset += 10
        pr.draw_text(b"Capture Controls (when in capture mode):", 10, y_offset, 12, pr.DARKGRAY)
        y_offset += 15
        pr.draw_text(b"S-Capture face | R-Review | 1-6-Jump to face | C-Clear all | Q-Quit", 10, y_offset, 10, pr.DARKGRAY)
    
    pr.end_drawing()

# Cleanup
if camera_available:
    cv_solver.release_camera()
pr.close_window()