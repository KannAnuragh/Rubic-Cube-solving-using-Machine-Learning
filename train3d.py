# These are like importing tools we need to build our program
import raylibpy as pr  # This helps us draw 3D graphics and windows
import numpy as np     # This helps us do math with numbers and arrays
import configs         # This has our settings like window size and colors
from rubik import Rubik  # This is our 3D Rubik's cube
from cv_solver import RubikCVSolver, integrate_cv_solver_with_rubik, map_move_to_game_input  # This helps solve the cube using a camera

# Create a window where we can see our cube (like opening a video game)
pr.init_window(configs.window_w, configs.window_h, "Rubik's Cube CV Solver")
rubik_cube = Rubik()  # Make a new Rubik's cube
cv_solver = RubikCVSolver()  # Make a new camera solver
rotation_queue = []  # This is like a line of moves waiting to happen

# These variables keep track of what mode we're in
solver_mode = False      # Are we using the camera to solve?
auto_solve = False       # Should the computer solve automatically?
solution_ready = False   # Do we have a solution ready to use?
capture_completed = False # Did we successfully take pictures of the cube?

# Set how fast our program runs (like setting the speed of a video game)
pr.set_target_fps(configs.fps)

# Try to turn on the camera (like asking permission to use your webcam)
try:
    cv_solver.initialize_camera()  # Turn on the camera
    camera_available = True        # Remember that camera works
    print("Camera initialized successfully")  # Tell us it worked
except Exception as e:  # If something goes wrong with the camera
    camera_available = False  # Remember that camera doesn't work
    print(f"Camera not available: {e}")  # Tell us what went wrong

# Show a loading screen while everything starts up
pr.begin_drawing()  # Start drawing on the screen
pr.clear_background(pr.RAYWHITE)  # Make the background white
pr.draw_text(b"Rubik's Cube CV Solver", 10, 10, 24, pr.DARKGRAY)  # Write the title
pr.draw_text(b"Initializing...", 10, 50, 16, pr.DARKGRAY)  # Write "starting up"
pr.end_drawing()  # Finish drawing
pr.wait_time(0.1)  # Wait a tiny bit

# This is the main loop - it runs over and over until we close the program
while not pr.window_should_close():  # Keep going until user clicks X
    # Check if the user is holding down the shift key (for backwards moves)
    shift_held = pr.is_key_down(pr.KEY_LEFT_SHIFT) or pr.is_key_down(pr.KEY_RIGHT_SHIFT)
    
    # Listen for keyboard buttons being pressed (like controls in a video game)
    
    # If user presses X and we have a camera and already took pictures
    if pr.is_key_pressed(pr.KEY_X) and camera_available and capture_completed:
        print("Starting recapture mode...")  # Tell user what's happening
        try:  # Try to do this, but be ready if something goes wrong
            if cv_solver.recapture_faces():  # Take new pictures of cube faces
                print("Recapture completed successfully!")  # Success message
                rubik_cube.update_colors(cv_solver.cube_state)  # Update cube colors
            else:
                print("Recapture cancelled or failed")  # Failed message
        except Exception as e:  # If something breaks
            print(f"Error during recapture: {e}")  # Tell us what broke

    # Manual controls - these let you turn the cube by hand using keyboard
    elif pr.is_key_pressed(pr.KEY_F):  # F key = Front face
        axis = np.array([0, 0, 1])  # Direction to rotate (front-back)
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)
    elif pr.is_key_pressed(pr.KEY_R):  # F key = Front face
        axis = np.array([1, 0, 0]) 
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)

    elif pr.is_key_pressed(pr.KEY_U):  # U key = Up face
        axis = np.array([0, 1, 0])  # Direction to rotate (up-down)
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 2, not shift_held)
    elif pr.is_key_pressed(pr.KEY_L):  # L key = Left face
        axis = np.array([1, 0, 0])  # Direction to rotate (left-right)
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    elif pr.is_key_pressed(pr.KEY_B):  # B key = Back face
        axis = np.array([0, 0, 1])  # Direction to rotate (front-back)
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    elif pr.is_key_pressed(pr.KEY_D):  # D key = Down face
        axis = np.array([0, 1, 0])  # Direction to rotate (up-down)
        rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, 0, not shift_held)
    
    # Computer Vision Solver Controls (using camera to solve automatically)
    elif pr.is_key_pressed(pr.KEY_C) and camera_available:  # C key = Capture with camera
        # Take pictures of the cube to see what colors are where
        print("Starting cube capture...")  # Tell user what's happening
        solver_mode = True  # Switch to camera solver mode
        try:  # Try to do this, but be ready if something goes wrong
            if cv_solver.capture_cube_state():  # Take pictures of all cube faces
                print("Cube state captured successfully!")  # Success message
                rubik_cube.update_colors(cv_solver.cube_state)  # Update cube with real colors
                capture_completed = True  # Remember we finished taking pictures
            else:
                print("Failed to capture cube state")  # Failed message
                capture_completed = False  # Remember we didn't finish

        except Exception as e:  # If something breaks
            print(f"Error capturing cube: {e}")  # Tell us what broke
            solver_mode = False      # Turn off solver mode
            capture_completed = False # Remember we didn't finish
    
    elif pr.is_key_pressed(pr.KEY_X) and camera_available:  # X key = Manual capture
        print("Starting manual cube capture mode...")  # Tell user what's happening
        try:  # Try to do this, but be ready if something goes wrong
            if cv_solver.capture_cube_state():  # Take pictures manually
                print("Cube state captured successfully!")  # Success message
                rubik_cube.update_colors(cv_solver.cube_state)  # Update cube colors
                capture_completed = True  # Remember we finished
            else:
                print("Capture cancelled or incomplete")  # User cancelled
        except Exception as e:  # If something breaks
            print(f"Error during capture: {e}")  # Tell us what broke

    
    elif pr.is_key_pressed(pr.KEY_S) and solver_mode and capture_completed:  # S key = Solve
        # Figure out how to solve the cube we photographed
        try:  # Try to do this, but be ready if something goes wrong
            if cv_solver.solve_cube():  # Calculate solution steps
                solution_ready = True  # Remember we have a solution
                print("Solution generated! Press SPACE to step through moves or A for auto-solve.")
            else:
                print("Failed to generate solution")  # Couldn't figure it out
        except Exception as e:  # If something breaks
            print(f"Error generating solution: {e}")  # Tell us what broke
    
    elif pr.is_key_pressed(pr.KEY_SPACE) and solution_ready:  # SPACE = Next step
        # Do the next move in the solution (one step at a time)
        if not rubik_cube.is_rotating:  # Only if cube isn't already moving
            move_data = integrate_cv_solver_with_rubik(rubik_cube, cv_solver)  # Get next move
            if move_data:  # If we got a valid move
                axis, level, clockwise = move_data  # Break down the move details
                rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, clockwise)  # Add move to queue
                
                # Move to the next step in our solution
                if not cv_solver.next_step():  # Try to go to next step
                    print("Solution complete! Cube should be solved.")  # We're done!
                    solution_ready = False  # No more solution to follow
                    solver_mode = False     # Exit solver mode
            else:
                print("No more moves or invalid move")  # Something went wrong
                solution_ready = False  # Stop the solution
    
    elif pr.is_key_pressed(pr.KEY_A) and solution_ready:  # A key = Auto-solve
        # Turn on/off automatic solving (computer does all moves by itself)
        auto_solve = not auto_solve  # Flip between on and off
        print(f"Auto-solve mode: {'ON' if auto_solve else 'OFF'}")  # Tell user current state
    
    elif pr.is_key_pressed(pr.KEY_ESCAPE):  # ESC key = Reset everything
        # Start over and clear everything
        cv_solver.reset_solution()  # Clear the solution
        solution_ready = False      # No solution ready
        solver_mode = False         # Exit solver mode
        auto_solve = False          # Turn off auto-solve
        capture_completed = False   # Forget captured images
        print("Solution reset")     # Tell user we reset
    
    # Auto-solve functionality (computer automatically does moves)
    if auto_solve and solution_ready and not rubik_cube.is_rotating:  # If auto-solve is on and cube isn't moving
        move_data = integrate_cv_solver_with_rubik(rubik_cube, cv_solver)  # Get next move
        if move_data:  # If we got a valid move
            move_steps = map_move_to_game_input(move_data)
            for axis, level, clockwise in move_steps:
                rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, clockwise)

            # Break down the move details
            rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, clockwise)  # Add move to queue
            
            if not cv_solver.next_step():  # Try to go to next step
                print("Auto-solve complete!")  # We finished solving!
                auto_solve = False      # Turn off auto-solve
                solution_ready = False  # No more solution
                solver_mode = False     # Exit solver mode
        else:  # Something went wrong
            auto_solve = False      # Turn off auto-solve
            solution_ready = False  # Stop the solution

    # Handle rotation animation (make the cube actually turn smoothly)
    rotation_queue, _ = rubik_cube.handle_rotation(rotation_queue)
    
    import numpy as np  # ensure this is at the top of your file if not already

    def map_move_to_game_input(move):
        clockwise = True
        times = 1

        if move.endswith("'"):
            clockwise = False
            move = move[0]
        elif move.endswith("2"):
            times = 2
            move = move[0]

        axis_map = {
            'U': (np.array([0, 1, 0]), 2),
            'D': (np.array([0, 1, 0]), 0),
            'R': (np.array([1, 0, 0]), 2),
            'L': (np.array([1, 0, 0]), 0),
            'F': (np.array([0, 0, 1]), 2),
            'B': (np.array([0, 0, 1]), 0)
        }

        axis, level = axis_map.get(move, (None, None))
        if axis is None:
            return []
        return [(axis, level, clockwise)] * times


    # Update camera position (so we can look around the cube)
    pr.update_camera(configs.camera, pr.CAMERA_THIRD_PERSON)

    # Draw everything on the screen (like painting a picture)
    pr.begin_drawing()  # Start drawing
    pr.clear_background(pr.RAYWHITE)  # Make background white
    
    pr.begin_mode3d(configs.camera)  # Switch to 3D drawing mode
    pr.draw_grid(20, 1.0)  # Draw a grid on the floor to help see depth

    # Draw all parts of the Rubik's cube
    for cube_group in rubik_cube.cubes:  # Go through each group of cube pieces
        for cube_part in cube_group:     # Go through each individual piece
            pr.draw_model(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.WHITE)      # Draw the colored piece
            pr.draw_model_wires(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.DARKGRAY)  # Draw black lines around it

    pr.end_mode3d()  # Stop 3D drawing mode
    
    # Draw UI (the text instructions on screen)
    y_offset = 10  # Start writing text 10 pixels from the top
    
    # Manual controls instructions
    pr.draw_text(b"Manual Controls:", 10, y_offset, 16, pr.DARKGRAY)  # Title
    y_offset += 25  # Move down 25 pixels for next line
    pr.draw_text(b"R-Right, L-Left, U-Up, D-Down, F-Front, B-Back", 10, y_offset, 12, pr.DARKGRAY)  # Key instructions
    y_offset += 20  # Move down for next line
    pr.draw_text(b"Hold SHIFT for counter-clockwise", 10, y_offset, 12, pr.DARKGRAY)  # Shift key info
    y_offset += 30  # Move down more for next section
    
    # CV Solver controls instructions
    pr.draw_text(b"CV Solver:", 10, y_offset, 16, pr.DARKGRAY)  # Camera solver title
    y_offset += 25  # Move down for next line
    
    if camera_available:  # If camera is working
        pr.draw_text(b"C-Capture cube | X-Recapture faces | S-Solve | SPACE-Next | A-Auto | ESC-Reset", 10, y_offset, 12, pr.DARKGRAY)  # Camera controls
        y_offset += 20  # Move down for next line
        
        if capture_completed:  # If we already took pictures
            pr.draw_text(b"Capture completed! Use X to recapture individual faces if needed", 10, y_offset, 11, pr.DARKGREEN)  # Success message
        else:  # If we haven't taken pictures yet
            pr.draw_text(b"Press C to start capturing cube faces", 10, y_offset, 11, pr.GRAY)  # Instruction
    else:  # If camera doesn't work
        pr.draw_text(b"Camera not available", 10, y_offset, 12, pr.RED)  # Error message
    y_offset += 25  # Move down for next section
    
    # Current status messages
    if rubik_cube.is_rotating:  # If cube is currently turning
        pr.draw_text(b"Rotating...", 10, y_offset, 16, pr.RED)  # Show that it's moving
        y_offset += 25  # Move down for next line
    
    if shift_held:  # If user is holding shift key
        pr.draw_text(b"SHIFT: Counter-Clockwise Mode", 10, y_offset, 14, pr.BLUE)  # Show shift mode
        y_offset += 25  # Move down for next line
    
    if solver_mode:  # If we're using camera solver
        pr.draw_text(b"CV Solver Mode Active", 10, y_offset, 16, pr.GREEN)  # Show solver is active
        y_offset += 25  # Move down for next line
        
        if solution_ready:  # If we have a solution ready to use
            # Show current move and progress
            current_move = cv_solver.get_current_move()  # Get what move comes next
            if current_move[0]:  # If there is a next move
                move_text = f"Next: {current_move[0]} - {current_move[1]}".encode()  # Create text showing next move
                pr.draw_text(move_text, 10, y_offset, 14, pr.DARKBLUE)  # Show the next move
                y_offset += 20  # Move down for next line
            
            current_step, total_steps = cv_solver.get_progress()  # Get how far along we are
            progress_text = f"Progress: {current_step}/{total_steps}".encode()  # Create progress text
            pr.draw_text(progress_text, 10, y_offset, 14, pr.DARKBLUE)  # Show progress
            y_offset += 20  # Move down for next line
            
            if auto_solve:  # If auto-solve is turned on
                pr.draw_text(b"AUTO-SOLVE ON", 10, y_offset, 14, pr.ORANGE)  # Show auto-solve status
                y_offset += 20  # Move down for next line
            
            if cv_solver.is_solved:  # If the cube is completely solved
                pr.draw_text(b"SOLVED!", 10, y_offset, 20, pr.GREEN)  # Show success message!
        elif capture_completed:  # If we took pictures but haven't solved yet
            pr.draw_text(b"Ready to solve! Press S to generate solution", 10, y_offset, 14, pr.BLUE)  # Instruction
    
    
    if capture_completed:  # If we've taken pictures, show capture controls
        y_offset += 10  # Move down a bit
        pr.draw_text(b"Capture Controls (when in capture mode):", 10, y_offset, 12, pr.DARKGRAY)  # Title
        y_offset += 15  # Move down for next line
        pr.draw_text(b"S-Capture face | R-Review | 1-6-Jump to face | C-Clear all | Q-Quit", 10, y_offset, 10, pr.DARKGRAY)  # Controls
    
    pr.end_drawing()  # Finish drawing everything on screen

# Cleanup (close everything properly when program ends)
if camera_available:  # If we used a camera
    cv_solver.release_camera()  # Turn off the camera properly
pr.close_window()  # Close the window