# Import the libraries we need - like getting tools from a toolbox
import cv2        # This helps us work with cameras and see colors
import numpy as np # This helps us do math with numbers and arrays
from ida_solver import ida_star


# This is like a smart helper that can see your Rubik's cube and help solve it
class RubikCVSolver:
    def __init__(self):
        # Start with no information about the cube
        self.cube_state = None
        # Start with an empty list of moves to solve the cube
        self.solution_moves = []
        # Keep track of which step we're on (start at step 0)
        self.current_step = 0
        # The cube starts unsolved
        self.is_solved = False
        # We don't have a camera yet
        self.camera = None

        # This is like a color dictionary - it tells the computer what each color looks like
        # Each color has a range of values (like saying "red is between this shade and that shade")
        self.color_ranges = {
            'white': ([0, 0, 180], [180, 60, 255]),      # White colors look like this
            'yellow': ([21, 100, 100], [35, 255, 255]),  # Yellow colors look like this
            'red': ([0, 100, 100], [2, 200, 200]),       # Red colors look like this
            'orange': ([1, 50, 50], [15, 255, 255]),     # Orange colors look like this
            'green': ([40, 50, 50], [80, 255, 255]),     # Green colors look like this
            'blue': ([50, 50, 50], [130, 255, 255])      # Blue colors look like this
        }

        # These are the names for each face of the cube (like Front, Back, etc.)
        self.face_names = ['F', 'B', 'R', 'L', 'U', 'D']

    def initialize_camera(self, camera_index=0):
        """Try to connect to the camera (like turning on a webcam)"""
        self.camera = cv2.VideoCapture(camera_index)
        # Check if the camera actually turned on
        if not self.camera.isOpened():
            # If camera won't work, complain about it
            raise Exception("Could not open camera")
        # Set the camera to take pictures that are 640 pixels wide
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # Set the camera to take pictures that are 480 pixels tall
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def capture_cube_state(self):
        """Capture the current state of the cube using the camera"""
        # Make sure we have a camera to work with
        if not self.camera:
            # If no camera, complain about it
            raise Exception("Camera not initialized")

        # Print instructions for the user
        print("=== Rubik Cube Capture ===")
        print("Press 1-6 to capture a face (can be in any order)")
        print("1=F, 2=R, 3=B, 4=L, 5=U, 6=D")
        print("Press Q when done")

        # Create an empty dictionary to store the colors we capture
        captured_faces = {}
        # List of face names in order (Front, Right, Back, Left, Up, Down)
        face_keys = ['F', 'R', 'B', 'L', 'U', 'D']
        # Friendly names for each face so users understand better
        face_labels = {
            'F': "FRONT", 'R': "RIGHT", 'B': "BACK",
            'L': "LEFT", 'U': "TOP", 'D': "BOTTOM"
        }

        # Keep taking pictures until the user is done
        while True:
            # Take a picture from the camera
            ret, frame = self.camera.read()
            # If the camera didn't give us a picture, try again
            if not ret:
                continue

            # Flip the image so it looks like a mirror (more natural for users)
            frame = cv2.flip(frame, 1)
            # Make a copy of the picture so we can draw on it
            display_frame = frame.copy()

            # Get the size of our picture
            h, w = frame.shape[:2]
            # Find the center of the picture
            center_x, center_y = w // 2, h // 2
            # Decide how big our capture box should be
            rect_size = 200
            # Calculate where to put the capture box (centered)
            x, y = center_x - rect_size // 2, center_y - rect_size // 2

            # Draw UI (User Interface - the things users see on screen)
            # Draw a yellow rectangle where users should put their cube face
            cv2.rectangle(display_frame, (x, y), (x + rect_size, y + rect_size), (0, 255, 255), 2)
            # Write instructions at the top of the screen
            cv2.putText(display_frame, "Press 1-6 to capture face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Write more detailed instructions
            cv2.putText(display_frame, "1=F 2=R 3=B 4=L 5=U 6=D, Q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Show which faces we've already captured
            for i, key in enumerate(face_keys):
                # Green color if we captured this face, gray if we haven't
                color = (0, 255, 0) if key in captured_faces else (100, 100, 100)
                # Show a checkmark if captured, dash if not
                cv2.putText(display_frame, f"{key}: {'Captured' if key in captured_faces else 'Not captured'}", 
                            (10, 100 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Show the picture on screen
            cv2.imshow("Rubik Cube Capture", display_frame)
            # Wait for the user to press a key
            key = cv2.waitKey(1) & 0xFF

            # Check if the user pressed numbers 1-6
            if key in [ord(str(i)) for i in range(1, 7)]:
                # Convert the key press to a number (1-6 becomes 0-5)
                idx = int(chr(key)) - 1
                # Get the face name (F, R, B, L, U, or D)
                face_name = face_keys[idx]
                # Tell the user what we're doing
                print(f"Capturing {face_labels[face_name]} face ({face_name})...")
                # Create information about where to look in the picture
                face_info = {'bbox': (x, y, rect_size, rect_size)}
                # Try to figure out what colors are on this face
                colors = self.extract_colors_from_face(frame, face_info)
                # If we successfully found colors
                if colors:
                    # Flip each row to correct the mirrored camera view
                    # Flip only certain faces to fix mirrored layout
                    if face_name in ['F', 'B', 'L', 'R']:
                        colors = [row[::-1] for row in colors]  # horizontal flip

                    captured_faces[face_name] = colors

                    # Tell the user we got it
                    print(f"Captured {face_name}:")
                    # Show the colors we found
                    for row in colors:
                        print(f"  {row}")
                    # Show a success message on screen
                    cv2.putText(display_frame, "Captured!", (center_x - 50, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    # Update the screen
                    cv2.imshow('Rubik Cube Capture', display_frame)
                    # Wait half a second so user can see the success message
                    cv2.waitKey(500)

            # If user pressed 'q', they want to quit
            elif key == ord('q'):
                print("Capture finished.")
                break

        # Close the camera window
        cv2.destroyAllWindows()

        # Check if we got all 6 faces
        if len(captured_faces) == 6:
            # Save all the face colors
            self.cube_state = captured_faces
            print("All faces captured!")
            return True
        else:
            # We didn't get all faces
            print(f"Only {len(captured_faces)} faces captured.")
            return False

    def recapture_faces(self):
        """This function lets users re-take pictures of cube faces if they made a mistake"""
        # Print instructions for the user
        print("=== Recapture Mode ===")
        print("Press 1-6 to choose face to recapture (1=F, 2=R, 3=B, 4=L, 5=U, 6=D), q to quit")

        # Map number keys to face names
        face_mapping = {
            ord('1'): 'F', ord('2'): 'R', ord('3'): 'B',  # 1=Front, 2=Right, 3=Back
            ord('4'): 'L', ord('5'): 'U', ord('6'): 'D'   # 4=Left, 5=Up, 6=Down
        }

        # Keep taking pictures until user quits
        while True:
            # Take a picture from the camera
            ret, frame = self.camera.read()
            # If camera didn't work, try again
            if not ret:
                continue

            # Flip the image like a mirror
            frame = cv2.flip(frame, 1)
            # Get the picture size
            h, w = frame.shape[:2]
            # Find the center of the picture
            center_x, center_y = w // 2, h // 2
            # Set the size of our capture box
            rect_size = 200
            # Calculate where to put the capture box
            x = center_x - rect_size // 2
            y = center_y - rect_size // 2

            # Make a copy so we can draw on it
            display_frame = frame.copy()
            # Write instructions on the screen
            cv2.putText(display_frame, "Press 1-6 to recapture that face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to exit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Draw the capture box
            cv2.rectangle(display_frame, (x, y), (x + rect_size, y + rect_size), (0, 255, 255), 2)

            # Show the picture on screen
            cv2.imshow("Recapture", display_frame)
            # Wait for user to press a key
            key = cv2.waitKey(1) & 0xFF

            # If user pressed a number key (1-6)
            if key in face_mapping:
                # Tell user what we're doing
                print(f"Recapturing face {face_mapping[key]}")
                # Set up the capture area
                face_info = {'bbox': (x, y, rect_size, rect_size)}
                # Try to detect colors in this area
                colors = self.extract_colors_from_face(frame, face_info)
                # If we found colors
                if colors:
                    # ✅ Flip each row to correct the mirrored camera view
                    # Flip only certain faces to fix mirrored layout
                    if face_name in ['F', 'B', 'L', 'R']:
                        colors = [row[::-1] for row in colors]  # horizontal flip

                    self.cube_state[face_mapping[key]] = colors

                    # Tell user we got it
                    print(f"Recaptured {face_mapping[key]}:", colors)
            # If user pressed 'q', they want to quit
            elif key == ord('q'):
                break

        # Close all camera windows
        cv2.destroyAllWindows()
        return True

    def extract_colors_from_face(self, frame, face_info):
        """Extract colors from a cube face in the camera frame"""
        # Get the area of the picture where the cube face should be
        x, y, w, h = face_info['bbox']
        # Cut out just that part of the picture
        face_roi = frame[y:y + h, x:x + w]
        # Make sure we actually got some pixels
        if face_roi.size == 0:
            return None
        # Make the face image bigger so it's easier to work with
        face_roi = cv2.resize(face_roi, (300, 300))
        # Create an empty list to store colors
        colors = []
        # Each small square on the face is 100 pixels big
        cell_size = 100
        # Look at each row of squares (3 rows total)
        for row in range(3):
            # Create a list for this row's colors
            row_colors = []
            # Look at each column in this row (3 columns total)
            for col in range(3):
                # Cut out one small square, leaving some border space
                cell = face_roi[row * cell_size + 20:(row + 1) * cell_size - 20,
                                col * cell_size + 20:(col + 1) * cell_size - 20]
                # Figure out what color this square is
                row_colors.append(self.get_dominant_color(cell))
            # Add this row of colors to our main list
            colors.append(row_colors)
        # Return all the colors we found
        return colors

    def get_dominant_color(self, image_section):
        """Determine the dominant color in an image section"""
        # Convert the image from normal colors (BGR) to HSV (better for detecting colors)
        hsv = cv2.cvtColor(image_section, cv2.COLOR_BGR2HSV)
        # Get the size of our image section
        h, w = hsv.shape[:2]
        # Look at just the center part (ignore the edges)
        center = hsv[h // 3:2 * h // 3, w // 3:2 * w // 3]
        # Count how many pixels we're looking at
        total_pixels = center.shape[0] * center.shape[1]

        # Start assuming we don't know what color this is
        best_match = 'unknown'
        # Keep track of the best color match we've found
        max_ratio = 0

        # Try each color in our color dictionary
        for color, (lower, upper) in self.color_ranges.items():
            # Convert the color ranges to the right format
            lower = np.array(lower)
            upper = np.array(upper)

            # Red is special because it wraps around in the color wheel
            if color == 'red':
                # Look for red in two different ranges
                mask1 = cv2.inRange(center, np.array([0, 120, 70]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(center, np.array([170, 120, 70]), np.array([180, 255, 255]))
                # Combine both red ranges
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                # For other colors, just look in one range
                mask = cv2.inRange(center, lower, upper)

            # Count how many pixels match this color
            match = cv2.countNonZero(mask) / total_pixels
            # If this is the best match so far and it's good enough
            if match > max_ratio and match > 0.3:
                max_ratio = match
                best_match = color
        # Return the color we think this is
        return best_match

    def solve_cube(self):
        """Use IDA* to solve the cube based on captured state"""
        if not self.cube_state:
            print("No cube state loaded")
            return False
        try:
            print("Running IDA* solver...")
            moves = ida_star(self.cube_state)
            if not moves:
                print("No solution found")
                return False
            self.solution_moves = [(m, f"Move {m}") for m in moves]
            self.current_step = 0
            self.is_solved = False
            print("Solution found:", moves)
            return True
        except Exception as e:
            print(f"IDA* failed: {e}")
            return False

    def get_current_move(self):
        """Get the current move to execute"""
        # If we've done all the moves, we're done
        if self.current_step >= len(self.solution_moves):
            return None, "Done"
        # Return the current move we should do
        return self.solution_moves[self.current_step]

    def next_step(self):
        """Move to the next step in the solution"""
        # Move to the next step
        self.current_step += 1
        # If we've done all steps, mark the cube as solved
        if self.current_step >= len(self.solution_moves):
            self.is_solved = True
            return False
        # There are still more steps to do
        return True

    def reset_solution(self):
        """Reset the solution to the beginning"""
        # Go back to the beginning
        self.current_step = 0
        # The cube isn't solved anymore
        self.is_solved = False

    def get_progress(self):
        """Get current progress through the solution"""
        # Return which step we're on and how many total steps there are
        return self.current_step, len(self.solution_moves)

    def map_move_to_game_input(self, move):
        """Convert a move string to game input format"""
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

    def convert_to_kociemba_string(self):
        """Convert the captured cube state into a Kociemba-compatible string"""
        if not self.cube_state:
            return None

        face_order = ['U', 'R', 'F', 'D', 'L', 'B']  # Kociemba expects this order
        color_face_map = {}

        # Step 1: Identify center color of each face
        for face in face_order:
            center_color = self.cube_state[face][1][1]
            color_face_map[center_color] = face

        # Step 2: Build the full string
        kociemba_str = ''
        for face in face_order:
            for row in self.cube_state[face]:
                for color in row:
                    kociemba_str += color_face_map.get(color, 'X')  # fallback to 'X' if unknown
        return kociemba_str

    def release_camera(self):
        """Release the camera and clean up"""
        # Turn off the camera if we have one
        if self.camera:
            self.camera.release()
            # Close all the windows we opened
            cv2.destroyAllWindows()


# Standalone helper functions
def map_move_to_game_input(move):
    """Standalone function to convert move string to game input format"""
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


def integrate_cv_solver_with_rubik(rubik_cube, cv_solver):
    """Connect the color-detecting solver with the 3D cube game"""
    if cv_solver.is_solved:
        return None
    move, _ = cv_solver.get_current_move()
    if not move:
        return None
    move_steps = cv_solver.map_move_to_game_input(move)
    if not move_steps:
        return None
    return move_steps[0]  # Return first move step (axis, level, clockwise)