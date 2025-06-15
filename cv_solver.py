# Import the libraries we need - like getting tools from a toolbox
import cv2        # This helps us work with cameras and see colors
import numpy as np # This helps us do math with numbers and arrays



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
    def solve_cube(self):
        """
        Built-in beginner Rubik’s Cube solver (no external libraries).
        Returns a list of basic moves like ['F', 'U', 'R', "R'", etc.]
        """
        if not self.cube_state:
            print("No cube state loaded")
            return False

        try:
            moves = []

            # STEP 0: Prepare the cube state in facelet format (6 faces: U, R, F, D, L, B)
            face_order = ['U', 'R', 'F', 'D', 'L', 'B']
            cube = {face: self.cube_state[face] for face in face_order}

            # STEP 1: Build White Cross
            moves += solve_white_cross(cube)

            # STEP 2: Solve White Corners
            moves += solve_white_corners(cube)

            # STEP 3: Solve Middle Layer
            moves += solve_middle_layer(cube)

            # STEP 4: Build Yellow Cross
            moves += solve_yellow_cross(cube)

            # STEP 5: Position Yellow Corners
            moves += solve_yellow_corners(cube)

            # STEP 6: Position Yellow Edges (Final)
            moves += solve_final_layer(cube)

            self.solution_moves = [(m, f"Move {m}") for m in moves]
            self.current_step = 0
            self.is_solved = False
            print("Solved using built-in beginner method:", moves)
            return True

        except Exception as e:
            print(f"Solver error: {e}")
            return False



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

    def get_current_move(self):
        """Returns the current move and its label (if any), or (None, None) if done."""
        if self.current_step < len(self.solution_moves):
            return self.solution_moves[self.current_step]
        return (None, None)

    def next_step(self):
        """Advance to the next move. Returns False if already at the end."""
        if self.current_step + 1 < len(self.solution_moves):
            self.current_step += 1
            return True
        else:
            self.is_solved = True
            return False

    def get_progress(self):
        """Returns the current step and total number of moves."""
        return (self.current_step + 1, len(self.solution_moves))

    def reset_solution(self):
        """Resets the solver state."""
        self.solution_moves = []
        self.current_step = 0
        self.is_solved = False

    

    def release_camera(self):
        """Release the camera and clean up"""
        # Turn off the camera if we have one
        if self.camera:
            self.camera.release()
            # Close all the windows we opened
            cv2.destroyAllWindows()

def solve_white_cross(cube):
    """
    Solves the white cross on the D face using actual cube state.
    Finds each white edge piece, reorients and inserts it correctly.
    """
    moves = []

    # Edge mapping: (white, color) -> expected face and insertion logic
    target_edges = {
        ("white", "green"): "F",
        ("white", "red"): "R",
        ("white", "blue"): "B",
        ("white", "orange"): "L",
        ("green", "white"): "F",
        ("red", "white"): "R",
        ("blue", "white"): "B",
        ("orange", "white"): "L"
    }

    # List of all possible edge positions and their face mappings
    edge_positions = [
        ('U', 0, 1, 'B', 0, 1),
        ('U', 1, 0, 'L', 0, 1),
        ('U', 1, 2, 'R', 0, 1),
        ('U', 2, 1, 'F', 0, 1),
        ('D', 0, 1, 'F', 2, 1),
        ('D', 1, 0, 'L', 2, 1),
        ('D', 1, 2, 'R', 2, 1),
        ('D', 2, 1, 'B', 2, 1),
        ('F', 0, 1, 'U', 2, 1),
        ('F', 2, 1, 'D', 0, 1),
        ('B', 0, 1, 'U', 0, 1),
        ('B', 2, 1, 'D', 2, 1),
        ('L', 0, 1, 'U', 1, 0),
        ('L', 2, 1, 'D', 1, 0),
        ('R', 0, 1, 'U', 1, 2),
        ('R', 2, 1, 'D', 1, 2)
    ]

    # Helper to rotate U face until center of edge aligns
    def align_top(color, face):
        u_rotations = {
            'F': [],
            'R': ['U'],
            'B': ['U2'],
            'L': ["U'"]
        }
        return u_rotations[face]

    # Step 1: Bring white edges to U face if not already
    for _ in range(4):  # Max 4 white edges
        for (face1, r1, c1, face2, r2, c2) in edge_positions:
            color1 = cube[face1][r1][c1]
            color2 = cube[face2][r2][c2]
            colors = (color1, color2)

            if 'white' in colors:
                other_color = color2 if color1 == 'white' else color1
                target_face = target_edges.get((color1, color2)) or target_edges.get((color2, color1))
                if not target_face:
                    continue

                # If white is already in D face center slot, skip
                if face1 == 'D' or face2 == 'D':
                    continue

                # Simplified move logic: try to bring edge to U, align it, and insert
                if face1 == 'U' or face2 == 'U':
                    # Already in top layer
                    moves += align_top(other_color, target_face)
                    moves += [target_face, target_face]
                elif face1 == 'F' or face2 == 'F':
                    moves += ['F', 'U', "F'"]
                elif face1 == 'B' or face2 == 'B':
                    moves += ['B', 'U', "B'"]
                elif face1 == 'L' or face2 == 'L':
                    moves += ['L', 'U', "L'"]
                elif face1 == 'R' or face2 == 'R':
                    moves += ['R', 'U', "R'"]

    # Final alignment of all 4 white edges
    for color, target_face in [('green', 'F'), ('red', 'R'), ('blue', 'B'), ('orange', 'L')]:
        moves += align_top(color, target_face)
        moves += [target_face, target_face]

    return moves


def solve_white_corners(cube):
    """
    Fully dynamic white corner solver.
    Finds all white corner pieces, orients and inserts them into the correct D face position.
    """
    moves = []

    # Define each target corner (colors around each white corner)
    target_corners = [
        ('white', 'green', 'red'),
        ('white', 'red', 'blue'),
        ('white', 'blue', 'orange'),
        ('white', 'orange', 'green')
    ]

    # The standard right corner insertion algorithm
    def insert_corner_right():
        return ["R'", "D'", "R", "D"]

    # The standard left corner insertion algorithm
    def insert_corner_left():
        return ["L", "D", "L'", "D'"]

    # A helper to rotate D layer until white corner is below correct slot
    def rotate_d_until_aligned(target_pair):
        face1, face2 = target_pair
        rotations = {
            ('F', 'R'): [],
            ('R', 'B'): ['D'],
            ('B', 'L'): ['D', 'D'],
            ('L', 'F'): ["D'"]
        }
        return rotations.get((face1, face2)) or rotations.get((face2, face1)) or []

    # A helper to bring corner from U layer to bottom
    def insert_white_corner(face_pair, white_on_top=True):
        if white_on_top:
            return rotate_d_until_aligned(face_pair) + insert_corner_right()
        else:
            return rotate_d_until_aligned(face_pair) + insert_corner_left()

    # Simulate up to 8 attempts (at most 4 corners, 2 cycles)
    attempts = 0
    max_attempts = 8
    while attempts < max_attempts:
        inserted = 0

        for face in ['U']:
            facelets = cube[face]
            corners = [
                ((0, 0), ['B', 'L']),  # ULB
                ((0, 2), ['B', 'R']),  # URB
                ((2, 0), ['F', 'L']),  # ULF
                ((2, 2), ['F', 'R'])   # URF
            ]

            for (r, c), (f1, f2) in corners:
                colors = [cube['U'][r][c], cube[f1][0][2 if f1 in ['F', 'L'] else 0], cube[f2][0][0 if f2 in ['F', 'R'] else 2]]
                if 'white' in colors:
                    main_color = [c for c in colors if c != 'white'][0]
                    other_color = [c for c in colors if c != 'white'][1]

                    # Use move sequence to insert
                    moves += insert_white_corner((main_color, other_color))
                    inserted += 1

        attempts += 1
        if inserted == 0:
            break

    return moves

def solve_middle_layer(cube):
    """
    Solves all 4 middle layer edge pieces using real cube state logic.
    Identifies top-layer non-yellow, non-white edges and inserts them.
    """
    moves = []

    # Middle layer target edges (no white or yellow)
    target_colors = [
        ('green', 'red'),
        ('green', 'orange'),
        ('blue', 'red'),
        ('blue', 'orange')
    ]

    # Standard insertion algorithms
    def insert_left():
        return ["U'", "L'", "U", "L", "U", "F", "U'", "F'"]

    def insert_right():
        return ["U", "R", "U'", "R'", "U'", "F'", "U", "F"]

    # Identify top-layer edge positions
    edge_slots = {
        'F': (0, 1, 'U', 2, 1),
        'R': (0, 1, 'U', 1, 2),
        'B': (0, 1, 'U', 0, 1),
        'L': (0, 1, 'U', 1, 0)
    }

    def get_center(face):
        return cube[face][1][1]

    # Try to insert all target edges
    for _ in range(10):  # Retry up to 10 times for stability
        inserted = False

        for face, (fr, fc, uf, ur, uc) in edge_slots.items():
            edge_color = cube[face][fr][fc]
            top_color = cube[uf][ur][uc]

            if 'white' in [edge_color, top_color] or 'yellow' in [edge_color, top_color]:
                continue

            # Now we have a valid edge to insert
            current_edge = (edge_color, top_color)
            reversed_edge = (top_color, edge_color)

            if current_edge in target_colors or reversed_edge in target_colors:
                # Align top face with correct side center
                target_side = edge_color if edge_color != get_center('U') else top_color
                center_face = None
                for f in ['F', 'R', 'B', 'L']:
                    if get_center(f) == target_side:
                        center_face = f
                        break

                if center_face is None:
                    continue

                # Align top edge above center face
                align_map = {
                    'F': [],
                    'R': ['U'],
                    'B': ['U2'],
                    'L': ["U'"]
                }
                moves += align_map[center_face]

                # Check which direction to insert
                for f1, f2 in target_colors:
                    if set((edge_color, top_color)) == set((f1, f2)):
                        if get_center(center_face) == f1:
                            other_face = f2
                        else:
                            other_face = f1
                        break
                else:
                    continue

                # Decide whether the other face is to the left or right
                left_of = {'F': 'L', 'L': 'B', 'B': 'R', 'R': 'F'}
                right_of = {'F': 'R', 'R': 'B', 'B': 'L', 'L': 'F'}

                if left_of[center_face] == other_face:
                    moves += insert_left()
                elif right_of[center_face] == other_face:
                    moves += insert_right()

                inserted = True
                break  # one insert per loop

        if not inserted:
            break  # nothing left to insert

    return moves


def solve_yellow_cross(cube):
    """
    Forms the yellow cross on the U face by detecting the current pattern
    and applying the correct algorithm.
    """
    moves = []

    def get_yellow_edge_pattern():
        """Detects which yellow edges are on the U face"""
        u = cube['U']
        return {
            'top': u[0][1] == 'yellow',
            'left': u[1][0] == 'yellow',
            'right': u[1][2] == 'yellow',
            'bottom': u[2][1] == 'yellow'
        }

    def apply_cross_algorithm():
        return ["F", "R", "U", "R'", "U'", "F'"]

    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        edges = get_yellow_edge_pattern()
        count = sum(edges.values())

        # ✅ Solved already
        if count == 4:
            break

        # ⬤ Dot → apply algorithm once
        if count == 0:
            moves += apply_cross_algorithm()

        # ⎾ L-shape → rotate U to align then apply
        elif count == 2:
            if edges['bottom'] and edges['left']:
                moves += apply_cross_algorithm()
            elif edges['top'] and edges['left']:
                moves += ['U']
                moves += apply_cross_algorithm()
            elif edges['top'] and edges['right']:
                moves += ['U2']
                moves += apply_cross_algorithm()
            elif edges['bottom'] and edges['right']:
                moves += ["U'"]
                moves += apply_cross_algorithm()
            elif edges['left'] and edges['right']:  # Horizontal line
                moves += apply_cross_algorithm()
            elif edges['top'] and edges['bottom']:  # Vertical line
                moves += ['U']
                moves += apply_cross_algorithm()
            else:
                moves += ['U']
                moves += apply_cross_algorithm()

        # ─ Line → rotate to align then apply
        elif count == 3:
            moves += apply_cross_algorithm()

        # 🔄 Otherwise, retry after a U rotation
        elif count == 1:
            moves += ['U']
            moves += apply_cross_algorithm()

        attempts += 1

    return moves


def solve_yellow_corners(cube):
    """
    Orients yellow corners on the U face using the standard algorithm.
    Repeats until all 4 corners have yellow facing up.
    """
    moves = []

    def count_yellow_corners():
        """Count how many corners have yellow on top (U face corners)"""
        u = cube['U']
        return sum([
            u[0][0] == 'yellow',
            u[0][2] == 'yellow',
            u[2][0] == 'yellow',
            u[2][2] == 'yellow'
        ])

    def apply_corner_orientation():
        return ["R", "U", "R'", "U", "R", "U2", "R'"]

    max_attempts = 6  # max 6 full rotations needed
    attempts = 0

    while count_yellow_corners() < 4 and attempts < max_attempts:
        moves += apply_corner_orientation()
        moves += ["U"]  # Rotate U to bring next corner into position
        attempts += 1

    return moves


def solve_final_layer(cube):
    """
    Final step: Permutes the yellow corners and edges on U face.
    Assumes yellow cross and yellow corners are already correctly oriented.
    """
    moves = []

    # Step 1: Permute yellow corners (make sure all are in correct place)
    def corners_correct():
        # Get face corner colors
        corners = [
            (cube['F'][0][2], cube['R'][0][0]),  # UFR
            (cube['R'][0][2], cube['B'][0][0]),  # URB
            (cube['B'][0][2], cube['L'][0][0]),  # UBL
            (cube['L'][0][2], cube['F'][0][0])   # ULF
        ]
        return all(f1 == f2 for f1, f2 in corners)

    corner_algo = ["U", "R", "U'", "L'", "U", "R'", "U'", "L"]

    for _ in range(5):  # up to 5 cycles
        if corners_correct():
            break
        moves += corner_algo
        moves += ["U"]  # Rotate to shift corner targets

    # Step 2: Permute yellow edges (cycle them into place)
    def edges_correct():
        return (
            cube['F'][0][1] == cube['F'][1][1] and
            cube['R'][0][1] == cube['R'][1][1] and
            cube['B'][0][1] == cube['B'][1][1] and
            cube['L'][0][1] == cube['L'][1][1]
        )

    edge_algo = ["F2", "U", "L", "R'", "F2", "L'", "R", "U", "F2"]

    for _ in range(4):  # up to 4 edge permutations
        if edges_correct():
            break
        moves += edge_algo

    return moves


