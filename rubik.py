# Import the libraries we need - like getting tools from a toolbox
import raylibpy as pr  # This helps us draw 3D graphics on the screen
import numpy as np     # This helps us do math with numbers and arrays
import random          # This gives us random numbers


# This is like a blueprint for making one small cube piece
class Cube:
    def __init__(self, size, center, face_color):
        # Save the size of our cube (how big it is)
        self.size = size
        # Save where the center of our cube should be (like an address in 3D space)
        self.center = np.array(center, dtype=float)
        # Save what color this cube face should be
        self.face_color = face_color
        # Start with no rotation (like the cube is sitting normally)
        self.orientation = np.eye(3)
        # We don't have a 3D model yet, so set it to None
        self.model = None
        # Create the shape of our cube
        self.gen_mesh(size)
        # Turn that shape into something we can see on screen
        self.create_model()
    

 

    def gen_mesh(self, scale):
        # Check if we got different sizes for width, height, depth
        if isinstance(scale, tuple):
            # Make a cube with different dimensions (like a rectangle box)
            self.mesh = pr.gen_mesh_cube(*scale)
        else:
            # Make a perfect cube where all sides are the same size
            self.mesh = pr.gen_mesh_cube(scale, scale, scale)
    
    def create_model(self):
        # Turn our cube shape into something the computer can draw
        self.model = pr.load_model_from_mesh(self.mesh)
        # Paint our cube with the color we want
        self.model.materials[0].maps[pr.MATERIAL_MAP_DIFFUSE].color = self.face_color
        # Update where the cube should appear on screen
        self.update_transform()
    
    def update_transform(self):
        """Update the model's transformation matrix"""
        # Create a movement instruction to put the cube in the right place
        translation = pr.matrix_translate(self.center[0], self.center[1], self.center[2])
        
        # Convert orientation matrix to raylib format
        # Figure out how much and which way our cube is rotated
        axis, angle = self.get_rotation_axis_angle()
        # Only rotate if there's actually some rotation to do
        if angle > 0.001:  # Only apply rotation if there's a meaningful angle
            # Create rotation instruction
            rotation = pr.matrix_rotate(axis, angle)  # angle is already in radians from get_rotation_axis_angle
            # Combine rotation and movement instructions
            transform = pr.matrix_multiply(rotation, translation)
        else:
            # If no rotation needed, just use the movement instruction
            transform = translation
            
        # Tell our 3D model how to position itself
        self.model.transform = transform

    def rotate(self, axis, theta):
        """Rotate the cube around the specified axis by theta radians"""
        # Choose which way to rotate based on the axis (like choosing x, y, or z direction)
        if axis == 0:  # X-axis (like rolling forward/backward)
            # Create a rotation matrix for X-axis using trigonometry
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif axis == 1:  # Y-axis (like spinning left/right)
            # Create a rotation matrix for Y-axis using trigonometry
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 2:  # Z-axis (like turning a steering wheel)
            # Create a rotation matrix for Z-axis using trigonometry
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        else:
            # If someone gives us a bad axis number, complain!
            raise ValueError("Invalid axis")
        
        # Rotate the center position (move the cube's position)
        self.center = rotation_matrix @ self.center
        # Update orientation (remember how the cube is rotated)
        self.orientation = rotation_matrix @ self.orientation
        # Update the model transform (tell the computer the new position)
        self.update_transform()

    def get_rotation_axis_angle(self):
        """Extract rotation axis and angle from orientation matrix"""
        # Calculate how much our cube is rotated using the orientation matrix
        trace = np.trace(self.orientation)
        # Convert the trace to an angle using inverse cosine
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        # If there's barely any rotation, don't bother
        if angle < 0.001:  # No rotation
            # Use a default axis pointing up
            axis = pr.Vector3(0, 0, 1)  # Default axis
            return axis, 0
        
        # Extract axis (figure out which direction we're rotating around)
        # These formulas come from rotation matrix math
        rx = self.orientation[2, 1] - self.orientation[1, 2]
        ry = self.orientation[0, 2] - self.orientation[2, 0]
        rz = self.orientation[1, 0] - self.orientation[0, 1]
        
        # Normalize the axis vector (make it the right length)
        axis_vec = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        # Convert to the format raylib expects
        axis = pr.Vector3(axis_vec[0], axis_vec[1], axis_vec[2])
        
        # Return both the axis and angle
        return axis, angle  # Return angle in radians

# This is like a blueprint for the whole Rubik's cube
class Rubik:
    def __init__(self):
        # Start with an empty list to hold all our small cubes
        self.cubes = []
        # Create a dictionary that converts color names to actual colors
        self.colors = {
            'white': pr.RAYWHITE,
            'yellow': pr.YELLOW,
            'green': pr.LIME,
            'blue': pr.BLUE,
            'orange': pr.ORANGE,
            'red': pr.RED,
        }
        # FIX: Add color_lookup as an alias to colors for compatibility
        self.color_lookup = self.colors

        # Keep track of whether we're currently rotating a face
        self.is_rotating = False
        # Keep track of how much we've rotated so far
        self.rotation_angle = 0
        # Remember which direction we're rotating around
        self.rotating_axis = None
        # Remember which layer we're rotating
        self.level = None
        # Remember which cubes are part of the rotation
        self.segment = None
        # Remember how much we want to rotate in total
        self.target_rotation = 0
        # Build our 3x3x3 Rubik's cube
        self.generate_rubik(2)

    def update_colors(self, cube_state):
        """Apply captured face colors to the cube's stickers"""
        print(f"Updating colors with cube state: {cube_state}")
        
        face_map = {
            'F': (2, 2),  # Front face - Z axis, positive level
            'B': (2, 0),  # Back face - Z axis, negative level
            'R': (0, 2),  # Right face - X axis, positive level
            'L': (0, 0),  # Left face - X axis, negative level
            'U': (1, 2),  # Up face - Y axis, positive level
            'D': (1, 0)   # Down face - Y axis, negative level
        }

        for face, (axis, level) in face_map.items():
            facelets = cube_state.get(face)
            if not facelets:
                print(f"No facelets found for face {face}")
                continue

            print(f"Processing face {face} with colors: {facelets}")
            unsorted_seg = self.get_face(np.eye(3)[axis], level)
            segment = self.sort_face(unsorted_seg, axis, level)

            for row in range(3):
                for col in range(3):
                    i = row * 3 + col
                    if i >= len(segment): 
                        continue

                    cube_index = segment[i]
                    if cube_index >= len(self.cubes):
                        continue

                    # FIX: Get the correct sticker part based on the face
                    sticker_index = self.get_sticker_index(axis, level)
                    if sticker_index < len(self.cubes[cube_index]):
                        cubelet = self.cubes[cube_index][sticker_index]  # Get the specific sticker
                        
                        # Correct orientation for each face
                        if face == 'F':
                            color_name = facelets[2 - row][col]           # Flip vertically
                        elif face == 'B':
                            color_name = facelets[2 - row][col]       # Flip vertically + horizontally
                        elif face == 'L':
                            color_name = facelets[2 - row][col]       # Flip vertically + horizontally
                        elif face == 'R':
                            color_name = facelets[2 - row][col]           # Flip vertically
                        elif face == 'U':
                            color_name = facelets[row][2 - col]           # Mirror horizontally only
                        elif face == 'D':
                            color_name = facelets[2 - row][col]           # Flip vertically


                        color = self.color_lookup.get(color_name, pr.GRAY)

                        print(f"Setting cube {cube_index} sticker {sticker_index} to {color_name} ({color})")

                        # Update the color of this sticker
                        cubelet.face_color = color
                        cubelet.model.materials[0].maps[pr.MATERIAL_MAP_DIFFUSE].color = color

    def get_sticker_index(self, axis, level):
        """Get the index of the sticker on a cube piece"""
        # Cube structure: [center, front, back, right, left, top, bottom]
        # Indices:         [0,      1,     2,    3,     4,    5,   6]
        # Figure out which sticker face we want based on the axis and level
        if axis == 0:  # X-axis (left/right)
            return 3 if level == 2 else 4  # Right (3) or Left (4)
        elif axis == 1:  # Y-axis (up/down)
            return 5 if level == 2 else 6  # Top (5) or Bottom (6)
        elif axis == 2:  # Z-axis (front/back)
            return 1 if level == 2 else 2  # Front (1) or Back (2)
        # Default to the center piece
        return 0

    def generate_rubik(self, size):
        # Define the 6 standard Rubik's cube colors
        colors = [pr.RED, pr.ORANGE, pr.WHITE, pr.YELLOW, pr.GREEN, pr.BLUE]
        # Set how far apart each small cube should be
        offset = size + 0.05
        # Set how thick the colored stickers should be
        face_thickness = 0.05
        # Set a tiny gap between the cube and its stickers
        sticker_offset = 0.01

        # Define the shapes for stickers on different faces
        size_x = (size, face_thickness, size)  # Stickers for top/bottom faces
        size_y = (face_thickness, size, size)  # Stickers for left/right faces
        size_z = (size, size, face_thickness)  # Stickers for front/back faces

        # Create a 3x3x3 grid of cube pieces
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    # Decide what color each face should be
                    face_colors = [
                        colors[0] if x == 2 else pr.BLACK,  # Right face (red if on right edge)
                        colors[1] if x == 0 else pr.BLACK,  # Left face (orange if on left edge)
                        colors[2] if y == 2 else pr.BLACK,  # Top face (white if on top edge)
                        colors[3] if y == 0 else pr.BLACK,  # Bottom face (yellow if on bottom edge)
                        colors[4] if z == 2 else pr.BLACK,  # Front face (green if on front edge)
                        colors[5] if z == 0 else pr.BLACK,  # Back face (blue if on back edge)
                    ]

                    # Calculate where this cube piece should be positioned
                    center_pos = np.array([(x - 1) * offset, (y - 1) * offset, (z - 1) * offset])
                    # Create the main black cube body
                    center = Cube((size, size, size), center_pos, pr.BLACK)

                    # Create face stickers (the colored squares)
                    # Calculate positions for each sticker (slightly offset from the cube center)
                    front_pos = center_pos + np.array([0, 0, size/2 + sticker_offset])
                    back_pos = center_pos + np.array([0, 0, -size/2 - sticker_offset])
                    right_pos = center_pos + np.array([size/2 + sticker_offset, 0, 0])
                    left_pos = center_pos + np.array([-size/2 - sticker_offset, 0, 0])
                    top_pos = center_pos + np.array([0, size/2 + sticker_offset, 0])
                    bottom_pos = center_pos + np.array([0, -size/2 - sticker_offset, 0])

                    # Create the actual sticker cubes
                    front = Cube(size_z, front_pos, face_colors[4])
                    back = Cube(size_z, back_pos, face_colors[5])
                    right = Cube(size_y, right_pos, face_colors[0])
                    left = Cube(size_y, left_pos, face_colors[1])
                    top = Cube(size_x, top_pos, face_colors[2])
                    bottom = Cube(size_x, bottom_pos, face_colors[3])

                    # Group all parts of this cube piece together
                    self.cubes.append([center, front, back, right, left, top, bottom])

        # Return all the cubes we created
        return self.cubes

    def choose_piece(self, piece, axis_index, level):
        """Determine if a piece belongs to the specified level on the given axis"""
        # Get the coordinate of this piece along the specified axis
        coord = round(piece[0].center[axis_index], 1)
        # Check if this piece belongs to the left/bottom/back layer
        if level == 0 and coord < 0:
            return True
        # Check if this piece belongs to the middle layer
        elif level == 1 and abs(coord) < 0.1:  # Middle layer
            return True
        # Check if this piece belongs to the right/top/front layer
        elif level == 2 and coord > 0:
            return True
        # This piece doesn't belong to the specified layer
        return False
    
    def get_face(self, axis, level):
        """Get all pieces that belong to a specific face/layer"""
        # Figure out which axis we're looking at (0=x, 1=y, 2=z)
        axis_index = np.nonzero(axis)[0][0]
        # Find all cube pieces that belong to this layer
        segment = [i for i, cube in enumerate(self.cubes)
                   if self.choose_piece(cube, axis_index, level)]
        return segment

    def handle_rotation(self, rotation_queue, animation_step=None):
        """Handle the rotation animation"""
        # Start new rotation if queue has items and not currently rotating
        # If we have rotations waiting and we're not busy rotating
        if rotation_queue and not self.is_rotating:
            # Get the next rotation from our to-do list
            self.target_rotation, self.rotating_axis, self.level = rotation_queue.pop(0)
            
            # Add small random offset to avoid floating point issues
            # Add a tiny random number to avoid computer math errors
            if self.target_rotation > 0:
                self.target_rotation += random.uniform(0, 1) * 10**-3
            else:
                self.target_rotation -= random.uniform(0, 1) * 10**-3

            # Find all the cube pieces that need to rotate together
            self.segment = self.get_face(self.rotating_axis, self.level)
            # Start from zero rotation
            self.rotation_angle = 0
            # Mark that we're now rotating
            self.is_rotating = True
        
        # Continue rotation if in progress
        # If we're currently in the middle of a rotation
        if self.is_rotating:
            # Check if we still need to rotate more
            if abs(self.rotation_angle - self.target_rotation) > 0.001:
                # Calculate rotation step
                # Figure out how much we still need to rotate
                diff = abs(self.target_rotation - self.rotation_angle)
                # Don't rotate too fast - limit the step size
                delta_angle = min(np.radians(2), diff)  # Increased speed slightly
                
                # Choose which direction to rotate
                if self.target_rotation > self.rotation_angle:
                    self.rotation_angle += delta_angle
                else:
                    self.rotation_angle -= delta_angle
                    delta_angle = -delta_angle
            else:
                # Rotation complete
                # We're done rotating
                delta_angle = 0
                self.is_rotating = False
                # Move to the next animation step if we're tracking steps
                if animation_step is not None:
                    animation_step += 1

            # Apply rotation to all cubes in the segment
            # Actually rotate all the cube pieces that need to move
            if self.rotating_axis is not None:
                # Figure out which axis we're rotating around
                nonzero_indices = np.atleast_1d(self.rotating_axis).nonzero()[0]
                
                # If we found a valid axis
                if len(nonzero_indices) > 0:
                    axis_index = nonzero_indices[0]
                    
                    # Rotate each cube piece in our segment
                    for cube_id in self.segment:
                        # Make sure this cube exists
                        if cube_id < len(self.cubes):
                            cube = self.cubes[cube_id]
                            # Rotate each part of this cube piece
                            for part in cube:
                                # Only rotate if we're actually moving
                                if abs(delta_angle) > 0.001:  # Only rotate if there's meaningful movement
                                    part.rotate(axis_index, delta_angle)
        
        # Return the updated queue and animation step
        return rotation_queue, animation_step

    def sort_face(self, segment, axis, level):
        # Define a function to calculate sorting order for each cube piece
        def key(idx):
            # Get the center position of this cube piece
            cx, cy, cz = self.cubes[idx][0].center

            # Choose how to sort based on which face we're looking at
            if axis == 2:  # Front or Back face
                row = cy   # Positive Y for row (bottom to top, then flip)
                col = cx if level == 2 else -cx  # Normal X for Front, flipped for Back
            elif axis == 0:  # Right or Left face
                row = cy   # Positive Y for row
                col = -cz if level == 2 else cz  # Flipped Z for Right, normal for Left
            elif axis == 1:  # Up or Down face  
                row = cz if level == 2 else -cz  # Normal Z for Up, flipped for Down
                col = cx   # Normal X
            # Return the sorting key (row first, then column)
            return (row, col)

        # Sort the cube pieces using our custom sorting function
        return sorted(segment, key=key)

    def add_rotation(self, rotation_queue, axis, level, clockwise=True):
        """Add a rotation to the queue"""
        # Decide how much to rotate (90 degrees clockwise or counter-clockwise)
        angle = np.pi/2 if clockwise else -np.pi/2
        # Add this rotation to our to-do list
        rotation_queue.append((angle, axis, level))
        # Return the updated to-do list
        return rotation_queue