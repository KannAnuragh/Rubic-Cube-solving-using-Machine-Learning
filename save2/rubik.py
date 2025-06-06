import raylibpy as pr
import numpy as np
import random

class Cube:
    def __init__(self, size, center, face_color):
        self.size = size
        self.center = np.array(center, dtype=float)
        self.face_color = face_color
        self.orientation = np.eye(3)
        self.model = None
        self.gen_mesh(size)
        self.create_model()
    
    def gen_mesh(self, scale):
        if isinstance(scale, tuple):
            self.mesh = pr.gen_mesh_cube(*scale)
        else:
            self.mesh = pr.gen_mesh_cube(scale, scale, scale)
    
    def create_model(self):
        self.model = pr.load_model_from_mesh(self.mesh)
        self.model.materials[0].maps[pr.MATERIAL_MAP_DIFFUSE].color = self.face_color
        self.update_transform()
    
    def update_transform(self):
        """Update the model's transformation matrix"""
        translation = pr.matrix_translate(self.center[0], self.center[1], self.center[2])
        
        # Convert orientation matrix to raylib format
        axis, angle = self.get_rotation_axis_angle()
        if angle > 0.001:  # Only apply rotation if there's a meaningful angle
            rotation = pr.matrix_rotate(axis, angle)  # angle is already in radians from get_rotation_axis_angle
            transform = pr.matrix_multiply(rotation, translation)
        else:
            transform = translation
            
        self.model.transform = transform

    def rotate(self, axis, theta):
        """Rotate the cube around the specified axis by theta radians"""
        if axis == 0:  # X-axis
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif axis == 1:  # Y-axis
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 2:  # Z-axis
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid axis")
        
        # Rotate the center position
        self.center = rotation_matrix @ self.center
        # Update orientation
        self.orientation = rotation_matrix @ self.orientation
        # Update the model transform
        self.update_transform()

    def get_rotation_axis_angle(self):
        """Extract rotation axis and angle from orientation matrix"""
        trace = np.trace(self.orientation)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle < 0.001:  # No rotation
            axis = pr.Vector3(0, 0, 1)  # Default axis
            return axis, 0
        
        # Extract axis
        rx = self.orientation[2, 1] - self.orientation[1, 2]
        ry = self.orientation[0, 2] - self.orientation[2, 0]
        rz = self.orientation[1, 0] - self.orientation[0, 1]
        
        axis_vec = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        axis = pr.Vector3(axis_vec[0], axis_vec[1], axis_vec[2])
        
        return axis, angle  # Return angle in radians

class Rubik:
    def __init__(self):
        self.cubes = []
        self.is_rotating = False
        self.rotation_angle = 0
        self.rotating_axis = None
        self.level = None
        self.segment = None
        self.target_rotation = 0
        self.generate_rubik(2)

    def generate_rubik(self, size):
        colors = [pr.RED, pr.ORANGE, pr.WHITE, pr.YELLOW, pr.GREEN, pr.BLUE]
        offset = size + 0.05
        face_thickness = 0.05
        sticker_offset = 0.01

        size_x = (size, face_thickness, size)
        size_y = (face_thickness, size, size)
        size_z = (size, size, face_thickness)

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    face_colors = [
                        colors[0] if x == 2 else pr.BLACK,  # Right face
                        colors[1] if x == 0 else pr.BLACK,  # Left face
                        colors[2] if y == 2 else pr.BLACK,  # Top face
                        colors[3] if y == 0 else pr.BLACK,  # Bottom face
                        colors[4] if z == 2 else pr.BLACK,  # Front face
                        colors[5] if z == 0 else pr.BLACK,  # Back face
                    ]

                    center_pos = np.array([(x - 1) * offset, (y - 1) * offset, (z - 1) * offset])
                    center = Cube((size, size, size), center_pos, pr.BLACK)

                    # Create face stickers
                    front_pos = center_pos + np.array([0, 0, size/2 + sticker_offset])
                    back_pos = center_pos + np.array([0, 0, -size/2 - sticker_offset])
                    right_pos = center_pos + np.array([size/2 + sticker_offset, 0, 0])
                    left_pos = center_pos + np.array([-size/2 - sticker_offset, 0, 0])
                    top_pos = center_pos + np.array([0, size/2 + sticker_offset, 0])
                    bottom_pos = center_pos + np.array([0, -size/2 - sticker_offset, 0])

                    front = Cube(size_z, front_pos, face_colors[4])
                    back = Cube(size_z, back_pos, face_colors[5])
                    right = Cube(size_y, right_pos, face_colors[0])
                    left = Cube(size_y, left_pos, face_colors[1])
                    top = Cube(size_x, top_pos, face_colors[2])
                    bottom = Cube(size_x, bottom_pos, face_colors[3])

                    self.cubes.append([center, front, back, right, left, top, bottom])

        return self.cubes

    def choose_piece(self, piece, axis_index, level):
        """Determine if a piece belongs to the specified level on the given axis"""
        coord = round(piece[0].center[axis_index], 1)
        if level == 0 and coord < 0:
            return True
        elif level == 1 and abs(coord) < 0.1:  # Middle layer
            return True
        elif level == 2 and coord > 0:
            return True
        return False
    
    def get_face(self, axis, level):
        """Get all pieces that belong to a specific face/layer"""
        axis_index = np.nonzero(axis)[0][0]
        segment = [i for i, cube in enumerate(self.cubes)
                   if self.choose_piece(cube, axis_index, level)]
        return segment

    def handle_rotation(self, rotation_queue, animation_step=None):
        """Handle the rotation animation"""
        # Start new rotation if queue has items and not currently rotating
        if rotation_queue and not self.is_rotating:
            self.target_rotation, self.rotating_axis, self.level = rotation_queue.pop(0)
            
            # Add small random offset to avoid floating point issues
            if self.target_rotation > 0:
                self.target_rotation += random.uniform(0, 1) * 10**-3
            else:
                self.target_rotation -= random.uniform(0, 1) * 10**-3

            self.segment = self.get_face(self.rotating_axis, self.level)
            self.rotation_angle = 0
            self.is_rotating = True
        
        # Continue rotation if in progress
        if self.is_rotating:
            if abs(self.rotation_angle - self.target_rotation) > 0.001:
                # Calculate rotation step
                diff = abs(self.target_rotation - self.rotation_angle)
                delta_angle = min(np.radians(2), diff)  # Increased speed slightly
                
                if self.target_rotation > self.rotation_angle:
                    self.rotation_angle += delta_angle
                else:
                    self.rotation_angle -= delta_angle
                    delta_angle = -delta_angle
            else:
                # Rotation complete
                delta_angle = 0
                self.is_rotating = False
                if animation_step is not None:
                    animation_step += 1

            # Apply rotation to all cubes in the segment
            if self.rotating_axis is not None:
                nonzero_indices = np.atleast_1d(self.rotating_axis).nonzero()[0]
                
                if len(nonzero_indices) > 0:
                    axis_index = nonzero_indices[0]
                    
                    for cube_id in self.segment:
                        if cube_id < len(self.cubes):
                            cube = self.cubes[cube_id]
                            for part in cube:
                                if abs(delta_angle) > 0.001:  # Only rotate if there's meaningful movement
                                    part.rotate(axis_index, delta_angle)
        
        return rotation_queue, animation_step

    def add_rotation(self, rotation_queue, axis, level, clockwise=True):
        """Add a rotation to the queue"""
        angle = np.pi/2 if clockwise else -np.pi/2
        rotation_queue.append((angle, axis, level))
        return rotation_queue
    
