import pyray as pr
import numpy as np


class Cube:
    def __init__(self, size, center, face_color):
        self.size = size
        self.center = center
        self.face_color = face_color
        self.orientation = np.eye(3)

        self.model = None
        self.gen_meshe(size)
        self.create_model()
    

    def gen_meshe(self, scale:tuple):
        self.mesh = pr.gen_mesh_cube(*scale)
    
    def create_model(self):
        self.model = pr.load_model_from_mesh(self.mesh)
        self.model.materials[0].maps[pr.MATERIAL_MAP_DIFFUSE].color = self.face_color
        self.model.transform = pr.matrix_translate(self.center[0], self.center[1], self.center[2])


class Rubik:
    def __init__(self) -> None:
        self.cubes = []
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
                        colors[0] if x == 2 else pr.BLACK,
                        colors[1] if x == 0 else pr.BLACK,
                        colors[2] if y == 2 else pr.BLACK,
                        colors[3] if y == 0 else pr.BLACK,
                        colors[4] if z == 2 else pr.BLACK,
                        colors[5] if z == 0 else pr.BLACK,
                    ]

                    center_pos = np.array([(x - 1) * offset, (y - 1) * offset, (z - 1) * offset])
                    center = Cube((size, size, size), center_pos, pr.BLACK)

                    front_pos  = center_pos + np.array([0, 0, size/2 + sticker_offset])
                    back_pos   = center_pos + np.array([0, 0, -size/2 - sticker_offset])
                    right_pos  = center_pos + np.array([size/2 + sticker_offset, 0, 0])
                    left_pos   = center_pos + np.array([-size/2 - sticker_offset, 0, 0])
                    top_pos    = center_pos + np.array([0, size/2 + sticker_offset, 0])
                    bottom_pos = center_pos + np.array([0, -size/2 - sticker_offset, 0])

                    front  = Cube(size_z, front_pos,  face_colors[4])
                    back   = Cube(size_z, back_pos,   face_colors[5])
                    right  = Cube(size_y, right_pos,  face_colors[0])
                    left   = Cube(size_y, left_pos,   face_colors[1])
                    top    = Cube(size_x, top_pos,    face_colors[2])
                    bottom = Cube(size_x, bottom_pos, face_colors[3])

                    self.cubes.append([center, front, back, right, left, top, bottom])

        return self.cubes
            
        