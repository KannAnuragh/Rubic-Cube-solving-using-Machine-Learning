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
        self.model.transform = pr.matrix_translate(self.center[0], self.center[1], self.center[2])


class Rubik:
    def __init__(self) -> None:
        self.generate_rubik(2)

    def generate_rubik(self, size):
        x,y,z = 0,0,0
        piece = Cube((size, size, size), [x,y,z], pr.BLACK)
        return piece