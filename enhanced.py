import copy
from typing import List, Tuple, Dict, Set

class RubiksCube:
    """
    A complete Rubik's Cube representation with state tracking and solving algorithms.
    """
    
    def __init__(self, initial_state=None):
        """Initialize cube with solved state or provided state"""
        if initial_state is None:
            # Solved cube state
            self.state = {
                'U': [['yellow' for _ in range(3)] for _ in range(3)],
                'D': [['white' for _ in range(3)] for _ in range(3)],
                'F': [['green' for _ in range(3)] for _ in range(3)],
                'B': [['blue' for _ in range(3)] for _ in range(3)],
                'R': [['red' for _ in range(3)] for _ in range(3)],
                'L': [['orange' for _ in range(3)] for _ in range(3)]
            }
        else:
            self.state = copy.deepcopy(initial_state)
    
    def copy(self):
        """Create a deep copy of the cube"""
        return RubiksCube(self.state)
    
    def apply_move(self, move: str):
        """Apply a single move to the cube"""
        if move == "R":
            self._rotate_face('R')
            self._rotate_adjacent_r()
        elif move == "R'":
            self._rotate_face('R', counterclockwise=True)
            self._rotate_adjacent_r(counterclockwise=True)
        elif move == "R2":
            self.apply_move("R")
            self.apply_move("R")
        elif move == "L":
            self._rotate_face('L')
            self._rotate_adjacent_l()
        elif move == "L'":
            self._rotate_face('L', counterclockwise=True)
            self._rotate_adjacent_l(counterclockwise=True)
        elif move == "L2":
            self.apply_move("L")
            self.apply_move("L")
        elif move == "U":
            self._rotate_face('U')
            self._rotate_adjacent_u()
        elif move == "U'":
            self._rotate_face('U', counterclockwise=True)
            self._rotate_adjacent_u(counterclockwise=True)
        elif move == "U2":
            self.apply_move("U")
            self.apply_move("U")
        elif move == "D":
            self._rotate_face('D')
            self._rotate_adjacent_d()
        elif move == "D'":
            self._rotate_face('D', counterclockwise=True)
            self._rotate_adjacent_d(counterclockwise=True)
        elif move == "D2":
            self.apply_move("D")
            self.apply_move("D")
        elif move == "F":
            self._rotate_face('F')
            self._rotate_adjacent_f()
        elif move == "F'":
            self._rotate_face('F', counterclockwise=True)
            self._rotate_adjacent_f(counterclockwise=True)
        elif move == "F2":
            self.apply_move("F")
            self.apply_move("F")
        elif move == "B":
            self._rotate_face('B')
            self._rotate_adjacent_b()
        elif move == "B'":
            self._rotate_face('B', counterclockwise=True)
            self._rotate_adjacent_b(counterclockwise=True)
        elif move == "B2":
            self.apply_move("B")
            self.apply_move("B")
    
    def apply_moves(self, moves: List[str]):
        """Apply a sequence of moves"""
        for move in moves:
            if move.strip():  # Skip empty moves
                self.apply_move(move.strip())
    
    def _rotate_face(self, face: str, counterclockwise: bool = False):
        """Rotate a face 90 degrees"""
        f = self.state[face]
        if counterclockwise:
            # Counterclockwise rotation
            self.state[face] = [[f[j][2-i] for j in range(3)] for i in range(3)]
        else:
            # Clockwise rotation
            self.state[face] = [[f[2-j][i] for j in range(3)] for i in range(3)]
    
    def _rotate_adjacent_r(self, counterclockwise: bool = False):
        """Rotate adjacent edges for R move"""
        if counterclockwise:
            temp = [self.state['U'][i][2] for i in range(3)]
            for i in range(3):
                self.state['U'][i][2] = self.state['F'][i][2]
                self.state['F'][i][2] = self.state['D'][i][2]
                self.state['D'][i][2] = self.state['B'][2-i][0]
                self.state['B'][2-i][0] = temp[i]
        else:
            temp = [self.state['U'][i][2] for i in range(3)]
            for i in range(3):
                self.state['U'][i][2] = self.state['B'][2-i][0]
                self.state['B'][2-i][0] = self.state['D'][i][2]
                self.state['D'][i][2] = self.state['F'][i][2]
                self.state['F'][i][2] = temp[i]
    
    def _rotate_adjacent_l(self, counterclockwise: bool = False):
        """Rotate adjacent edges for L move"""
        if counterclockwise:
            temp = [self.state['U'][i][0] for i in range(3)]
            for i in range(3):
                self.state['U'][i][0] = self.state['B'][2-i][2]
                self.state['B'][2-i][2] = self.state['D'][i][0]
                self.state['D'][i][0] = self.state['F'][i][0]
                self.state['F'][i][0] = temp[i]
        else:
            temp = [self.state['U'][i][0] for i in range(3)]
            for i in range(3):
                self.state['U'][i][0] = self.state['F'][i][0]
                self.state['F'][i][0] = self.state['D'][i][0]
                self.state['D'][i][0] = self.state['B'][2-i][2]
                self.state['B'][2-i][2] = temp[i]
    
    def _rotate_adjacent_u(self, counterclockwise: bool = False):
        """Rotate adjacent edges for U move"""
        if counterclockwise:
            temp = self.state['F'][0][:]
            self.state['F'][0] = self.state['R'][0][:]
            self.state['R'][0] = self.state['B'][0][:]
            self.state['B'][0] = self.state['L'][0][:]
            self.state['L'][0] = temp
        else:
            temp = self.state['F'][0][:]
            self.state['F'][0] = self.state['L'][0][:]
            self.state['L'][0] = self.state['B'][0][:]
            self.state['B'][0] = self.state['R'][0][:]
            self.state['R'][0] = temp
    
    def _rotate_adjacent_d(self, counterclockwise: bool = False):
        """Rotate adjacent edges for D move"""
        if counterclockwise:
            temp = self.state['F'][2][:]
            self.state['F'][2] = self.state['L'][2][:]
            self.state['L'][2] = self.state['B'][2][:]
            self.state['B'][2] = self.state['R'][2][:]
            self.state['R'][2] = temp
        else:
            temp = self.state['F'][2][:]
            self.state['F'][2] = self.state['R'][2][:]
            self.state['R'][2] = self.state['B'][2][:]
            self.state['B'][2] = self.state['L'][2][:]
            self.state['L'][2] = temp
    
    def _rotate_adjacent_f(self, counterclockwise: bool = False):
        """Rotate adjacent edges for F move"""
        if counterclockwise:
            temp = [self.state['U'][2][i] for i in range(3)]
            for i in range(3):
                self.state['U'][2][i] = self.state['R'][2-i][0]
                self.state['R'][2-i][0] = self.state['D'][0][2-i]
                self.state['D'][0][2-i] = self.state['L'][i][2]
                self.state['L'][i][2] = temp[i]
        else:
            temp = [self.state['U'][2][i] for i in range(3)]
            for i in range(3):
                self.state['U'][2][i] = self.state['L'][2-i][2]
                self.state['L'][2-i][2] = self.state['D'][0][i]
                self.state['D'][0][i] = self.state['R'][i][0]
                self.state['R'][i][0] = temp[2-i]
    
    def _rotate_adjacent_b(self, counterclockwise: bool = False):
        """Rotate adjacent edges for B move"""
        if counterclockwise:
            temp = [self.state['U'][0][i] for i in range(3)]
            for i in range(3):
                self.state['U'][0][i] = self.state['L'][2-i][0]
                self.state['L'][2-i][0] = self.state['D'][2][i]
                self.state['D'][2][i] = self.state['R'][i][2]
                self.state['R'][i][2] = temp[2-i]
        else:
            temp = [self.state['U'][0][i] for i in range(3)]
            for i in range(3):
                self.state['U'][0][i] = self.state['R'][2-i][2]
                self.state['R'][2-i][2] = self.state['D'][2][2-i]
                self.state['D'][2][2-i] = self.state['L'][i][0]
                self.state['L'][i][0] = temp[i]

class CubeSolver:
    """Advanced Rubik's Cube solver with dynamic state analysis"""
    
    def __init__(self, cube: RubiksCube):
        self.cube = cube.copy()
        self.solution_moves = []
        self.max_moves_per_step = 50  # Prevent infinite loops
    
    def solve(self) -> List[str]:
        """Complete solving algorithm with validation"""
        self.solution_moves = []
        
        try:
            # Step 1: White Cross
            print("Solving white cross...")
            moves = self.solve_white_cross()
            if not self._validate_white_cross():
                print("Warning: White cross validation failed")
            self.solution_moves.extend(moves)
            
            # Step 2: White Corners  
            print("Solving white corners...")
            moves = self.solve_white_corners()
            if not self._validate_white_corners():
                print("Warning: White corners validation failed")
            self.solution_moves.extend(moves)
            
            # Step 3: Middle Layer
            print("Solving middle layer...")
            moves = self.solve_middle_layer()
            if not self._validate_middle_layer():
                print("Warning: Middle layer validation failed")
            self.solution_moves.extend(moves)
            
            # Step 4: Yellow Cross
            print("Solving yellow cross...")
            moves = self.solve_yellow_cross()
            if not self._validate_yellow_cross():
                print("Warning: Yellow cross validation failed")
            self.solution_moves.extend(moves)
            
            # Step 5: Yellow Corners
            print("Solving yellow corners...")
            moves = self.solve_yellow_corners()
            if not self._validate_yellow_corners():
                print("Warning: Yellow corners validation failed")
            self.solution_moves.extend(moves)
            
            # Step 6: Final Layer
            print("Solving final layer...")
            moves = self.solve_final_layer()
            self.solution_moves.extend(moves)
            
            # Final validation
            if self.is_solved():
                print("✓ Cube solved successfully!")
            else:
                print("✗ Cube solving failed - final validation error")
            
            # Optimize move sequence
            self.solution_moves = self._optimize_moves(self.solution_moves)
            
        except Exception as e:
            print(f"Error during solving: {e}")
            return []
        
        return self.solution_moves
    
    def _validate_white_cross(self) -> bool:
        """Validate that white cross is correctly formed"""
        d = self.cube.state['D']
        
        # Check white cross on D face
        if not (d[0][1] == 'white' and d[1][0] == 'white' and 
                d[1][2] == 'white' and d[2][1] == 'white'):
            return False
        
        # Check that side colors match centers
        side_checks = [
            (self.cube.state['F'][2][1], self.cube.state['F'][1][1]),
            (self.cube.state['R'][2][1], self.cube.state['R'][1][1]),
            (self.cube.state['B'][2][1], self.cube.state['B'][1][1]),
            (self.cube.state['L'][2][1], self.cube.state['L'][1][1])
        ]
        
        return all(edge_color == center_color for edge_color, center_color in side_checks)
    
    def _validate_white_corners(self) -> bool:
        """Validate that all white corners are correctly positioned"""
        # Check that all D face corners are white
        d = self.cube.state['D']
        if not (d[0][0] == 'white' and d[0][2] == 'white' and 
                d[2][0] == 'white' and d[2][2] == 'white'):
            return False
        
        # Check corner color alignment
        corner_checks = [
            # Front-right corner
            (self.cube.state['F'][2][2], self.cube.state['F'][1][1],
             self.cube.state['R'][2][0], self.cube.state['R'][1][1]),
            # Right-back corner
            (self.cube.state['R'][2][2], self.cube.state['R'][1][1],
             self.cube.state['B'][2][0], self.cube.state['B'][1][1]),
            # Back-left corner
            (self.cube.state['B'][2][2], self.cube.state['B'][1][1],
             self.cube.state['L'][2][2], self.cube.state['L'][1][1]),
            # Left-front corner
            (self.cube.state['L'][2][0], self.cube.state['L'][1][1],
             self.cube.state['F'][2][0], self.cube.state['F'][1][1])
        ]
        
        return all(c1 == center1 and c2 == center2 
                  for c1, center1, c2, center2 in corner_checks)
    
    def _validate_middle_layer(self) -> bool:
        """Validate that middle layer is correctly solved"""
        for face in ['F', 'R', 'B', 'L']:
            center_color = self.cube.state[face][1][1]
            
            # Check left and right middle edges
            if (self.cube.state[face][1][0] != center_color or
                self.cube.state[face][1][2] != center_color):
                return False
        
        return True
    
    def _validate_yellow_cross(self) -> bool:
        """Validate that yellow cross is formed on U face"""
        return self._is_yellow_cross_formed()
    
    def _validate_yellow_corners(self) -> bool:
        """Validate that all yellow corners are oriented correctly"""
        return self._all_yellow_corners_oriented()
    
    def _optimize_moves(self, moves: List[str]) -> List[str]:
        """Optimize move sequence by removing redundant moves"""
        if not moves:
            return moves
        
        optimized = []
        i = 0
        
        while i < len(moves):
            current = moves[i]
            count = 1
            
            # Count consecutive identical moves
            while (i + count < len(moves) and 
                   moves[i + count] == current):
                count += 1
            
            # Optimize based on count
            if count % 4 == 0:
                # 4 identical moves = no move
                pass
            elif count % 4 == 1:
                optimized.append(current)
            elif count % 4 == 2:
                optimized.append(current + "2")
            elif count % 4 == 3:
                optimized.append(current + "'")
            
            i += count
        
        # Remove opposite moves (R followed by R')
        further_optimized = []
        i = 0
        
        while i < len(optimized):
            if (i + 1 < len(optimized) and 
                self._are_opposite_moves(optimized[i], optimized[i + 1])):
                i += 2  # Skip both moves
            else:
                further_optimized.append(optimized[i])
                i += 1
        
        return further_optimized
    
    def _are_opposite_moves(self, move1: str, move2: str) -> bool:
        """Check if two moves are opposites (cancel each other out)"""
        if len(move1) == 1 and move2 == move1 + "'":
            return True
        if len(move2) == 1 and move1 == move2 + "'":
            return True
        if move1.endswith("'") and move2 == move1[:-1]:
            return True
        if move2.endswith("'") and move1 == move2[:-1]:
            return True
        return False
    
    def solve_white_cross(self) -> List[str]:
        """Solve white cross on bottom (D face)"""
        moves = []
        target_edges = [
            ('white', 'green'),
            ('white', 'red'), 
            ('white', 'blue'),
            ('white', 'orange')
        ]
        
        for target in target_edges:
            edge_moves = self._solve_white_edge(target)
            moves.extend(edge_moves)
            self.cube.apply_moves(edge_moves)
        
        return moves
    
    def _solve_white_edge(self, target_colors: Tuple[str, str]) -> List[str]:
        """Find and solve a specific white edge piece"""
        moves = []
        white, color = target_colors
        
        # Find the edge piece
        edge_pos = self._find_edge(white, color)
        if not edge_pos:
            return moves
        
        face1, pos1, face2, pos2 = edge_pos
        
        # If already in correct position, skip
        if self._is_white_edge_solved(color):
            return moves
        
        # Move edge to top layer if not already there
        if 'U' not in [face1, face2]:
            moves.extend(self._move_edge_to_top(face1, pos1, face2, pos2))
            self.cube.apply_moves(moves)
        
        # Orient and insert the edge
        moves.extend(self._insert_white_edge(color))
        
        return moves
    
    def _find_edge(self, color1: str, color2: str) -> Tuple[str, Tuple[int, int], str, Tuple[int, int]]:
        """Find an edge piece with the given colors"""
        edges = [
            # U face edges
            ('U', (0, 1), 'B', (0, 1)),
            ('U', (1, 0), 'L', (0, 1)),
            ('U', (1, 2), 'R', (0, 1)),
            ('U', (2, 1), 'F', (0, 1)),
            # D face edges  
            ('D', (0, 1), 'F', (2, 1)),
            ('D', (1, 0), 'L', (2, 1)),
            ('D', (1, 2), 'R', (2, 1)),
            ('D', (2, 1), 'B', (2, 1)),
            # Middle edges
            ('F', (1, 0), 'L', (1, 2)),
            ('F', (1, 2), 'R', (1, 0)),
            ('B', (1, 0), 'R', (1, 2)),
            ('B', (1, 2), 'L', (1, 0))
        ]
        
        for face1, pos1, face2, pos2 in edges:
            c1 = self.cube.state[face1][pos1[0]][pos1[1]]
            c2 = self.cube.state[face2][pos2[0]][pos2[1]]
            if {c1, c2} == {color1, color2}:
                return face1, pos1, face2, pos2
        
        return None
    
    def _is_white_edge_solved(self, color: str) -> bool:
        """Check if a white edge is in correct position"""
        face_map = {'green': 'F', 'red': 'R', 'blue': 'B', 'orange': 'L'}
        face = face_map[color]
        
        # Check if white is on D face and color matches center
        d_color = self.cube.state['D'][self._get_d_edge_pos(face)][1] if face == 'F' else \
                  self.cube.state['D'][1][self._get_d_edge_pos(face)] if face in ['L', 'R'] else \
                  self.cube.state['D'][self._get_d_edge_pos(face)][1]
        
        face_color = self.cube.state[face][2][1]
        
        return d_color == 'white' and face_color == color
    
    def _get_d_edge_pos(self, face: str) -> int:
        """Get D face position index for edge"""
        pos_map = {'F': 0, 'R': 1, 'B': 2, 'L': 1}
        return pos_map[face]
    
    def _move_edge_to_top(self, face1: str, pos1: Tuple[int, int], face2: str, pos2: Tuple[int, int]) -> List[str]:
        """Move an edge from middle/bottom to top layer"""
        moves = []
        
        if face1 == 'D' or face2 == 'D':
            # Edge is on bottom, bring to top
            bottom_face = face2 if face1 == 'D' else face1
            moves.extend([bottom_face + '2'])
        else:
            # Edge is in middle layer
            if face1 in ['F', 'R', 'B', 'L']:
                moves.extend([face1, 'U', face1 + "'"])
            else:
                moves.extend([face2, 'U', face2 + "'"])
        
        return moves
    
    def _insert_white_edge(self, color: str) -> List[str]:
        """Insert white edge from top to correct position"""
        moves = []
        face_map = {'green': 'F', 'red': 'R', 'blue': 'B', 'orange': 'L'}
        target_face = face_map[color]
        
        # Align edge above target position
        u_moves_map = {'F': 0, 'R': 1, 'B': 2, 'L': 3}
        
        # Find current position of edge in U layer
        current_pos = None
        for i, face in enumerate(['F', 'R', 'B', 'L']):
            u_color = self.cube.state['U'][2 if face == 'F' else (0 if face == 'B' else (1, 2 if face == 'R' else 0)[i % 2])][1 if face in ['F', 'B'] else (2 if face == 'R' else 0)]
            f_color = self.cube.state[face][0][1]
            
            if {u_color, f_color} == {'white', color}:
                current_pos = i
                break
        
        if current_pos is not None:
            target_pos = u_moves_map[target_face]
            u_moves = (target_pos - current_pos) % 4
            moves.extend(['U'] * u_moves)
            
            # Insert the edge
            moves.extend([target_face + '2'])
        
        return moves
    
    def solve_white_corners(self) -> List[str]:
        """Solve all white corners"""
        moves = []
        target_corners = [
            ('white', 'green', 'red'),
            ('white', 'red', 'blue'),
            ('white', 'blue', 'orange'),
            ('white', 'orange', 'green')
        ]
        
        for target in target_corners:
            corner_moves = self._solve_white_corner(target)
            moves.extend(corner_moves)
            self.cube.apply_moves(corner_moves)
        
        return moves
    
    def _solve_white_corner(self, target_colors: Tuple[str, str, str]) -> List[str]:
        """Solve a specific white corner"""
        moves = []
        
        # Find the corner
        corner_pos = self._find_corner(target_colors)
        if not corner_pos:
            return moves
        
        # If corner is in bottom layer but wrong position/orientation
        if self._corner_in_bottom(corner_pos):
            moves.extend(self._remove_corner_from_bottom(corner_pos))
            self.cube.apply_moves(moves)
        
        # Move corner to correct position and orient
        corner_moves = self._insert_white_corner(target_colors)
        moves.extend(corner_moves)
        
        return moves
    
    def _find_corner(self, colors: Tuple[str, str, str]) -> Tuple:
        """Find a corner with given colors"""
        corners = [
            # U layer corners
            ('U', (0, 0), 'L', (0, 0), 'B', (0, 2)),  # ULB
            ('U', (0, 2), 'B', (0, 0), 'R', (0, 2)),  # UBR  
            ('U', (2, 0), 'F', (0, 0), 'L', (0, 2)),  # UFL
            ('U', (2, 2), 'R', (0, 0), 'F', (0, 2)),  # URF
            # D layer corners
            ('D', (0, 0), 'L', (2, 0), 'F', (2, 2)),  # DLF
            ('D', (0, 2), 'F', (2, 0), 'R', (2, 2)),  # DFR
            ('D', (2, 0), 'B', (2, 0), 'L', (2, 2)),  # DBL
            ('D', (2, 2), 'R', (2, 0), 'B', (2, 2))   # DRB
        ]
        
        for corner in corners:
            corner_colors = set()
            for i in range(0, len(corner), 2):
                face, pos = corner[i], corner[i+1]
                corner_colors.add(self.cube.state[face][pos[0]][pos[1]])
            
            if corner_colors == set(colors):
                return corner
        
        return None
    
    def _corner_in_bottom(self, corner_pos: Tuple) -> bool:
        """Check if corner is in bottom layer"""
        return corner_pos[0] == 'D'
    
    def _remove_corner_from_bottom(self, corner_pos: Tuple) -> List[str]:
        """Remove corner from bottom layer to top"""
        # Use R U R' U' algorithm to move corner up
        return ["R", "U", "R'", "U'"]
    
    def _insert_white_corner(self, colors: Tuple[str, str, str]) -> List[str]:
        """Insert white corner from top layer"""
        moves = []
        
        # Simplified corner insertion - position corner above target slot
        # Then use repeated R U R' U' until white is on bottom
        max_attempts = 6
        for _ in range(max_attempts):
            # Check if corner is solved
            if self._is_white_corner_solved(colors):
                break
            
            # Apply corner algorithm
            moves.extend(["R", "U", "R'", "U'"])
            self.cube.apply_moves(["R", "U", "R'", "U'"])
        
        return moves
    
    def _is_white_corner_solved(self, colors: Tuple[str, str, str]) -> bool:
        """Check if white corner is correctly positioned"""
        white, color1, color2 = colors
        
        # Define corner positions based on colors
        face_map = {'green': 'F', 'red': 'R', 'blue': 'B', 'orange': 'L'}
        
        # Find the expected position
        if set([color1, color2]) == set(['green', 'red']):
            # Front-right corner
            d_pos = self.cube.state['D'][0][2]
            f_pos = self.cube.state['F'][2][2] 
            r_pos = self.cube.state['R'][2][0]
            return d_pos == 'white' and f_pos == 'green' and r_pos == 'red'
        elif set([color1, color2]) == set(['red', 'blue']):
            # Right-back corner
            d_pos = self.cube.state['D'][2][2]
            r_pos = self.cube.state['R'][2][2]
            b_pos = self.cube.state['B'][2][0]
            return d_pos == 'white' and r_pos == 'red' and b_pos == 'blue'
        elif set([color1, color2]) == set(['blue', 'orange']):
            # Back-left corner
            d_pos = self.cube.state['D'][2][0]
            b_pos = self.cube.state['B'][2][2]
            l_pos = self.cube.state['L'][2][2]
            return d_pos == 'white' and b_pos == 'blue' and l_pos == 'orange'
        elif set([color1, color2]) == set(['orange', 'green']):
            # Left-front corner
            d_pos = self.cube.state['D'][0][0]
            l_pos = self.cube.state['L'][2][0]
            f_pos = self.cube.state['F'][2][0]
            return d_pos == 'white' and l_pos == 'orange' and f_pos == 'green'
        
        return False
    
    def solve_middle_layer(self) -> List[str]:
        """Solve middle layer edges"""
        moves = []
        
        # Find and solve each middle layer edge
        target_edges = [
            ('green', 'red'), ('green', 'orange'),
            ('blue', 'red'), ('blue', 'orange')
        ]
        
        for edge in target_edges:
            if not self._is_middle_edge_solved(edge):
                edge_moves = self._solve_middle_edge(edge)
                moves.extend(edge_moves)
                self.cube.apply_moves(edge_moves)
        
        return moves
    
    def _is_middle_edge_solved(self, colors: Tuple[str, str]) -> bool:
        """Check if middle edge is correctly positioned"""
        color1, color2 = colors
        face_map = {'green': 'F', 'red': 'R', 'blue': 'B', 'orange': 'L'}
        
        face1, face2 = face_map[color1], face_map[color2]
        
        # Check all possible middle edge positions
        middle_edges = [
            ('F', (1, 0), 'L', (1, 2)),  # FL edge
            ('F', (1, 2), 'R', (1, 0)),  # FR edge  
            ('B', (1, 0), 'R', (1, 2)),  # BR edge
            ('B', (1, 2), 'L', (1, 0))   # BL edge
        ]
        
        for f1, pos1, f2, pos2 in middle_edges:
            c1 = self.cube.state[f1][pos1[0]][pos1[1]]
            c2 = self.cube.state[f2][pos2[0]][pos2[1]]
            
            if {c1, c2} == {color1, color2}:
                # Check if colors match their face centers
                center1 = self.cube.state[f1][1][1]
                center2 = self.cube.state[f2][1][1] 
                return c1 == center1 and c2 == center2
        
        return False
    
    def _solve_middle_edge(self, colors: Tuple[str, str]) -> List[str]:
        """Solve a specific middle layer edge"""
        moves = []
        
        # Find edge in top layer
        edge_pos = self._find_edge(colors[0], colors[1])
        if not edge_pos:
            return moves
        
        # Use appropriate algorithm to insert edge
        if self._edge_goes_right(colors):
            moves.extend(["U", "R", "U'", "R'", "U'", "F'", "U", "F"])
        else:
            moves.extend(["U'", "L'", "U", "L", "U", "F", "U'", "F'"])
        
        return moves
    
    def _edge_goes_right(self, colors: Tuple[str, str]) -> bool:
        """Determine if edge should be inserted to the right"""
        color1, color2 = colors
        
        # Find edge position in top layer
        edge_pos = self._find_edge(color1, color2)
        if not edge_pos:
            return True
        
        face1, pos1, face2, pos2 = edge_pos
        
        # Determine which color is on U face and which is on side
        if face1 == 'U':
            u_color = self.cube.state[face1][pos1[0]][pos1[1]]
            side_color = self.cube.state[face2][pos2[0]][pos2[1]]
            side_face = face2
        else:
            u_color = self.cube.state[face2][pos2[0]][pos2[1]]
            side_color = self.cube.state[face1][pos1[0]][pos1[1]]
            side_face = face1
        
        # Check if the side color matches the face center it's on
        face_center = self.cube.state[side_face][1][1]
        
        if side_color == face_center:
            # Need to go right if the other color's face is to the right
            other_color = u_color
            right_faces = {'F': 'R', 'R': 'B', 'B': 'L', 'L': 'F'}
            target_face = None
            
            for face in ['F', 'R', 'B', 'L']:
                if self.cube.state[face][1][1] == other_color:
                    target_face = face
                    break
            
            return target_face == right_faces.get(side_face, 'R')
        
        return True
    
    def solve_yellow_cross(self) -> List[str]:
        """Form yellow cross on top face"""
        moves = []
        algorithm = ["F", "R", "U", "R'", "U'", "F'"]
        
        max_attempts = 4
        for _ in range(max_attempts):
            if self._is_yellow_cross_formed():
                break
            
            # Apply OLL algorithm for cross
            moves.extend(algorithm)
            self.cube.apply_moves(algorithm)
        
        return moves
    
    def _is_yellow_cross_formed(self) -> bool:
        """Check if yellow cross is formed on U face"""
        u = self.cube.state['U']
        return (u[0][1] == 'yellow' and u[1][0] == 'yellow' and 
                u[1][2] == 'yellow' and u[2][1] == 'yellow')
    
    def solve_yellow_corners(self) -> List[str]:
        """Orient all yellow corners"""
        moves = []
        algorithm = ["R", "U", "R'", "U", "R", "U2", "R'"]
        
        max_attempts = 8
        for _ in range(max_attempts):
            if self._all_yellow_corners_oriented():
                break
            
            moves.extend(algorithm)
            self.cube.apply_moves(algorithm)
            
            # Rotate U to bring next corner into position
            moves.append("U")
            self.cube.apply_move("U")
        
        return moves
    
    def _all_yellow_corners_oriented(self) -> bool:
        """Check if all corners have yellow on top"""
        u = self.cube.state['U']
        return (u[0][0] == 'yellow' and u[0][2] == 'yellow' and 
                u[2][0] == 'yellow' and u[2][2] == 'yellow')
    
    def solve_final_layer(self) -> List[str]:
        """Permute final layer pieces"""
        moves = []
        
        # Corner permutation
        corner_moves = self._permute_corners()
        moves.extend(corner_moves)
        self.cube.apply_moves(corner_moves)
        
        # Edge permutation  
        edge_moves = self._permute_edges()
        moves.extend(edge_moves)
        self.cube.apply_moves(edge_moves)
        
        return moves
    
    def _permute_corners(self) -> List[str]:
        """Permute yellow corners into correct positions"""
        moves = []
        algorithm = ["U", "R", "U'", "L'", "U", "R'", "U'", "L"]
        
        max_attempts = 5
        for _ in range(max_attempts):
            if self._corners_permuted():
                break
            
            moves.extend(algorithm)
            self.cube.apply_moves(algorithm)
        
        return moves
    
    def _permute_edges(self) -> List[str]:  
        """Permute yellow edges into correct positions"""
        moves = []
        algorithm = ["F2", "U", "L", "R'", "F2", "L'", "R", "U", "F2"]
        
        max_attempts = 4
        for _ in range(max_attempts):
            if self._edges_permuted():
                break
            
            moves.extend(algorithm)
            self.cube.apply_moves(algorithm)
        
        return moves
    
    def _corners_permuted(self) -> bool:
        """Check if corners are in correct positions"""
        # Check each corner position
        corners_to_check = [
            # (D_pos, F_pos, R_pos) for front-right corner
            ((0, 2), 'F', (2, 2), 'R', (2, 0)),
            # (D_pos, R_pos, B_pos) for right-back corner  
            ((2, 2), 'R', (2, 2), 'B', (2, 0)),
            # (D_pos, B_pos, L_pos) for back-left corner
            ((2, 0), 'B', (2, 2), 'L', (2, 2)),
            # (D_pos, L_pos, F_pos) for left-front corner
            ((0, 0), 'L', (2, 0), 'F', (2, 0))
        ]
        
        for d_pos, f1, pos1, f2, pos2 in corners_to_check:
            d_color = self.cube.state['D'][d_pos[0]][d_pos[1]]
            f1_color = self.cube.state[f1][pos1[0]][pos1[1]]
            f2_color = self.cube.state[f2][pos2[0]][pos2[1]]
            
            # Check if colors match their respective face centers
            if not (d_color == 'white' and 
                   f1_color == self.cube.state[f1][1][1] and 
                   f2_color == self.cube.state[f2][1][1]):
                return False
        
        return True
    
    def _edges_permuted(self) -> bool:
        """Check if edges are in correct positions"""
        # Check if each face has its center color on all edge positions
        faces_to_check = ['F', 'R', 'B', 'L']
        
        for face in faces_to_check:
            center_color = self.cube.state[face][1][1]
            
            # Check top edge of face
            if self.cube.state[face][0][1] != center_color:
                return False
            
            # Check side edges (left and right)
            if self.cube.state[face][1][0] != center_color:
                return False
            if self.cube.state[face][1][2] != center_color:
                return False
            
            # Check bottom edge
            if self.cube.state[face][2][1] != center_color:
                return False
        
        return True
    
    def is_solved(self) -> bool:
        """Check if cube is completely solved"""
        for face in ['U', 'D', 'F', 'B', 'R', 'L']:
            center_color = self.cube.state[face][1][1]
            for row in self.cube.state[face]:
                for color in row:
                    if color != center_color:
                        return False
        return True


# Usage example:
def solve_cube(initial_state):
    """
    Main function to solve a Rubik's cube
    
    Args:
        initial_state: Dictionary representing the cube state
        
    Returns:
        List of moves to solve the cube
    """
    cube = RubiksCube(initial_state)
    solver = CubeSolver(cube)
    
    try:
        solution = solver.solve()
        return solution
    except Exception as e:
        print(f"Error solving cube: {e}")
        return []


# Example usage:
if __name__ == "__main__":
    # Example scrambled state (you would replace this with actual cube state)
    scrambled_state = {
        'U': [['red', 'white', 'blue'], 
              ['green', 'yellow', 'orange'], 
              ['yellow', 'red', 'white']],
        'D': [['orange', 'blue', 'yellow'], 
              ['red', 'white', 'green'], 
              ['white', 'orange', 'blue']],
        'F': [['yellow', 'green', 'red'], 
              ['white', 'green', 'blue'], 
              ['orange', 'yellow', 'red']],
        'B': [['white', 'orange', 'green'], 
              ['yellow', 'blue', 'red'], 
              ['blue', 'white', 'orange']],
        'R': [['green', 'red', 'yellow'], 
              ['orange', 'red', 'white'], 
              ['blue', 'green', 'yellow']],
        'L': [['red', 'blue', 'white'], 
              ['yellow', 'orange', 'green'], 
              ['orange', 'red', 'blue']]
    }
    
    # Solve the cube
    solution = solve_cube(scrambled_state)
    print(f"Solution moves: {solution}")
    print(f"Total moves: {len(solution)}")
    
    # Verify solution
    test_cube = RubiksCube(scrambled_state)
    test_cube.apply_moves(solution)
    solver_test = CubeSolver(test_cube)
    print(f"Cube solved: {solver_test.is_solved()}")