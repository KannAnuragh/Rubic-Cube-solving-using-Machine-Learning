import numpy as np
import time
from collections import deque
import json
import os
from typing import List, Dict, Tuple, Optional

class RubiksCube:
    """
    Represents a 3x3x3 Rubik's cube with 6 faces, each having 9 stickers.
    Faces: 0=Top(White), 1=Left(Orange), 2=Front(Green), 3=Right(Red), 4=Back(Blue), 5=Bottom(Yellow)
    """
    
    def __init__(self):
        # Initialize solved cube state
        self.cube = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Top (White)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Left (Orange)
            [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Front (Green)
            [3, 3, 3, 3, 3, 3, 3, 3, 3],  # Right (Red)
            [4, 4, 4, 4, 4, 4, 4, 4, 4],  # Back (Blue)
            [5, 5, 5, 5, 5, 5, 5, 5, 5]   # Bottom (Yellow)
        ], dtype=np.int8)
        
        # Color mapping for display
        self.colors = {0: 'W', 1: 'O', 2: 'G', 3: 'R', 4: 'B', 5: 'Y'}
        self.color_names = {0: 'White', 1: 'Orange', 2: 'Green', 3: 'Red', 4: 'Blue', 5: 'Yellow'}
    
    def copy(self):
        """Create a deep copy of the cube"""
        new_cube = RubiksCube()
        new_cube.cube = self.cube.copy()
        return new_cube
    
    def apply_move(self, move: str):
        """Apply a single move to the cube"""
        if move == 'R':
            self._rotate_R()
        elif move == "R'":
            self._rotate_R_prime()
        elif move == 'R2':
            self._rotate_R()
            self._rotate_R()
        elif move == 'L':
            self._rotate_L()
        elif move == "L'":
            self._rotate_L_prime()
        elif move == 'L2':
            self._rotate_L()
            self._rotate_L()
        elif move == 'U':
            self._rotate_U()
        elif move == "U'":
            self._rotate_U_prime()
        elif move == 'U2':
            self._rotate_U()
            self._rotate_U()
        elif move == 'D':
            self._rotate_D()
        elif move == "D'":
            self._rotate_D_prime()
        elif move == 'D2':
            self._rotate_D()
            self._rotate_D()
        elif move == 'F':
            self._rotate_F()
        elif move == "F'":
            self._rotate_F_prime()
        elif move == 'F2':
            self._rotate_F()
            self._rotate_F()
        elif move == 'B':
            self._rotate_B()
        elif move == "B'":
            self._rotate_B_prime()
        elif move == 'B2':
            self._rotate_B()
            self._rotate_B()
    
    def apply_moves(self, moves: str):
        """Apply a sequence of moves"""
        if not moves.strip():
            return
        move_list = moves.strip().split()
        for move in move_list:
            if move:
                self.apply_move(move)
    
    def _rotate_face_clockwise(self, face: int):
        """Rotate a face 90 degrees clockwise"""
        temp = self.cube[face].copy()
        self.cube[face] = np.array([
            temp[6], temp[3], temp[0],
            temp[7], temp[4], temp[1],
            temp[8], temp[5], temp[2]
        ])
    
    def _rotate_R(self):
        """Right face clockwise"""
        self._rotate_face_clockwise(3)
        temp = [self.cube[0][2], self.cube[0][5], self.cube[0][8]]
        self.cube[0][2], self.cube[0][5], self.cube[0][8] = self.cube[2][2], self.cube[2][5], self.cube[2][8]
        self.cube[2][2], self.cube[2][5], self.cube[2][8] = self.cube[5][2], self.cube[5][5], self.cube[5][8]
        self.cube[5][2], self.cube[5][5], self.cube[5][8] = self.cube[4][6], self.cube[4][3], self.cube[4][0]
        self.cube[4][6], self.cube[4][3], self.cube[4][0] = temp[0], temp[1], temp[2]
    
    def _rotate_R_prime(self):
        """Right face counterclockwise"""
        for _ in range(3):
            self._rotate_R()
    
    def _rotate_L(self):
        """Left face clockwise"""
        self._rotate_face_clockwise(1)
        temp = [self.cube[0][0], self.cube[0][3], self.cube[0][6]]
        self.cube[0][0], self.cube[0][3], self.cube[0][6] = self.cube[4][8], self.cube[4][5], self.cube[4][2]
        self.cube[4][8], self.cube[4][5], self.cube[4][2] = self.cube[5][0], self.cube[5][3], self.cube[5][6]
        self.cube[5][0], self.cube[5][3], self.cube[5][6] = self.cube[2][0], self.cube[2][3], self.cube[2][6]
        self.cube[2][0], self.cube[2][3], self.cube[2][6] = temp[0], temp[1], temp[2]
    
    def _rotate_L_prime(self):
        """Left face counterclockwise"""
        for _ in range(3):
            self._rotate_L()
    
    def _rotate_U(self):
        """Up face clockwise"""
        self._rotate_face_clockwise(0)
        temp = [self.cube[2][0], self.cube[2][1], self.cube[2][2]]
        self.cube[2][0], self.cube[2][1], self.cube[2][2] = self.cube[3][0], self.cube[3][1], self.cube[3][2]
        self.cube[3][0], self.cube[3][1], self.cube[3][2] = self.cube[4][0], self.cube[4][1], self.cube[4][2]
        self.cube[4][0], self.cube[4][1], self.cube[4][2] = self.cube[1][0], self.cube[1][1], self.cube[1][2]
        self.cube[1][0], self.cube[1][1], self.cube[1][2] = temp[0], temp[1], temp[2]
    
    def _rotate_U_prime(self):
        """Up face counterclockwise"""
        for _ in range(3):
            self._rotate_U()
    
    def _rotate_D(self):
        """Down face clockwise"""
        self._rotate_face_clockwise(5)
        temp = [self.cube[2][6], self.cube[2][7], self.cube[2][8]]
        self.cube[2][6], self.cube[2][7], self.cube[2][8] = self.cube[1][6], self.cube[1][7], self.cube[1][8]
        self.cube[1][6], self.cube[1][7], self.cube[1][8] = self.cube[4][6], self.cube[4][7], self.cube[4][8]
        self.cube[4][6], self.cube[4][7], self.cube[4][8] = self.cube[3][6], self.cube[3][7], self.cube[3][8]
        self.cube[3][6], self.cube[3][7], self.cube[3][8] = temp[0], temp[1], temp[2]
    
    def _rotate_D_prime(self):
        """Down face counterclockwise"""
        for _ in range(3):
            self._rotate_D()
    
    def _rotate_F(self):
        """Front face clockwise"""
        self._rotate_face_clockwise(2)
        temp = [self.cube[0][6], self.cube[0][7], self.cube[0][8]]
        self.cube[0][6], self.cube[0][7], self.cube[0][8] = self.cube[1][8], self.cube[1][5], self.cube[1][2]
        self.cube[1][8], self.cube[1][5], self.cube[1][2] = self.cube[5][2], self.cube[5][1], self.cube[5][0]
        self.cube[5][2], self.cube[5][1], self.cube[5][0] = self.cube[3][0], self.cube[3][3], self.cube[3][6]
        self.cube[3][0], self.cube[3][3], self.cube[3][6] = temp[0], temp[1], temp[2]
    
    def _rotate_F_prime(self):
        """Front face counterclockwise"""
        for _ in range(3):
            self._rotate_F()
    
    def _rotate_B(self):
        """Back face clockwise"""
        self._rotate_face_clockwise(4)
        temp = [self.cube[0][0], self.cube[0][1], self.cube[0][2]]
        self.cube[0][0], self.cube[0][1], self.cube[0][2] = self.cube[3][2], self.cube[3][5], self.cube[3][8]
        self.cube[3][2], self.cube[3][5], self.cube[3][8] = self.cube[5][8], self.cube[5][7], self.cube[5][6]
        self.cube[5][8], self.cube[5][7], self.cube[5][6] = self.cube[1][6], self.cube[1][3], self.cube[1][0]
        self.cube[1][6], self.cube[1][3], self.cube[1][0] = temp[0], temp[1], temp[2]
    
    def _rotate_B_prime(self):
        """Back face counterclockwise"""
        for _ in range(3):
            self._rotate_B()
    
    def is_solved(self) -> bool:
        """Check if the cube is solved"""
        for face in range(6):
            if not all(self.cube[face] == face):
                return False
        return True
    
    def print_cube(self):
        """Print a visual representation of the cube"""
        print("\nCube state:")
        print("    " + " ".join([self.colors[x] for x in self.cube[0][:3]]))
        print("    " + " ".join([self.colors[x] for x in self.cube[0][3:6]]))
        print("    " + " ".join([self.colors[x] for x in self.cube[0][6:9]]))
        print()
        
        for i in range(3):
            row = ""
            for face in [1, 2, 3, 4]:  # Left, Front, Right, Back
                row += " ".join([self.colors[x] for x in self.cube[face][i*3:(i+1)*3]]) + " "
            print(row)
        print()
        
        print("    " + " ".join([self.colors[x] for x in self.cube[5][:3]]))
        print("    " + " ".join([self.colors[x] for x in self.cube[5][3:6]]))
        print("    " + " ".join([self.colors[x] for x in self.cube[5][6:9]]))
        print()
    

    def test_moves(self, moves: str):
        """
        Test a sequence of moves and show the result
        Usage: test_moves("L D Bi Fi")
        """
        print(f"Testing moves: {moves}")
        print("Initial state:")
        self.print_cube()
        
        # Parse moves (handle 'i' suffix for inverse moves)
        move_list = moves.strip().split()
        parsed_moves = []
        
        for move in move_list:
            if move.endswith('i'):
                # Convert 'Bi' to 'B''
                base_move = move[:-1]
                parsed_moves.append(base_move + "'")
            else:
                parsed_moves.append(move)
        
        # Apply the moves
        for move in parsed_moves:
            print(f"Applying: {move}")
            self.apply_move(move)
        
        print(f"\nFinal state after moves: {' '.join(parsed_moves)}")
        self.print_cube()
        print(f"Is solved: {self.is_solved()}")

    

class ThistlethwaiteSolver:
    """
    Thistlethwaite algorithm implementation for solving Rubik's cube
    """
    
    def __init__(self, load_tables=True):
        self.stage_moves = {
            0: ["R", "R'", "R2", "L", "L'", "L2", "U", "U'", "U2", 
                "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"],
            1: ["R", "R'", "R2", "L", "L'", "L2", "U", "U'", "U2", "D", "D'", "D2"],
            2: ["R2", "L2", "U", "U'", "U2", "D", "D'", "D2"],
            3: ["U", "U'", "U2", "D", "D'", "D2"]
        }
        
        # Lookup tables for each stage
        self.lookup_tables = [{}, {}, {}, {}]
        self.max_depth = [50, 50, 50, 50]  # Maximum search depth for each stage
        
        # Load lookup tables from files if requested
        if load_tables:
            self._load_lookup_tables()
    
    def _load_lookup_tables(self):
        """Load lookup tables from stage files"""
        import os

        self.lookup_tables = [{} for _ in range(4)]  # ensure list of dicts exists
        stage_files = ["stage0.txt", "stage1.txt", "stage2.txt", "stage3.txt"]

        for stage, filename in enumerate(stage_files):
            if os.path.exists(filename):
                print(f"Loading {filename}...")
                try:
                    with open(filename, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue

                            # First 7 chars are always the hash key (keep as string)
                            hash_key = line[:7].strip()
                            move_text = line[8:].strip()

                            # Remove markers like NP, ., etc.
                            move_text = self._parse_moves(move_text)

                            if move_text:
                                self.lookup_tables[stage][hash_key] = move_text

                    print(f"Loaded {len(self.lookup_tables[stage])} entries from {filename}")

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: {filename} not found, using BFS for stage {stage + 1}")

    def _parse_moves(self, moves_str: str) -> str:
        """
        Parse moves from file format to standard notation
        Converts: iU -> U', 2D -> D2, etc.
        """
        if not moves_str:
            return ""
        
        # Remove markers
        moves_str = moves_str.replace("NP", "").replace(".", "").strip()
        
        if not moves_str:
            return ""
        
        parts = moves_str.split()
        parsed_moves = []
        
        for move in parts:
            if not move:
                continue
            
            # Handle inverse moves (i prefix)
            if move.startswith('i'):
                base_move = move[1:]
                if base_move in ['U', 'D', 'R', 'L', 'F', 'B']:
                    parsed_moves.append(base_move + "'")
                continue
            
            # Handle double moves (2 prefix)  
            if move.startswith('2'):
                base_move = move[1:]
                if base_move in ['U', 'D', 'R', 'L', 'F', 'B']:
                    parsed_moves.append(base_move + "2")
                continue
            
            # Handle regular moves
            if move in ['U', 'D', 'R', 'L', 'F', 'B']:
                parsed_moves.append(move)
        
        return " ".join(parsed_moves)
    
    def solve(self, cube: RubiksCube) -> str:
        """
        Solve the cube using Thistlethwaite algorithm
        Returns the solution as a string of moves
        """
        if cube.is_solved():
            return ""
        
        print("Starting Thistlethwaite solve...")
        solution = []
        
        for stage in range(4):
            print(f"Stage {stage + 1}/4...")
            stage_solution = self._solve_stage(cube, stage)
            
            if stage_solution is None:
                print(f"Failed to solve stage {stage + 1}")
                return None
            
            solution.extend(stage_solution)
            cube.apply_moves(" ".join(stage_solution))
            
            print(f"Stage {stage + 1} completed with {len(stage_solution)} moves")
        
        final_solution = " ".join(solution)
        print(f"Cube solved! Total moves: {len(solution)}")
        return self._optimize_moves(final_solution)
    
    def _solve_stage(self, cube: RubiksCube, stage: int) -> Optional[List[str]]:
        """
        Solve a specific stage using lookup table or breadth-first search
        """
        if self._is_stage_complete(cube, stage):
            return []
        
        # First try lookup table if available
        if self.lookup_tables[stage]:
            cube_hash = self._hash_cube_for_stage(cube, stage)
            print(f"[DEBUG] Stage {stage} hash: {cube_hash}")

            if cube_hash in self.lookup_tables[stage]:
                moves_str = self.lookup_tables[stage][cube_hash]
                print(f"[DEBUG] Stage {stage} hash: {cube_hash}")

                if moves_str:
                    return moves_str.split()
        
        # Fallback to BFS if lookup table doesn't have this state
        print(f"Using BFS for stage {stage + 1} (not in lookup table)")
        return self._solve_stage_bfs(cube, stage)
    
    def _solve_stage_bfs(self, cube: RubiksCube, stage: int) -> Optional[List[str]]:
        """
        Solve a specific stage using breadth-first search (fallback method)
        """
        # Use BFS to find solution for this stage
        queue = deque([(cube.copy(), [])])
        visited = set()
        
        for depth in range(self.max_depth[stage]):
            next_queue = deque()
            
            while queue:
                current_cube, moves = queue.popleft()
                
                # Try each allowed move for this stage
                for move in self.stage_moves[stage]:
                    test_cube = current_cube.copy()
                    test_cube.apply_move(move)
                    
                    # Create a simple hash of the cube state
                    cube_hash = self._hash_cube_for_stage(test_cube, stage)
                    
                    if cube_hash in visited:
                        continue
                    
                    visited.add(cube_hash)
                    new_moves = moves + [move]
                    
                    if self._is_stage_complete(test_cube, stage):
                        return new_moves
                    
                    if len(new_moves) < self.max_depth[stage]:
                        next_queue.append((test_cube, new_moves))
            
            queue = next_queue
            if not queue:
                break
        
        return None
    
    def _is_stage_complete(self, cube: RubiksCube, stage: int) -> bool:
        """
        Check if a stage is complete based on Thistlethwaite criteria
        """
        if stage == 0:
            # Stage 1: All edges oriented (no bad edges)
            return self._check_edge_orientation(cube)
        elif stage == 1:
            # Stage 2: All corners oriented and E-slice edges in E-slice
            return self._check_corner_orientation(cube) and self._check_e_slice(cube)
        elif stage == 2:
            # Stage 3: All corners in correct layer, E-slice solved
            return self._check_corner_permutation(cube) and self._check_e_slice_solved(cube)
        elif stage == 3:
            # Stage 4: Completely solved
            return cube.is_solved()
        
        return False
    
    def _check_edge_orientation(self, cube: RubiksCube) -> bool:
        """Check if all edges are correctly oriented"""
        # Simplified check - in practice this would be more complex
        edge_positions = [
            (0, 1), (0, 3), (0, 5), (0, 7),  # Top edges
            (2, 1), (2, 3), (2, 5), (2, 7),  # Front edges
            (5, 1), (5, 3), (5, 5), (5, 7)   # Bottom edges
        ]
        
        good_orientations = 0
        for face, pos in edge_positions:
            if pos in [1, 7]:  # Top/bottom edges of face
                if cube.cube[face][pos] in [0, 5]:  # White or yellow
                    good_orientations += 1
            else:  # Left/right edges of face
                if cube.cube[face][pos] not in [0, 5]:  # Not white or yellow
                    good_orientations += 1
        
        return good_orientations >= 10  # Simplified threshold
    
    def _check_corner_orientation(self, cube: RubiksCube) -> bool:
        """Check if all corners are correctly oriented"""
        # Count corners with white/yellow stickers on top/bottom
        corner_positions = [(0, 0), (0, 2), (0, 6), (0, 8), (5, 0), (5, 2), (5, 6), (5, 8)]
        correct_corners = sum(1 for face, pos in corner_positions 
                            if cube.cube[face][pos] in [0, 5])
        return correct_corners >= 6
    
    def _check_e_slice(self, cube: RubiksCube) -> bool:
        """Check if E-slice edges are in E-slice"""
        # Simplified - check if middle layer edges are reasonably positioned
        middle_edges = [(1, 5), (2, 3), (3, 5), (4, 3)]
        return True  # Simplified for demo
    
    def _check_corner_permutation(self, cube: RubiksCube) -> bool:
        """Check if corners are in correct relative positions"""
        # Simplified check
        return True
    
    def _check_e_slice_solved(self, cube: RubiksCube) -> bool:
        """Check if E-slice is completely solved"""
        # Simplified check
        return True
    
    def _hash_cube_for_stage(self, cube: RubiksCube, stage: int) -> str:
        """
        Create a hash of relevant cube features for the stage
        This needs to match the hashing used in your lookup table files
        """
        # For now, we'll create a simple hash based on cube state
        # You may need to adjust this to match your specific hash format
        
        if stage == 0:
            # Stage 1: Hash edge orientations
            return self._hash_edge_orientation(cube)
        elif stage == 1:
            # Stage 2: Hash corner orientations and E-slice
            return self._hash_corner_and_e_slice(cube)
        elif stage == 2:
            # Stage 3: Hash corner positions and E-slice solved
            return self._hash_corner_permutation(cube)
        else:
            # Stage 4: Hash remaining cube state
            return self._hash_final_state(cube)
    
    def _hash_edge_orientation(self, cube: RubiksCube) -> str:
        """Create hash for edge orientations (Stage 1)"""
        # This is a simplified hash - you may need to adjust based on your lookup table format
        edge_data = []
        
        # Get edge pieces and their orientations
        edges = [
            (0, 1), (0, 3), (0, 5), (0, 7),  # Top edges
            (1, 1), (1, 3), (1, 5), (1, 7),  # Left edges  
            (2, 1), (2, 3), (2, 5), (2, 7),  # Front edges
            (3, 1), (3, 3), (3, 5), (3, 7),  # Right edges
            (4, 1), (4, 3), (4, 5), (4, 7),  # Back edges
            (5, 1), (5, 3), (5, 5), (5, 7)   # Bottom edges
        ]
        
        for face, pos in edges:
            edge_data.append(cube.cube[face][pos])
        
        # Convert to string hash (you may need to adjust format)
        hash_val = 0
        for i, val in enumerate(edge_data):
            hash_val += int(val) * (2 ** i)
        
        return f"{hash_val:06d}"  # 6-digit hash to match your format
    
    def _hash_corner_and_e_slice(self, cube: RubiksCube) -> str:
        """Create hash for corner orientations and E-slice (Stage 2)"""
        # Simplified hash for demonstration
        corner_data = []
        corners = [(0, 0), (0, 2), (0, 6), (0, 8), (5, 0), (5, 2), (5, 6), (5, 8)]
        
        for face, pos in corners:
            corner_data.append(cube.cube[face][pos])
        
        hash_val = sum(int(v) for v in corner_data) % 1000000

        return f"{hash_val:06d}"
    
    def _hash_corner_permutation(self, cube: RubiksCube) -> str:
        """
        Create hash for corner permutations (Stage 2)
        Updated to match 7-digit format used in stage2.txt
        """
        # Use all 8 corner facelets (on U and D faces)
        corner_indices = [
            (0, 0), (0, 2), (0, 6), (0, 8),  # U face corners
            (5, 0), (5, 2), (5, 6), (5, 8)   # D face corners
        ]

        corner_vals = [int(cube.cube[f][i]) for f, i in corner_indices]

        hash_val = 0
        for idx, val in enumerate(corner_vals):
            hash_val += val * (6 ** idx)  # base-6 encoding to ensure unique 7-digit hash

        return f"{hash_val:07d}"  # pad to 7 digits (e.g., 0000001)

    
    def _hash_final_state(self, cube: RubiksCube) -> str:
        """Create hash for final state (Stage 4)"""
        # Simple hash of entire cube state
        flat_cube = cube.cube.flatten()
        hash_val = sum(i * int(flat_cube[i]) for i in range(len(flat_cube))) % 1000000

        return f"{hash_val:06d}"
    
    def _optimize_moves(self, moves: str) -> str:
        """Optimize move sequence by canceling redundant moves"""
        if not moves.strip():
            return ""
        
        move_list = moves.strip().split()
        optimized = []
        
        i = 0
        while i < len(move_list):
            current = move_list[i]
            
            # Look ahead for same face moves
            if i + 1 < len(move_list):
                next_move = move_list[i + 1]
                if current[0] == next_move[0]:  # Same face
                    combined = self._combine_moves(current, next_move)
                    if combined != "":
                        optimized.append(combined)
                    i += 2
                    continue
            
            optimized.append(current)
            i += 1
        
        return " ".join(optimized)
    
    def _combine_moves(self, move1: str, move2: str) -> str:
        """Combine two moves of the same face"""
        face = move1[0]
        
        # Count total rotations
        count1 = 1 if len(move1) == 1 else (2 if move1.endswith('2') else 3)
        count2 = 1 if len(move2) == 1 else (2 if move2.endswith('2') else 3)
        
        total = (count1 + count2) % 4
        
        if total == 0:
            return ""
        elif total == 1:
            return face
        elif total == 2:
            return face + "2"
        else:
            return face + "'"


def input_cube() -> RubiksCube:
    """Get cube state from user input"""
    cube = RubiksCube()
    
    print("Enter your cube configuration:")
    print("Use: W=White, O=Orange, G=Green, R=Red, B=Blue, Y=Yellow")
    print("Enter 9 colors for each face (left to right, top to bottom)")
    
    face_names = ["Top", "Left", "Front", "Right", "Back", "Bottom"]
    color_map = {'W': 0, 'O': 1, 'G': 2, 'R': 3, 'B': 4, 'Y': 5}
    
    for i, face_name in enumerate(face_names):
        while True:
            try:
                colors = input(f"{face_name} face (9 colors): ").upper().replace(" ", "")
                if len(colors) != 9:
                    print("Please enter exactly 9 colors")
                    continue
                
                if not all(c in color_map for c in colors):
                    print("Please use only W, O, G, R, B, Y")
                    continue
                
                cube.cube[i] = [color_map[c] for c in colors]
                break
            except Exception as e:
                print(f"Error: {e}. Please try again.")
    
    return cube


def create_example_lookup_files():
    """Create example lookup table files for demonstration"""
    print("Creating example lookup table files...")
    
    # Example entries for each stage
    examples = {
        "stage0.txt": [
            "000000 ",
            "000001 R U R'",
            "000002 iU L 2D 2B iD iF",
            "000003 F R U R' F'",
            "000004 R U2 R' U R U R'",
        ],
        "stage1.txt": [
            "000000 ",
            "000001 R U R' U'",
            "000002 R U2 R2 U' R",
            "000003 U R U' R'",
            "000004 R U R' iU",
        ],
        "stage2.txt": [
            "000000 ",
            "000001 R2 U R2 iU",
            "000002 U 2R iU 2L",
            "000003 2R U2 2R U2",
            "000004 U 2R U2 2R iU",
        ],
        "stage3.txt": [
            "000000 ",
            "000001 U iU",
            "000002 U2",
            "000003 iU",
            "000004 U",
        ]
    }
    
    for filename, entries in examples.items():
        with open(filename, 'w') as f:
            for entry in entries:
                f.write(entry + " NP .\n")
        print(f"Created {filename} with {len(entries)} entries")


def main():
    """Main function to run the solver"""
    print("Thistlethwaite Rubik's Cube Solver")
    print("=" * 40)
    
    # Check if lookup files exist, if not create examples
    stage_files = ["stage0.txt", "stage1.txt", "stage2.txt", "stage3.txt"]
    if not all(os.path.exists(f) for f in stage_files):
        create_choice = input("Lookup table files not found. Create example files? (y/n): ").lower()
        if create_choice == 'y':
            create_example_lookup_files()
    
    # Option to use scrambled cube or input custom cube
    choice = input("Use scrambled cube (s) or input your own (i)? ").lower()
    
    if choice == 's':
        # Create a scrambled cube
        cube = RubiksCube()
        scramble = "R U R' F R F' U R U' R' F R F' U R U2 R'"
        print(f"Applying scramble: {scramble}")
        cube.apply_moves(scramble)
    else:
        cube = input_cube()
    
    print("\nInitial cube state:")
    cube.print_cube()
    
    if cube.is_solved():
        print("Cube is already solved!")
        return

    def add_test_option_to_main():
        """Add this code snippet to your main() function after the cube is created"""
        
        # ... existing code ...
        
        # Add test moves option
        test_choice = input("Do you want to test some moves first? (y/n): ").lower()
        if test_choice == 'y':
            while True:
                test_moves_input = input("Enter moves to test (or 'done' to continue): ")
                if test_moves_input.lower() == 'done':
                    break
                cube.test_moves(test_moves_input)
                
                # Ask if they want to keep the current state or reset
                keep_choice = input("Keep current state? (y/n): ").lower()
                if keep_choice == 'n':
                    if choice == 's':
                        cube = RubiksCube()
                        cube.apply_moves(scramble)
                    else:
                        cube = input_cube()
    # Solve the cube
    solver = ThistlethwaiteSolver()
    start_time = time.time()
    solution = solver.solve(cube)
    solve_time = time.time() - start_time
    
    if solution:
        print(f"\nSolution found in {solve_time:.2f} seconds:")
        print(f"Moves: {solution}")
        print(f"Move count: {len(solution.split())}")
        
        # Verify solution
        test_cube = RubiksCube()
        if choice == 's':
            test_cube.apply_moves(scramble)
        else:
            test_cube = input_cube()
        
        test_cube.apply_moves(solution)
        print(f"\nVerification: {'PASSED' if test_cube.is_solved() else 'FAILED'}")
        
        if test_cube.is_solved():
            print("Final cube state:")
            test_cube.print_cube()
    else:
        print("Failed to find solution!")


if __name__ == "__main__":
    main()