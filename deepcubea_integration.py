"""
DeepCubeA Integration Module

This module integrates the DeepCubeA optimal solver with your existing
Rubik's cube project, including the CV solver and 3D visualization.

Usage:
    from deepcubea_integration import DeepCubeAIntegration
    
    # Initialize
    solver = DeepCubeAIntegration()
    
    # Train (if not already trained)
    solver.train_if_needed()
    
    # Solve from CV captured state
    solution = solver.solve_from_cv_state(cv_cube_state)
    
    # Get moves for 3D cube
    moves = solver.convert_to_rubik_moves(solution)
"""

import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import threading
import queue

# Import the DeepCubeA solver
try:
    from deepcubea import DeepCubeASolver, RubiksCube3x3
    DEEPCUBEA_AVAILABLE = True
except ImportError:
    DEEPCUBEA_AVAILABLE = False
    print("Warning: DeepCubeA module not available")


class DeepCubeAIntegration:
    """
    Integration class for using DeepCubeA with the existing Rubik's cube project
    
    Features:
    - Converts CV solver state to DeepCubeA format
    - Converts DeepCubeA moves to 3D visualization format
    - Provides async solving for UI responsiveness
    - Handles training and model management
    """
    
    # Move mapping from DeepCubeA to your 3D cube format
    MOVE_TO_3D = {
        'U':  {'axis': np.array([0, 1, 0]), 'layer': 2, 'clockwise': True},
        "U'": {'axis': np.array([0, 1, 0]), 'layer': 2, 'clockwise': False},
        'D':  {'axis': np.array([0, 1, 0]), 'layer': 0, 'clockwise': False},
        "D'": {'axis': np.array([0, 1, 0]), 'layer': 0, 'clockwise': True},
        'R':  {'axis': np.array([1, 0, 0]), 'layer': 2, 'clockwise': True},
        "R'": {'axis': np.array([1, 0, 0]), 'layer': 2, 'clockwise': False},
        'L':  {'axis': np.array([1, 0, 0]), 'layer': 0, 'clockwise': False},
        "L'": {'axis': np.array([1, 0, 0]), 'layer': 0, 'clockwise': True},
        'F':  {'axis': np.array([0, 0, 1]), 'layer': 2, 'clockwise': True},
        "F'": {'axis': np.array([0, 0, 1]), 'layer': 2, 'clockwise': False},
        'B':  {'axis': np.array([0, 0, 1]), 'layer': 0, 'clockwise': False},
        "B'": {'axis': np.array([0, 0, 1]), 'layer': 0, 'clockwise': True},
    }
    
    # Color name normalization
    COLOR_ALIASES = {
        # Standard names
        'white': 'white', 'yellow': 'yellow', 'red': 'red',
        'orange': 'orange', 'green': 'green', 'blue': 'blue',
        # Single letter codes
        'W': 'white', 'Y': 'yellow', 'R': 'red',
        'O': 'orange', 'G': 'green', 'B': 'blue',
        # Lowercase single letters
        'w': 'white', 'y': 'yellow', 'r': 'red',
        'o': 'orange', 'g': 'green', 'b': 'blue',
    }
    
    def __init__(self, 
                 model_path: str = 'deepcubea_model.pth',
                 auto_load: bool = True):
        """
        Initialize the DeepCubeA integration
        
        Args:
            model_path: Path to the trained model
            auto_load: Whether to load model automatically if it exists
        """
        self.model_path = model_path
        self.solver = None
        self.is_solving = False
        self.solution_queue = queue.Queue()
        
        if not DEEPCUBEA_AVAILABLE:
            print("DeepCubeA not available. Please check deepcubea.py")
            return
        
        # Initialize solver
        if auto_load and os.path.exists(model_path):
            self.solver = DeepCubeASolver(model_path=model_path)
            print(f"✓ DeepCubeA model loaded from {model_path}")
        else:
            print(f"DeepCubeA model not found at {model_path}")
            print("Run train_if_needed() or train() to create a model")
    
    @property
    def is_ready(self) -> bool:
        """Check if solver is ready to solve"""
        return self.solver is not None and self.solver.net is not None
    
    def train_if_needed(self,
                        n_iterations: int = 5000,
                        force_retrain: bool = False) -> bool:
        """
        Train the model if it doesn't exist
        
        Args:
            n_iterations: Training iterations
            force_retrain: Force retraining even if model exists
            
        Returns:
            True if model is ready
        """
        if not DEEPCUBEA_AVAILABLE:
            return False
        
        if not force_retrain and os.path.exists(self.model_path):
            if self.solver is None:
                self.solver = DeepCubeASolver(model_path=self.model_path)
            print("Model already exists and loaded")
            return True
        
        print(f"Training DeepCubeA model ({n_iterations} iterations)...")
        print("This may take a while...")
        
        self.solver = DeepCubeASolver(model_path=self.model_path)
        self.solver.train(
            n_iterations=n_iterations,
            batch_size=500,
            scramble_depth=20,
            save_freq=500
        )
        
        return self.is_ready
    
    def train(self, **kwargs):
        """
        Train the model with custom parameters
        
        Keyword Args:
            n_iterations: Number of training iterations
            batch_size: Batch size per iteration  
            scramble_depth: Maximum scramble depth
            save_freq: How often to save model
        """
        if not DEEPCUBEA_AVAILABLE:
            print("DeepCubeA not available")
            return
        
        if self.solver is None:
            self.solver = DeepCubeASolver(model_path=self.model_path)
        
        self.solver.train(**kwargs)
    
    def normalize_cv_state(self, cv_cube_state: Dict) -> Dict:
        """
        Normalize CV cube state colors to standard names
        
        Args:
            cv_cube_state: Dict from CV solver with various color formats
            
        Returns:
            Normalized dict with standard color names
        """
        normalized = {}
        
        for face, colors in cv_cube_state.items():
            normalized_face = []
            for row in colors:
                normalized_row = []
                for color in row:
                    # Normalize color name
                    color_str = str(color).lower().strip()
                    normalized_color = self.COLOR_ALIASES.get(color_str, color_str)
                    normalized_row.append(normalized_color)
                normalized_face.append(normalized_row)
            normalized[face] = normalized_face
        
        return normalized
    
    def cv_state_to_deepcubea(self, cv_cube_state: Dict) -> np.ndarray:
        """
        Convert CV solver cube state to DeepCubeA format
        
        Args:
            cv_cube_state: Dict with keys 'F', 'R', 'B', 'L', 'U', 'D'
                          Each containing 3x3 array of color names
                          
        Returns:
            54-element numpy array for DeepCubeA
        """
        # Normalize colors first
        normalized = self.normalize_cv_state(cv_cube_state)
        
        # Color to index mapping
        color_to_idx = {
            'white': 0,
            'yellow': 1,
            'red': 2,
            'orange': 3,
            'green': 4,
            'blue': 5
        }
        
        # Face order in DeepCubeA state array
        face_order = ['U', 'D', 'F', 'B', 'R', 'L']
        
        state = np.zeros(54, dtype=np.int8)
        
        for face_idx, face_name in enumerate(face_order):
            if face_name not in normalized:
                print(f"Warning: Face {face_name} not found in cube state")
                continue
            
            face_colors = normalized[face_name]
            
            for i in range(3):
                for j in range(3):
                    color = face_colors[i][j]
                    color_idx = color_to_idx.get(color, 0)
                    sticker_idx = face_idx * 9 + i * 3 + j
                    state[sticker_idx] = color_idx
        
        return state
    
    def solve_from_cv_state(self, 
                            cv_cube_state: Dict,
                            max_nodes: int = 500000,
                            time_limit: float = 30.0,
                            verbose: bool = True) -> Optional[List[str]]:
        """
        Solve cube from CV solver state
        
        Args:
            cv_cube_state: Dict from CV solver
            max_nodes: Maximum search nodes
            time_limit: Time limit in seconds
            verbose: Print progress
            
        Returns:
            List of move strings or None
        """
        if not self.is_ready:
            print("Solver not ready. Train or load a model first.")
            return None
        
        # Convert state
        state = self.cv_state_to_deepcubea(cv_cube_state)
        
        if verbose:
            print("Converting CV state to DeepCubeA format...")
            print(f"State: {state}")
        
        # Solve
        solution = self.solver.solve(
            state=state,
            max_nodes=max_nodes,
            time_limit=time_limit,
            verbose=verbose
        )
        
        return solution
    
    def solve_async(self, 
                    cv_cube_state: Dict,
                    max_nodes: int = 500000,
                    time_limit: float = 30.0):
        """
        Solve cube asynchronously (non-blocking)
        
        Use is_solving property to check status
        Use get_solution() to get result
        
        Args:
            cv_cube_state: Dict from CV solver
            max_nodes: Maximum search nodes
            time_limit: Time limit in seconds
        """
        if self.is_solving:
            print("Already solving")
            return
        
        self.is_solving = True
        
        def solve_worker():
            try:
                solution = self.solve_from_cv_state(
                    cv_cube_state,
                    max_nodes=max_nodes,
                    time_limit=time_limit,
                    verbose=True
                )
                self.solution_queue.put(('success', solution))
            except Exception as e:
                self.solution_queue.put(('error', str(e)))
            finally:
                self.is_solving = False
        
        thread = threading.Thread(target=solve_worker)
        thread.start()
    
    def get_solution(self) -> Optional[Tuple[str, any]]:
        """
        Get solution from async solve (non-blocking)
        
        Returns:
            Tuple of (status, result) or None if not ready
            status: 'success' or 'error'
            result: List of moves or error message
        """
        try:
            return self.solution_queue.get_nowait()
        except queue.Empty:
            return None
    
    def convert_to_rubik_moves(self, 
                                solution: List[str]) -> List[Dict]:
        """
        Convert DeepCubeA solution to 3D Rubik cube format
        
        Args:
            solution: List of move strings from DeepCubeA
            
        Returns:
            List of dicts with 'axis', 'layer', 'clockwise' for 3D cube
        """
        moves_3d = []
        
        for move in solution:
            if move in self.MOVE_TO_3D:
                moves_3d.append(self.MOVE_TO_3D[move].copy())
            else:
                print(f"Warning: Unknown move '{move}'")
        
        return moves_3d
    
    def get_solution_for_rubik_class(self, 
                                      solution: List[str],
                                      rubik_cube) -> List:
        """
        Convert solution to rotation queue entries for your Rubik class
        
        Args:
            solution: List of move strings
            rubik_cube: Your Rubik class instance
            
        Returns:
            List of rotation queue entries
        """
        rotation_queue = []
        
        for move in solution:
            if move in self.MOVE_TO_3D:
                move_info = self.MOVE_TO_3D[move]
                rotation_queue = rubik_cube.add_rotation(
                    rotation_queue,
                    move_info['axis'],
                    move_info['layer'],
                    move_info['clockwise']
                )
        
        return rotation_queue
    
    def validate_cube_state(self, cv_cube_state: Dict) -> Tuple[bool, str]:
        """
        Validate that a cube state is valid/solvable
        
        Args:
            cv_cube_state: Dict from CV solver
            
        Returns:
            Tuple of (is_valid, message)
        """
        normalized = self.normalize_cv_state(cv_cube_state)
        
        # Check all 6 faces present
        required_faces = {'F', 'R', 'B', 'L', 'U', 'D'}
        if set(normalized.keys()) != required_faces:
            missing = required_faces - set(normalized.keys())
            return False, f"Missing faces: {missing}"
        
        # Count colors (should have exactly 9 of each)
        color_counts = {}
        for face in normalized.values():
            for row in face:
                for color in row:
                    color_counts[color] = color_counts.get(color, 0) + 1
        
        expected_colors = {'white', 'yellow', 'red', 'orange', 'green', 'blue'}
        
        for color in expected_colors:
            count = color_counts.get(color, 0)
            if count != 9:
                return False, f"Invalid count for {color}: {count} (expected 9)"
        
        # Check for unknown colors
        unknown = set(color_counts.keys()) - expected_colors
        if unknown:
            return False, f"Unknown colors detected: {unknown}"
        
        return True, "Cube state appears valid"
    
    def test_solver(self, 
                    n_tests: int = 5,
                    scramble_depth: int = 15) -> Dict:
        """
        Test the solver on random scrambles
        
        Args:
            n_tests: Number of tests
            scramble_depth: Scramble depth
            
        Returns:
            Dict with test results
        """
        if not self.is_ready:
            return {'error': 'Solver not ready'}
        
        self.solver.test(n_tests=n_tests, scramble_depth=scramble_depth)
        return {'status': 'completed'}


# =============================================================================
# Helper functions for direct use in train3d.py
# =============================================================================

_deepcubea_solver = None

def get_deepcubea_solver(model_path: str = 'deepcubea_model.pth') -> DeepCubeAIntegration:
    """
    Get or create the global DeepCubeA solver instance
    
    Args:
        model_path: Path to model file
        
    Returns:
        DeepCubeAIntegration instance
    """
    global _deepcubea_solver
    
    if _deepcubea_solver is None:
        _deepcubea_solver = DeepCubeAIntegration(model_path=model_path)
    
    return _deepcubea_solver


def solve_with_deepcubea(cv_cube_state: Dict,
                          model_path: str = 'deepcubea_model.pth',
                          time_limit: float = 30.0) -> Optional[List[str]]:
    """
    Convenience function to solve cube using DeepCubeA
    
    Args:
        cv_cube_state: Dict from CV solver
        model_path: Path to model
        time_limit: Time limit in seconds
        
    Returns:
        List of move strings or None
    """
    solver = get_deepcubea_solver(model_path)
    
    if not solver.is_ready:
        print("DeepCubeA model not trained. Training now (this takes a while)...")
        solver.train_if_needed(n_iterations=5000)
    
    return solver.solve_from_cv_state(
        cv_cube_state,
        time_limit=time_limit
    )


# =============================================================================
# Command-line interface
# =============================================================================

def main():
    """Command-line interface for DeepCubeA integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepCubeA Integration')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the solver')
    parser.add_argument('--iterations', type=int, default=5000, help='Training iterations')
    parser.add_argument('--model', type=str, default='deepcubea_model.pth', help='Model path')
    
    args = parser.parse_args()
    
    solver = DeepCubeAIntegration(model_path=args.model)
    
    if args.train:
        solver.train(
            n_iterations=args.iterations,
            batch_size=500,
            scramble_depth=20
        )
    
    if args.test:
        if not solver.is_ready:
            print("Model not ready. Train first with --train")
        else:
            solver.test_solver(n_tests=10, scramble_depth=15)


if __name__ == "__main__":
    main()
