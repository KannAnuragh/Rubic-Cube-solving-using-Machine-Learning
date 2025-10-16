

import numpy as np
import torch
from ml_solver import RubiksCubeEnv, RubiksDQNAgent
import os


class MLSolverIntegration:
    """
    Integration class to use ML solver with your existing project
    """
    
    def __init__(self, model_path='rubiks_dqn.pth', cube_size=2):
        self.model_path = model_path
        self.cube_size = cube_size
        self.agent = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if model exists
        self.model_trained = os.path.exists(model_path)
        
        if self.model_trained:
            self._load_model()
            print(f"✓ ML model loaded from {model_path}")
        else:
            print(f"✗ ML model not found at {model_path}")
            print(f"  Run: python ml_solver.py --mode train --episodes 5000")
    
    def _load_model(self):
        """Load the trained ML model"""
        env = RubiksCubeEnv(cube_size=self.cube_size)
        state_size = len(env.get_state_vector())
        action_size = env.n_actions
        
        self.agent = RubiksDQNAgent(state_size, action_size, self.device)
        self.agent.load(self.model_path)
        self.agent.epsilon = 0.0  # No exploration during inference
    
    def convert_cv_state_to_ml(self, cv_cube_state):
        """
        Convert CV solver cube state to ML environment format
        
        Args:
            cv_cube_state: Dict with keys 'F', 'R', 'B', 'L', 'U', 'D'
                          Each containing 3x3 array of color names
                          
        Returns:
            RubiksCubeEnv with converted state
        """
        # Note: ML solver uses 2x2x2 cube for training speed
        # For 3x3x3 integration, you'd need to retrain with cube_size=3
        
        env = RubiksCubeEnv(cube_size=self.cube_size)
        
        # Color name to index mapping
        color_to_idx = {
            'white': 0,
            'yellow': 1,
            'red': 2,
            'orange': 3,
            'green': 4,
            'blue': 5
        }
        
        # Face mapping: CV notation -> ML notation
        face_map = {
            'U': 'U',  # Up (white)
            'D': 'D',  # Down (yellow)
            'F': 'F',  # Front (green/red depending on orientation)
            'B': 'B',  # Back
            'R': 'R',  # Right
            'L': 'L'   # Left
        }
        
        # Convert each face
        for cv_face, ml_face in face_map.items():
            if cv_face not in cv_cube_state:
                continue
            
            cv_colors = cv_cube_state[cv_face]
            
            # For 2x2 cube, take top-left 2x2 section of 3x3 cube
            if self.cube_size == 2:
                for i in range(2):
                    for j in range(2):
                        color_name = cv_colors[i][j]
                        color_idx = color_to_idx.get(color_name, 0)
                        env.state[ml_face][i][j] = color_idx
            else:
                # For 3x3 cube, direct conversion
                for i in range(3):
                    for j in range(3):
                        color_name = cv_colors[i][j]
                        color_idx = color_to_idx.get(color_name, 0)
                        env.state[ml_face][i][j] = color_idx
        
        return env
    
    def solve_with_ml(self, cv_cube_state, max_steps=50, verbose=True):
        """
        Solve cube using trained ML model
        
        Args:
            cv_cube_state: Cube state from CV solver
            max_steps: Maximum solving steps
            verbose: Print progress
            
        Returns:
            List of move strings ['R', 'U', 'F', ...] or None if failed
        """
        if not self.model_trained:
            print("Error: ML model not trained. Train first with:")
            print("  python ml_solver.py --mode train --episodes 5000")
            return None
        
        # Convert state
        env = self.convert_cv_state_to_ml(cv_cube_state)
        
        if verbose:
            print("\n🤖 ML Solver starting...")
            print(f"   Initial state converted to {self.cube_size}x{self.cube_size}x{self.cube_size} format")
        
        # Solve using ML agent
        state = env.get_state_vector()
        moves = []
        
        for step in range(max_steps):
            action = self.agent.select_action(state, training=False)
            move_name = env.action_names[action]
            
            next_state, reward, done = env.step(action)
            moves.append(move_name)
            state = next_state
            
            if done:
                if verbose:
                    print(f"   ✓ Solved in {step + 1} moves!")
                    print(f"   Solution: {' '.join(moves)}")
                return moves
            
            if verbose and (step + 1) % 10 == 0:
                print(f"   ... {step + 1} moves attempted")
        
        if verbose:
            print(f"   ✗ Could not solve within {max_steps} moves")
            print(f"   Attempted: {' '.join(moves[:20])}...")
        
        return None
    
    def convert_ml_moves_to_3d(self, ml_moves):
        """
        Convert ML move notation to 3D cube rotation format
        
        Args:
            ml_moves: List of moves ['U', 'R', 'F', ...]
            
        Returns:
            List of (axis, level, clockwise) tuples for your 3D cube
        """
        move_mapping = {
            'U': (np.array([0, 1, 0]), 2, True),   # Y-axis, top level, clockwise
            'D': (np.array([0, 1, 0]), 0, True),   # Y-axis, bottom level, clockwise
            'R': (np.array([1, 0, 0]), 2, True),   # X-axis, right level, clockwise
            'L': (np.array([1, 0, 0]), 0, True),   # X-axis, left level, clockwise
            'F': (np.array([0, 0, 1]), 2, True),   # Z-axis, front level, clockwise
            'B': (np.array([0, 0, 1]), 0, True),   # Z-axis, back level, clockwise
        }
        
        rotations = []
        for move in ml_moves:
            if move in move_mapping:
                rotations.append(move_mapping[move])
        
        return rotations
    
    def is_model_ready(self):
        """Check if ML model is trained and ready"""
        return self.model_trained


class MLTrainingHelper:
    """
    Helper class to train the ML model from your main application
    """
    
    @staticmethod
    def train_model(episodes=5000, scramble_depth=5, model_path='rubiks_dqn.pth'):
        """
        Train ML model (can be called from your UI)
        
        Args:
            episodes: Number of training episodes (more = better, but slower)
            scramble_depth: How scrambled the training cubes will be
            model_path: Where to save the trained model
        """
        from ml_solver import train_agent
        
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODEL")
        print("="*60)
        print(f"Episodes: {episodes}")
        print(f"Scramble Depth: {scramble_depth} moves")
        print(f"This will take approximately {episodes * 0.02 / 60:.1f} minutes")
        print("\nTraining in progress...")
        
        agent = train_agent(
            n_episodes=episodes,
            scramble_moves=scramble_depth,
            model_path=model_path
        )
        
        print("\n✓ Training complete!")
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ You can now use ML solver in the application")
        
        return agent
    
    @staticmethod
    def quick_train(model_path='rubiks_dqn.pth'):
        """Quick training for demo (less accurate but faster)"""
        print("\n⚡ Quick training mode (demo only, not optimal)")
        return MLTrainingHelper.train_model(
            episodes=1000, 
            scramble_depth=3, 
            model_path=model_path
        )
    
    @staticmethod
    def full_train(model_path='rubiks_dqn.pth'):
        """Full training for best results"""
        print("\n🎯 Full training mode (recommended for best results)")
        return MLTrainingHelper.train_model(
            episodes=10000, 
            scramble_depth=7, 
            model_path=model_path
        )


def example_integration():
    """
    Example of how to integrate ML solver into train3d.py
    """
    print("\n" + "="*60)
    print("EXAMPLE: Integrating ML Solver into Your Project")
    print("="*60)
    
    # Initialize ML solver
    ml_solver = MLSolverIntegration()
    
    # Check if model is ready
    if not ml_solver.is_model_ready():
        print("\n❌ ML Model not trained yet!")
        print("\nOption 1: Quick train (demo - 5 minutes)")
        print("  python ml_solver.py --mode train --episodes 1000 --scramble 3")
        print("\nOption 2: Full train (best results - 30 minutes)")
        print("  python ml_solver.py --mode train --episodes 10000 --scramble 7")
        print("\nOr use the helper:")
        print("  from integrate_ml import MLTrainingHelper")
        print("  MLTrainingHelper.quick_train()  # or full_train()")
        return
    
    # Example: Simulate a scrambled cube state from CV
    example_cv_state = {
        'U': [['yellow', 'white', 'yellow'],
              ['white', 'yellow', 'white'],
              ['yellow', 'white', 'yellow']],
        'D': [['white', 'yellow', 'white'],
              ['yellow', 'white', 'yellow'],
              ['white', 'yellow', 'white']],
        'F': [['green', 'red', 'green'],
              ['red', 'green', 'red'],
              ['green', 'red', 'green']],
        'B': [['blue', 'orange', 'blue'],
              ['orange', 'blue', 'orange'],
              ['blue', 'orange', 'blue']],
        'R': [['red', 'green', 'red'],
              ['green', 'red', 'green'],
              ['red', 'green', 'red']],
        'L': [['orange', 'blue', 'orange'],
              ['blue', 'orange', 'blue'],
              ['orange', 'blue', 'orange']]
    }
    
    print("\n📷 Simulating CV capture of scrambled cube...")
    
    # Solve with ML
    print("\n🤖 Solving with Machine Learning...")
    ml_moves = ml_solver.solve_with_ml(example_cv_state)
    
    if ml_moves:
        # Convert to 3D rotations
        rotations = ml_solver.convert_ml_moves_to_3d(ml_moves)
        
        print(f"\n✅ ML Solution ready!")
        print(f"   Move sequence: {' '.join(ml_moves)}")
        print(f"   Total moves: {len(ml_moves)}")
        print(f"\n💡 Apply to 3D cube with:")
        print(f"   for axis, level, clockwise in rotations:")
        print(f"       rotation_queue = rubik_cube.add_rotation(")
        print(f"           rotation_queue, axis, level, clockwise)")
    else:
        print("\n❌ ML solver could not find solution")


# Code snippet to add to train3d.py
TRAIN3D_INTEGRATION_CODE = '''
# ============================================
# ADD THIS TO YOUR train3d.py FILE
# ============================================

# At the top with other imports:
from integrate_ml import MLSolverIntegration

# After initializing cv_solver:
ml_solver = MLSolverIntegration()
ml_solution_ready = False
ml_solution_moves = []

# In the main loop, add this key handler:
elif pr.is_key_pressed(pr.KEY_M) and capture_completed:
    # M key = ML Solve
    if ml_solver.is_model_ready():
        print("Solving with Machine Learning...")
        ml_moves = ml_solver.solve_with_ml(cv_solver.cube_state)
        
        if ml_moves:
            # Convert ML moves to rotation queue format
            rotations = ml_solver.convert_ml_moves_to_3d(ml_moves)
            
            # Add all rotations to queue
            for axis, level, clockwise in rotations:
                rotation_queue = rubik_cube.add_rotation(
                    rotation_queue, axis, level, clockwise)
            
            ml_solution_ready = True
            print(f"ML solution applied: {len(ml_moves)} moves")
        else:
            print("ML solver could not find solution")
    else:
        print("ML model not trained. Run: python ml_solver.py --mode train")

# Add to UI display:
if ml_solver.is_model_ready():
    pr.draw_text(b"M-ML Solve", 10, y_offset, 12, pr.DARKGRAY)
else:
    pr.draw_text(b"ML model not trained", 10, y_offset, 12, pr.RED)
y_offset += 20
'''


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train-quick':
            MLTrainingHelper.quick_train()
        elif sys.argv[1] == 'train-full':
            MLTrainingHelper.full_train()
        elif sys.argv[1] == 'example':
            example_integration()
        else:
            print("Usage:")
            print("  python integrate_ml.py train-quick  # Quick demo training")
            print("  python integrate_ml.py train-full   # Full training")
            print("  python integrate_ml.py example      # Show integration example")
    else:
        print("\n" + "="*60)
        print("ML SOLVER INTEGRATION MODULE")
        print("="*60)
        print("\nThis module connects the ML solver to your existing project.")
        print("\nUsage:")
        print("  python integrate_ml.py train-quick  # Quick training (5 min)")
        print("  python integrate_ml.py train-full   # Full training (30 min)")
        print("  python integrate_ml.py example      # Show how to use")
        print("\nIn your train3d.py:")
        print("  from integrate_ml import MLSolverIntegration")
        print("  ml_solver = MLSolverIntegration()")
        print("  solution = ml_solver.solve_with_ml(cv_solver.cube_state)")
        print("\n" + "="*60)
        
        # Check if model exists
        if os.path.exists('rubiks_dqn.pth'):
            print("\n✓ Trained model found: rubiks_dqn.pth")
            print("  You can use the ML solver now!")
        else:
            print("\n⚠ No trained model found")
            print("  Train one with: python integrate_ml.py train-quick")