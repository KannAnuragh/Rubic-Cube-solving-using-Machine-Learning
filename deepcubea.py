"""
DeepCubeA: Solving Rubik's Cube with Deep Reinforcement Learning and A* Search

This implementation is based on the paper:
"Solving the Rubik's Cube with Deep Reinforcement Learning and Search"
by Forest Agostinelli et al. (Nature Machine Intelligence, 2019)

Key Components:
1. ResNet-based Value and Policy Networks
2. Autodidactic Iteration (ADI) training
3. Batch Weighted A* Search (BWAS) for optimal solutions

The goal is to find solutions within 20-21 moves (God's Number)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import heapq
import random
from collections import deque, defaultdict
from tqdm import tqdm
import pickle
import os
from typing import List, Tuple, Dict, Optional, Set
import time


# =============================================================================
# RUBIK'S CUBE ENVIRONMENT (3x3x3 for optimal solving)
# =============================================================================

class RubiksCube3x3:
    """
    3x3x3 Rubik's Cube Environment optimized for DeepCubeA
    
    State representation: 54 stickers (6 faces × 9 stickers)
    Each sticker is one of 6 colors (0-5)
    
    Moves: 12 moves (6 faces × 2 directions: clockwise and counter-clockwise)
    U, U', D, D', F, F', B, B', L, L', R, R'
    """
    
    # Standard color indices
    WHITE = 0   # Up
    YELLOW = 1  # Down
    RED = 2     # Front
    ORANGE = 3  # Back
    GREEN = 4   # Right
    BLUE = 5    # Left
    
    # Face indices in state array
    FACE_U = 0
    FACE_D = 1
    FACE_F = 2
    FACE_B = 3
    FACE_R = 4
    FACE_L = 5
    
    # Move indices
    MOVES = ['U', "U'", 'D', "D'", 'F', "F'", 'B', "B'", 'L', "L'", 'R', "R'"]
    N_MOVES = 12
    
    # Inverse moves mapping
    INVERSE_MOVES = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 
                     6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}
    
    def __init__(self):
        self.state = self._solved_state()
        self._precompute_moves()
    
    def _solved_state(self) -> np.ndarray:
        """Return the solved cube state"""
        state = np.zeros(54, dtype=np.int8)
        for face in range(6):
            state[face * 9:(face + 1) * 9] = face
        return state
    
    def _precompute_moves(self):
        """Precompute move permutations for efficiency"""
        # Each move is a permutation of the 54 stickers
        # We precompute these for fast application
        
        self.move_perms = []
        
        for move_idx in range(self.N_MOVES):
            perm = np.arange(54, dtype=np.int8)
            
            face = move_idx // 2
            clockwise = (move_idx % 2 == 0)
            
            # Apply the move to get permutation
            perm = self._compute_move_perm(face, clockwise)
            self.move_perms.append(perm)
    
    def _compute_move_perm(self, face: int, clockwise: bool) -> np.ndarray:
        """Compute permutation array for a single move"""
        perm = np.arange(54, dtype=np.int8)
        
        # Face rotation (the face itself rotates)
        face_start = face * 9
        face_indices = list(range(face_start, face_start + 9))
        
        # Corner indices on face: 0, 2, 8, 6 (clockwise order)
        # Edge indices on face: 1, 5, 7, 3 (clockwise order)
        corners = [0, 2, 8, 6]
        edges = [1, 5, 7, 3]
        
        if clockwise:
            # Rotate corners clockwise
            for i in range(4):
                perm[face_start + corners[(i + 1) % 4]] = face_start + corners[i]
            # Rotate edges clockwise
            for i in range(4):
                perm[face_start + edges[(i + 1) % 4]] = face_start + edges[i]
        else:
            # Rotate counter-clockwise
            for i in range(4):
                perm[face_start + corners[i]] = face_start + corners[(i + 1) % 4]
            for i in range(4):
                perm[face_start + edges[i]] = face_start + edges[(i + 1) % 4]
        
        # Adjacent stickers that cycle
        # This depends on which face we're rotating
        adjacent = self._get_adjacent_stickers(face)
        
        if clockwise:
            # Cycle adjacent stickers clockwise
            temp = [perm[i] for i in adjacent[0]]
            for i in range(4):
                for j in range(3):
                    perm[adjacent[(i + 1) % 4][j]] = adjacent[i][j]
        else:
            # Cycle adjacent stickers counter-clockwise
            for i in range(4):
                for j in range(3):
                    perm[adjacent[i][j]] = adjacent[(i + 1) % 4][j]
        
        return perm
    
    def _get_adjacent_stickers(self, face: int) -> List[List[int]]:
        """Get the adjacent stickers that cycle when rotating a face"""
        # Returns 4 groups of 3 stickers each that cycle
        
        if face == self.FACE_U:  # Up face
            return [
                [2 * 9 + 0, 2 * 9 + 1, 2 * 9 + 2],  # Front top
                [5 * 9 + 0, 5 * 9 + 1, 5 * 9 + 2],  # Left top
                [3 * 9 + 0, 3 * 9 + 1, 3 * 9 + 2],  # Back top
                [4 * 9 + 0, 4 * 9 + 1, 4 * 9 + 2],  # Right top
            ]
        elif face == self.FACE_D:  # Down face
            return [
                [2 * 9 + 6, 2 * 9 + 7, 2 * 9 + 8],  # Front bottom
                [4 * 9 + 6, 4 * 9 + 7, 4 * 9 + 8],  # Right bottom
                [3 * 9 + 6, 3 * 9 + 7, 3 * 9 + 8],  # Back bottom
                [5 * 9 + 6, 5 * 9 + 7, 5 * 9 + 8],  # Left bottom
            ]
        elif face == self.FACE_F:  # Front face
            return [
                [0 * 9 + 6, 0 * 9 + 7, 0 * 9 + 8],  # Up bottom
                [4 * 9 + 0, 4 * 9 + 3, 4 * 9 + 6],  # Right left column
                [1 * 9 + 2, 1 * 9 + 1, 1 * 9 + 0],  # Down top (reversed)
                [5 * 9 + 8, 5 * 9 + 5, 5 * 9 + 2],  # Left right column (reversed)
            ]
        elif face == self.FACE_B:  # Back face
            return [
                [0 * 9 + 2, 0 * 9 + 1, 0 * 9 + 0],  # Up top (reversed)
                [5 * 9 + 0, 5 * 9 + 3, 5 * 9 + 6],  # Left left column
                [1 * 9 + 6, 1 * 9 + 7, 1 * 9 + 8],  # Down bottom
                [4 * 9 + 8, 4 * 9 + 5, 4 * 9 + 2],  # Right right column (reversed)
            ]
        elif face == self.FACE_R:  # Right face
            return [
                [2 * 9 + 2, 2 * 9 + 5, 2 * 9 + 8],  # Front right column
                [0 * 9 + 2, 0 * 9 + 5, 0 * 9 + 8],  # Up right column
                [3 * 9 + 6, 3 * 9 + 3, 3 * 9 + 0],  # Back left column (reversed)
                [1 * 9 + 2, 1 * 9 + 5, 1 * 9 + 8],  # Down right column
            ]
        elif face == self.FACE_L:  # Left face
            return [
                [2 * 9 + 0, 2 * 9 + 3, 2 * 9 + 6],  # Front left column
                [1 * 9 + 0, 1 * 9 + 3, 1 * 9 + 6],  # Down left column
                [3 * 9 + 8, 3 * 9 + 5, 3 * 9 + 2],  # Back right column (reversed)
                [0 * 9 + 0, 0 * 9 + 3, 0 * 9 + 6],  # Up left column
            ]
        
        return []
    
    def reset(self) -> np.ndarray:
        """Reset to solved state"""
        self.state = self._solved_state()
        return self.state.copy()
    
    def apply_move(self, move_idx: int) -> np.ndarray:
        """Apply a move and return new state"""
        new_state = np.zeros(54, dtype=np.int8)
        perm = self.move_perms[move_idx]
        for i in range(54):
            new_state[perm[i]] = self.state[i]
        self.state = new_state
        return self.state.copy()
    
    def apply_move_to_state(self, state: np.ndarray, move_idx: int) -> np.ndarray:
        """Apply a move to a given state (without modifying self.state)"""
        new_state = np.zeros(54, dtype=np.int8)
        perm = self.move_perms[move_idx]
        for i in range(54):
            new_state[perm[i]] = state[i]
        return new_state
    
    def is_solved(self, state: Optional[np.ndarray] = None) -> bool:
        """Check if cube is solved"""
        if state is None:
            state = self.state
        
        for face in range(6):
            face_color = state[face * 9]
            if not np.all(state[face * 9:(face + 1) * 9] == face_color):
                return False
        return True
    
    def scramble(self, n_moves: int) -> Tuple[np.ndarray, List[int]]:
        """Scramble the cube with random moves, return state and inverse moves"""
        self.reset()
        moves = []
        prev_move = -1
        
        for _ in range(n_moves):
            # Avoid undoing the previous move
            valid_moves = [m for m in range(self.N_MOVES) 
                          if m != self.INVERSE_MOVES.get(prev_move, -1)]
            move = random.choice(valid_moves)
            self.apply_move(move)
            moves.append(move)
            prev_move = move
        
        # Return inverse moves (to solve from this state)
        inverse_moves = [self.INVERSE_MOVES[m] for m in reversed(moves)]
        return self.state.copy(), inverse_moves
    
    def get_state_one_hot(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert state to one-hot encoding for neural network"""
        if state is None:
            state = self.state
        
        one_hot = np.zeros((54, 6), dtype=np.float32)
        one_hot[np.arange(54), state] = 1.0
        return one_hot.flatten()
    
    def get_all_children(self, state: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """Get all child states reachable from current state"""
        children = []
        for move_idx in range(self.N_MOVES):
            child_state = self.apply_move_to_state(state, move_idx)
            children.append((child_state, move_idx))
        return children


# =============================================================================
# NEURAL NETWORK ARCHITECTURE (ResNet-based)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = F.relu(out + residual)
        return out


class DeepCubeANet(nn.Module):
    """
    DeepCubeA Neural Network
    
    Architecture:
    - Input: One-hot encoded cube state (54 stickers × 6 colors = 324)
    - Fully connected layers with residual blocks
    - Two output heads:
        - Value head: Predicts distance to solved state
        - Policy head: Predicts probability of each move being optimal
    """
    
    def __init__(self, 
                 input_dim: int = 324,  # 54 * 6 one-hot
                 hidden_dim: int = 5000,
                 n_residual_blocks: int = 4,
                 n_moves: int = 12):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_moves = n_moves
        
        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)
        ])
        
        # Value head (predicts cost-to-go / distance to solved)
        self.value_fc1 = nn.Linear(hidden_dim, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        # Policy head (predicts move probabilities)
        self.policy_fc1 = nn.Linear(hidden_dim, 512)
        self.policy_fc2 = nn.Linear(512, n_moves)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Batch of one-hot encoded states (batch_size, 324)
            
        Returns:
            value: Predicted cost-to-go (batch_size, 1)
            policy: Move probabilities (batch_size, 12)
        """
        # Input projection
        h = F.relu(self.input_bn(self.input_fc(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)
        
        # Value head
        v = F.relu(self.value_fc1(h))
        value = self.value_fc2(v)
        
        # Policy head
        p = F.relu(self.policy_fc1(h))
        policy = F.softmax(self.policy_fc2(p), dim=-1)
        
        return value, policy
    
    def predict_value(self, x: torch.Tensor) -> torch.Tensor:
        """Predict only value (for A* heuristic)"""
        value, _ = self.forward(x)
        return value
    
    def predict_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Predict only policy"""
        _, policy = self.forward(x)
        return policy


# =============================================================================
# AUTODIDACTIC ITERATION (ADI) TRAINING
# =============================================================================

class AutodidacticIteration:
    """
    Autodidactic Iteration (ADI) for training DeepCubeA
    
    Key idea: Generate training data by scrambling from solved state
    and using value iteration to compute targets.
    
    For each scrambled state:
    - Value target = 1 + min(value of children)
    - Policy target = argmin(value of children)
    """
    
    def __init__(self,
                 hidden_dim: int = 5000,
                 n_residual_blocks: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr: float = 0.0001,
                 weight_decay: float = 0.00001):
        
        self.device = torch.device(device)
        self.cube = RubiksCube3x3()
        
        # Initialize network
        self.net = DeepCubeANet(
            hidden_dim=hidden_dim,
            n_residual_blocks=n_residual_blocks
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Training stats
        self.training_step = 0
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
    
    def generate_training_batch(self, 
                                 batch_size: int = 1000,
                                 scramble_depth: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of training data
        
        For each sample:
        1. Scramble cube with random depth d ~ Uniform(1, scramble_depth)
        2. Get all children states
        3. Compute value target = 1 + min(value of children)
        4. Compute policy target = argmin(value of children)
        
        Returns:
            states: (batch_size, 324) one-hot encoded states
            value_targets: (batch_size,) target values
            policy_targets: (batch_size,) target move indices
        """
        states = []
        
        for _ in range(batch_size):
            # Random scramble depth
            depth = random.randint(1, scramble_depth)
            
            # Scramble and get state
            state, _ = self.cube.scramble(depth)
            states.append(state)
        
        # Convert to one-hot
        states_onehot = np.array([
            self.cube.get_state_one_hot(s) for s in states
        ], dtype=np.float32)
        
        # Get all children for each state
        all_children = []
        for state in states:
            children = self.cube.get_all_children(state)
            children_onehot = np.array([
                self.cube.get_state_one_hot(c[0]) for c in children
            ], dtype=np.float32)
            all_children.append(children_onehot)
        
        # Stack all children (batch_size, 12, 324)
        all_children = np.stack(all_children)
        
        return states_onehot, all_children, states
    
    def compute_targets(self, 
                        children_onehot: np.ndarray,
                        raw_states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute value and policy targets using current network
        
        Args:
            children_onehot: (batch_size, 12, 324) children states
            raw_states: List of raw state arrays for checking solved
            
        Returns:
            value_targets: (batch_size,)
            policy_targets: (batch_size,)
        """
        batch_size = len(children_onehot)
        
        # Flatten children for batch prediction
        children_flat = children_onehot.reshape(-1, 324)
        children_tensor = torch.FloatTensor(children_flat).to(self.device)
        
        # Get values for all children
        with torch.no_grad():
            child_values, _ = self.net(children_tensor)
        
        child_values = child_values.cpu().numpy().reshape(batch_size, 12)
        
        # For each state, find best child
        value_targets = np.zeros(batch_size, dtype=np.float32)
        policy_targets = np.zeros(batch_size, dtype=np.int64)
        
        for i in range(batch_size):
            # Check if any child is solved (value = 0)
            min_child_value = float('inf')
            best_move = 0
            
            for move_idx in range(12):
                child_state = self.cube.apply_move_to_state(raw_states[i], move_idx)
                
                if self.cube.is_solved(child_state):
                    # Child is solved, value = 0
                    child_val = 0.0
                else:
                    child_val = child_values[i, move_idx]
                
                if child_val < min_child_value:
                    min_child_value = child_val
                    best_move = move_idx
            
            # Value target = 1 + min(child values)
            value_targets[i] = 1.0 + min_child_value
            policy_targets[i] = best_move
        
        return value_targets, policy_targets
    
    def train_step(self, 
                   states: np.ndarray,
                   value_targets: np.ndarray,
                   policy_targets: np.ndarray) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            states: (batch_size, 324)
            value_targets: (batch_size,)
            policy_targets: (batch_size,)
            
        Returns:
            Dictionary of losses
        """
        self.net.train()
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        value_targets_t = torch.FloatTensor(value_targets).unsqueeze(1).to(self.device)
        policy_targets_t = torch.LongTensor(policy_targets).to(self.device)
        
        # Forward pass
        value_pred, policy_pred = self.net(states_t)
        
        # Losses
        value_loss = F.mse_loss(value_pred, value_targets_t)
        policy_loss = F.cross_entropy(
            self.net.policy_fc2(F.relu(self.net.policy_fc1(
                self._get_features(states_t)
            ))),
            policy_targets_t
        )
        
        # Combined loss
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        # Record stats
        self.training_step += 1
        losses = {
            'total': total_loss.item(),
            'value': value_loss.item(),
            'policy': policy_loss.item()
        }
        
        self.losses.append(losses['total'])
        self.value_losses.append(losses['value'])
        self.policy_losses.append(losses['policy'])
        
        return losses
    
    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from input (for policy head)"""
        h = F.relu(self.net.input_bn(self.net.input_fc(x)))
        for block in self.net.residual_blocks:
            h = block(h)
        return h
    
    def train(self,
              n_iterations: int = 10000,
              batch_size: int = 1000,
              scramble_depth: int = 30,
              save_freq: int = 1000,
              model_path: str = 'deepcubea_model.pth',
              log_freq: int = 100):
        """
        Main training loop
        
        Args:
            n_iterations: Number of training iterations
            batch_size: Batch size per iteration
            scramble_depth: Maximum scramble depth
            save_freq: How often to save model
            model_path: Path to save model
            log_freq: How often to log progress
        """
        print(f"\n{'='*60}")
        print("DeepCubeA Training - Autodidactic Iteration")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Iterations: {n_iterations}")
        print(f"Batch size: {batch_size}")
        print(f"Scramble depth: 1-{scramble_depth}")
        print(f"{'='*60}\n")
        
        for iteration in tqdm(range(n_iterations), desc="Training"):
            # Generate training batch
            states, children, raw_states = self.generate_training_batch(
                batch_size=batch_size,
                scramble_depth=scramble_depth
            )
            
            # Compute targets using current network
            value_targets, policy_targets = self.compute_targets(children, raw_states)
            
            # Train step
            losses = self.train_step(states, value_targets, policy_targets)
            
            # Logging
            if (iteration + 1) % log_freq == 0:
                avg_total = np.mean(self.losses[-log_freq:])
                avg_value = np.mean(self.value_losses[-log_freq:])
                avg_policy = np.mean(self.policy_losses[-log_freq:])
                
                print(f"\nIteration {iteration + 1}/{n_iterations}")
                print(f"  Total Loss: {avg_total:.4f}")
                print(f"  Value Loss: {avg_value:.4f}")
                print(f"  Policy Loss: {avg_policy:.4f}")
            
            # Save model
            if (iteration + 1) % save_freq == 0:
                self.save(model_path)
        
        # Final save
        self.save(model_path)
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Model saved to: {model_path}")
        print(f"{'='*60}")
    
    def save(self, filepath: str):
        """Save model and training state"""
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'losses': self.losses,
            'value_losses': self.value_losses,
            'policy_losses': self.policy_losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.losses = checkpoint.get('losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        print(f"Model loaded from {filepath}")


# =============================================================================
# BATCH WEIGHTED A* SEARCH (BWAS)
# =============================================================================

class SearchNode:
    """Node in the search tree"""
    
    def __init__(self, 
                 state: np.ndarray,
                 g_cost: float,  # Cost from start
                 h_cost: float,  # Heuristic (network prediction)
                 parent: Optional['SearchNode'] = None,
                 move: Optional[int] = None):
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost  # Total estimated cost
        self.parent = parent
        self.move = move
        self.state_hash = self._hash_state(state)
    
    def _hash_state(self, state: np.ndarray) -> int:
        """Hash state for efficient lookup"""
        return hash(state.tobytes())
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def get_path(self) -> List[int]:
        """Reconstruct path from start to this node"""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.move)
            node = node.parent
        return list(reversed(path))


class BatchWeightedAStar:
    """
    Batch Weighted A* Search (BWAS) for optimal Rubik's Cube solving
    
    Uses the trained neural network as a heuristic to guide search.
    Processes nodes in batches for efficient GPU utilization.
    """
    
    def __init__(self,
                 net: DeepCubeANet,
                 device: torch.device,
                 batch_size: int = 10000,
                 weight: float = 0.6):
        """
        Args:
            net: Trained DeepCubeA network
            device: Torch device
            batch_size: Number of nodes to expand per batch
            weight: Weight for weighted A* (0 = greedy, 1 = standard A*)
        """
        self.net = net
        self.device = device
        self.batch_size = batch_size
        self.weight = weight
        self.cube = RubiksCube3x3()
    
    def get_heuristic_batch(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Get heuristic values for batch of states using neural network
        
        Args:
            states: List of cube states
            
        Returns:
            Array of heuristic values
        """
        if len(states) == 0:
            return np.array([])
        
        # Convert to one-hot
        states_onehot = np.array([
            self.cube.get_state_one_hot(s) for s in states
        ], dtype=np.float32)
        
        # Get network predictions
        self.net.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states_onehot).to(self.device)
            values, _ = self.net(states_tensor)
        
        return values.cpu().numpy().flatten()
    
    def solve(self,
              initial_state: np.ndarray,
              max_nodes: int = 1000000,
              time_limit: float = 60.0,
              verbose: bool = True) -> Optional[List[int]]:
        """
        Solve the cube using Batch Weighted A* Search
        
        Args:
            initial_state: Initial cube state
            max_nodes: Maximum nodes to expand
            time_limit: Time limit in seconds
            verbose: Print progress
            
        Returns:
            List of moves to solve, or None if not found
        """
        if self.cube.is_solved(initial_state):
            if verbose:
                print("Cube is already solved!")
            return []
        
        start_time = time.time()
        
        # Initialize search
        h_initial = self.get_heuristic_batch([initial_state])[0]
        start_node = SearchNode(initial_state, 0, h_initial * self.weight)
        
        # Priority queue (min heap) and visited set
        open_set = [start_node]
        heapq.heapify(open_set)
        
        closed_set: Set[int] = set()
        node_count = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print("Batch Weighted A* Search")
            print(f"{'='*60}")
            print(f"Initial heuristic: {h_initial:.2f}")
            print(f"Weight: {self.weight}")
            print(f"Batch size: {self.batch_size}")
            print(f"{'='*60}\n")
        
        while open_set and node_count < max_nodes:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                if verbose:
                    print(f"\nTime limit ({time_limit}s) exceeded.")
                return None
            
            # Extract batch of best nodes
            batch_nodes = []
            while open_set and len(batch_nodes) < self.batch_size:
                node = heapq.heappop(open_set)
                
                # Skip if already visited
                if node.state_hash in closed_set:
                    continue
                
                closed_set.add(node.state_hash)
                batch_nodes.append(node)
                node_count += 1
            
            if not batch_nodes:
                continue
            
            # Generate all children for batch
            all_children = []
            child_parents = []
            child_moves = []
            child_g_costs = []
            
            for node in batch_nodes:
                for move_idx in range(self.cube.N_MOVES):
                    child_state = self.cube.apply_move_to_state(node.state, move_idx)
                    child_hash = hash(child_state.tobytes())
                    
                    # Skip if already visited
                    if child_hash in closed_set:
                        continue
                    
                    # Check if solved
                    if self.cube.is_solved(child_state):
                        # Found solution!
                        solution = node.get_path() + [move_idx]
                        
                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"\n{'='*60}")
                            print("SOLUTION FOUND!")
                            print(f"{'='*60}")
                            print(f"Moves: {len(solution)}")
                            print(f"Nodes expanded: {node_count}")
                            print(f"Time: {elapsed:.2f}s")
                            print(f"Solution: {' '.join(self.cube.MOVES[m] for m in solution)}")
                            print(f"{'='*60}")
                        
                        return solution
                    
                    all_children.append(child_state)
                    child_parents.append(node)
                    child_moves.append(move_idx)
                    child_g_costs.append(node.g_cost + 1)
            
            # Get heuristics for all children in batch
            if all_children:
                h_values = self.get_heuristic_batch(all_children)
                
                # Create child nodes and add to open set
                for i, child_state in enumerate(all_children):
                    child_node = SearchNode(
                        state=child_state,
                        g_cost=child_g_costs[i],
                        h_cost=h_values[i] * self.weight,
                        parent=child_parents[i],
                        move=child_moves[i]
                    )
                    heapq.heappush(open_set, child_node)
            
            # Progress update
            if verbose and node_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Nodes: {node_count}, Open: {len(open_set)}, "
                      f"Time: {elapsed:.1f}s")
        
        if verbose:
            print(f"\nSearch exhausted after {node_count} nodes.")
        return None


# =============================================================================
# COMPLETE SOLVER API
# =============================================================================

class DeepCubeASolver:
    """
    Complete DeepCubeA Solver API
    
    Combines training and solving into a simple interface.
    """
    
    def __init__(self, 
                 model_path: str = 'deepcubea_model.pth',
                 device: str = 'auto'):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_path = model_path
        self.cube = RubiksCube3x3()
        self.net = None
        self.trainer = None
        self.searcher = None
        
        print(f"DeepCubeA Solver initialized on {self.device}")
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.net = DeepCubeANet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.net.eval()
        
        self.searcher = BatchWeightedAStar(
            net=self.net,
            device=self.device
        )
        
        print(f"Model loaded from {model_path}")
    
    def train(self,
              n_iterations: int = 10000,
              batch_size: int = 1000,
              scramble_depth: int = 30,
              save_freq: int = 1000,
              continue_training: bool = True):
        """
        Train the DeepCubeA model
        
        Args:
            n_iterations: Number of training iterations
            batch_size: Batch size per iteration
            scramble_depth: Maximum scramble depth
            save_freq: How often to save model
            continue_training: Whether to continue from existing model
        """
        self.trainer = AutodidacticIteration(device=str(self.device))
        
        if continue_training and os.path.exists(self.model_path):
            try:
                self.trainer.load(self.model_path)
                print("Continuing training from existing model")
            except:
                print("Could not load model, starting fresh")
        
        self.trainer.train(
            n_iterations=n_iterations,
            batch_size=batch_size,
            scramble_depth=scramble_depth,
            save_freq=save_freq,
            model_path=self.model_path
        )
        
        # Update solver with trained model
        self.net = self.trainer.net
        self.searcher = BatchWeightedAStar(
            net=self.net,
            device=self.device
        )
    
    def solve(self,
              state: Optional[np.ndarray] = None,
              scramble_moves: int = 0,
              max_nodes: int = 1000000,
              time_limit: float = 60.0,
              verbose: bool = True) -> Optional[List[str]]:
        """
        Solve a Rubik's cube state
        
        Args:
            state: Cube state (54-element array). If None, scrambles a new cube.
            scramble_moves: If state is None, scramble with this many moves
            max_nodes: Maximum nodes to expand in search
            time_limit: Time limit in seconds
            verbose: Print progress
            
        Returns:
            List of move strings (e.g., ["U", "R'", "F"]) or None if not solved
        """
        if self.net is None:
            print("No model loaded! Train or load a model first.")
            return None
        
        if state is None:
            state, _ = self.cube.scramble(scramble_moves)
            if verbose:
                print(f"Scrambled with {scramble_moves} moves")
        
        # Run A* search
        solution_moves = self.searcher.solve(
            initial_state=state,
            max_nodes=max_nodes,
            time_limit=time_limit,
            verbose=verbose
        )
        
        if solution_moves is not None:
            # Convert move indices to strings
            return [self.cube.MOVES[m] for m in solution_moves]
        
        return None
    
    def convert_cv_state(self, cv_cube_state: Dict) -> np.ndarray:
        """
        Convert CV solver cube state to DeepCubeA format
        
        Args:
            cv_cube_state: Dict with keys 'F', 'R', 'B', 'L', 'U', 'D'
                          Each containing 3x3 array of color names
                          
        Returns:
            54-element numpy array
        """
        color_to_idx = {
            'white': 0, 'W': 0, 'w': 0,
            'yellow': 1, 'Y': 1, 'y': 1,
            'red': 2, 'R': 2, 'r': 2,
            'orange': 3, 'O': 3, 'o': 3,
            'green': 4, 'G': 4, 'g': 4,
            'blue': 5, 'B': 5, 'b': 5
        }
        
        face_map = ['U', 'D', 'F', 'B', 'R', 'L']
        state = np.zeros(54, dtype=np.int8)
        
        for face_idx, face_name in enumerate(face_map):
            if face_name not in cv_cube_state:
                continue
            
            cv_colors = cv_cube_state[face_name]
            for i in range(3):
                for j in range(3):
                    color = cv_colors[i][j] if isinstance(cv_colors[i][j], str) else cv_colors[i, j]
                    color_idx = color_to_idx.get(color, 0)
                    sticker_idx = face_idx * 9 + i * 3 + j
                    state[sticker_idx] = color_idx
        
        return state
    
    def test(self, n_tests: int = 10, scramble_depth: int = 10, verbose: bool = True):
        """
        Test the solver on random scrambles
        
        Args:
            n_tests: Number of tests
            scramble_depth: Scramble depth
            verbose: Print details
        """
        if self.net is None:
            print("No model loaded! Train or load a model first.")
            return
        
        print(f"\n{'='*60}")
        print(f"Testing DeepCubeA on {n_tests} scrambles (depth {scramble_depth})")
        print(f"{'='*60}")
        
        solved = 0
        total_moves = 0
        total_time = 0
        
        for i in range(n_tests):
            print(f"\nTest {i+1}/{n_tests}")
            
            # Scramble
            state, optimal = self.cube.scramble(scramble_depth)
            
            # Solve
            start = time.time()
            solution = self.solve(state=state, verbose=verbose)
            elapsed = time.time() - start
            
            if solution:
                solved += 1
                total_moves += len(solution)
                total_time += elapsed
                print(f"✓ Solved in {len(solution)} moves, {elapsed:.2f}s")
            else:
                print(f"✗ Not solved")
        
        print(f"\n{'='*60}")
        print(f"Results: {solved}/{n_tests} solved ({solved/n_tests*100:.1f}%)")
        if solved > 0:
            print(f"Average moves: {total_moves/solved:.1f}")
            print(f"Average time: {total_time/solved:.2f}s")
        print(f"{'='*60}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DeepCubeA: Optimal Rubik\'s Cube Solver using Deep Learning + A* Search'
    )
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'solve'],
                       help='Mode: train, test, or solve')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Training iterations')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for training')
    parser.add_argument('--scramble', type=int, default=30,
                       help='Maximum scramble depth')
    parser.add_argument('--model', type=str, default='deepcubea_model.pth',
                       help='Model path')
    parser.add_argument('--tests', type=int, default=10,
                       help='Number of test cases')
    parser.add_argument('--time-limit', type=float, default=60.0,
                       help='Time limit for solving (seconds)')
    
    args = parser.parse_args()
    
    solver = DeepCubeASolver(model_path=args.model)
    
    if args.mode == 'train':
        print("\n" + "="*60)
        print("DEEPCUBEA TRAINING")
        print("="*60)
        solver.train(
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            scramble_depth=args.scramble
        )
        
    elif args.mode == 'test':
        print("\n" + "="*60)
        print("DEEPCUBEA TESTING")
        print("="*60)
        solver.test(
            n_tests=args.tests,
            scramble_depth=args.scramble
        )
        
    elif args.mode == 'solve':
        print("\n" + "="*60)
        print("DEEPCUBEA SOLVING")
        print("="*60)
        solution = solver.solve(
            scramble_moves=args.scramble,
            time_limit=args.time_limit
        )
        if solution:
            print(f"\nFinal solution: {' '.join(solution)}")


if __name__ == "__main__":
    main()
