

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from tqdm import tqdm
import pickle
import os

# Experience replay memory structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class RubiksCubeEnv:
    """
    Rubik's Cube Environment for Reinforcement Learning
    Simplified 2x2x2 cube for faster training (3x3x3 has too large state space)
    """
    
    def __init__(self, cube_size=2):
        self.cube_size = cube_size
        self.n_actions = 6  # U, D, L, R, F, B (only 90° clockwise for simplicity)
        self.action_names = ['U', 'D', 'L', 'R', 'F', 'B']
        self.reset()
    
    def reset(self):
        """Reset to solved state"""
        # Each face has cube_size^2 stickers
        # 0=White, 1=Yellow, 2=Red, 3=Orange, 4=Green, 5=Blue
        size = self.cube_size
        self.state = {
            'U': np.full((size, size), 0, dtype=np.int8),  # White (Up)
            'D': np.full((size, size), 1, dtype=np.int8),  # Yellow (Down)
            'F': np.full((size, size), 2, dtype=np.int8),  # Red (Front)
            'B': np.full((size, size), 3, dtype=np.int8),  # Orange (Back)
            'R': np.full((size, size), 4, dtype=np.int8),  # Green (Right)
            'L': np.full((size, size), 5, dtype=np.int8),  # Blue (Left)
        }
        return self.get_state_vector()
    
    def get_state_vector(self):
        """Convert cube state to neural network input (one-hot encoded)"""
        # Flatten all faces into a single vector
        state_list = []
        for face in ['U', 'D', 'F', 'B', 'R', 'L']:
            state_list.append(self.state[face].flatten())
        
        flat_state = np.concatenate(state_list)
        
        # One-hot encode (6 colors)
        one_hot = np.zeros((len(flat_state), 6))
        one_hot[np.arange(len(flat_state)), flat_state] = 1
        
        return one_hot.flatten().astype(np.float32)
    
    def apply_move(self, action):
        """Apply a move to the cube"""
        move_name = self.action_names[action]
        
        if move_name == 'U':
            self._rotate_U()
        elif move_name == 'D':
            self._rotate_D()
        elif move_name == 'L':
            self._rotate_L()
        elif move_name == 'R':
            self._rotate_R()
        elif move_name == 'F':
            self._rotate_F()
        elif move_name == 'B':
            self._rotate_B()
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        self.apply_move(action)
        
        reward = self.get_reward()
        done = self.is_solved()
        next_state = self.get_state_vector()
        
        return next_state, reward, done
    
    def get_reward(self):
        """Calculate reward based on cube state"""
        if self.is_solved():
            return 1.0  # Large reward for solving
        
        # Small reward for getting stickers on correct faces
        correct_stickers = 0
        total_stickers = 0
        
        for face_idx, face_name in enumerate(['U', 'D', 'F', 'B', 'R', 'L']):
            face = self.state[face_name]
            correct_stickers += np.sum(face == face_idx)
            total_stickers += face.size
        
        # Normalize to [-0.1, 0]
        progress = correct_stickers / total_stickers
        return (progress - 1.0) * 0.1
    
    def is_solved(self):
        """Check if cube is in solved state"""
        for face_idx, face_name in enumerate(['U', 'D', 'F', 'B', 'R', 'L']):
            if not np.all(self.state[face_name] == face_idx):
                return False
        return True
    
    def scramble(self, n_moves=10):
        """Scramble the cube with random moves"""
        for _ in range(n_moves):
            action = random.randint(0, self.n_actions - 1)
            self.apply_move(action)
        return self.get_state_vector()
    
    def copy(self):
        """Create a copy of the environment"""
        new_env = RubiksCubeEnv(self.cube_size)
        for face in ['U', 'D', 'F', 'B', 'R', 'L']:
            new_env.state[face] = self.state[face].copy()
        return new_env
    
    # Rotation implementations for 2x2x2 cube
    def _rotate_face_clockwise(self, face):
        """Rotate a face 90° clockwise"""
        self.state[face] = np.rot90(self.state[face], -1)
    
    def _rotate_U(self):
        """Rotate Up face"""
        self._rotate_face_clockwise('U')
        temp = self.state['F'][0, :].copy()
        self.state['F'][0, :] = self.state['R'][0, :]
        self.state['R'][0, :] = self.state['B'][0, :]
        self.state['B'][0, :] = self.state['L'][0, :]
        self.state['L'][0, :] = temp
    
    def _rotate_D(self):
        """Rotate Down face"""
        self._rotate_face_clockwise('D')
        temp = self.state['F'][-1, :].copy()
        self.state['F'][-1, :] = self.state['L'][-1, :]
        self.state['L'][-1, :] = self.state['B'][-1, :]
        self.state['B'][-1, :] = self.state['R'][-1, :]
        self.state['R'][-1, :] = temp
    
    def _rotate_F(self):
        """Rotate Front face"""
        self._rotate_face_clockwise('F')
        temp = self.state['U'][-1, :].copy()
        self.state['U'][-1, :] = np.flip(self.state['L'][:, -1])
        self.state['L'][:, -1] = self.state['D'][0, :]
        self.state['D'][0, :] = np.flip(self.state['R'][:, 0])
        self.state['R'][:, 0] = temp
    
    def _rotate_B(self):
        """Rotate Back face"""
        self._rotate_face_clockwise('B')
        temp = self.state['U'][0, :].copy()
        self.state['U'][0, :] = self.state['R'][:, -1]
        self.state['R'][:, -1] = np.flip(self.state['D'][-1, :])
        self.state['D'][-1, :] = self.state['L'][:, 0]
        self.state['L'][:, 0] = np.flip(temp)
    
    def _rotate_L(self):
        """Rotate Left face"""
        self._rotate_face_clockwise('L')
        temp = self.state['U'][:, 0].copy()
        self.state['U'][:, 0] = np.flip(self.state['B'][:, -1])
        self.state['B'][:, -1] = np.flip(self.state['D'][:, 0])
        self.state['D'][:, 0] = self.state['F'][:, 0]
        self.state['F'][:, 0] = temp
    
    def _rotate_R(self):
        """Rotate Right face"""
        self._rotate_face_clockwise('R')
        temp = self.state['U'][:, -1].copy()
        self.state['U'][:, -1] = self.state['F'][:, -1]
        self.state['F'][:, -1] = self.state['D'][:, -1]
        self.state['D'][:, -1] = np.flip(self.state['B'][:, 0])
        self.state['B'][:, 0] = np.flip(temp)


class DQN(nn.Module):
    """
    Deep Q-Network for Rubik's Cube
    Predicts Q-values (expected future rewards) for each action
    """
    
    def __init__(self, state_size, action_size, hidden_sizes=[512, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    """
    Experience Replay Memory for DQN
    Stores past experiences and samples random batches for training
    """
    
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        """Add experience to memory"""
        self.memory.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class RubiksDQNAgent:
    """
    Deep Q-Learning Agent for Rubik's Cube
    Uses neural network to learn solving policy through trial and error
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01     # Minimum exploration
        self.epsilon_decay = 0.995  # Exploration decay
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory_size = 10000
        
        # Neural networks (policy and target)
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_size)
        
        # Statistics
        self.training_step = 0
        self.losses = []
        self.rewards_history = []
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        With probability epsilon: random action (exploration)
        With probability 1-epsilon: best action according to Q-network (exploitation)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        """
        Perform one training step using experience replay
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model and training state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses,
            'rewards_history': self.rewards_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint.get('losses', [])
        self.rewards_history = checkpoint.get('rewards_history', [])
        print(f"Model loaded from {filepath}")


def train_agent(n_episodes=5000, max_steps=50, scramble_moves=5, 
                target_update_freq=10, save_freq=100, model_path='rubiks_dqn.pth'):
    """
    Train the DQN agent to solve Rubik's Cube
    
    Args:
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        scramble_moves: Number of moves to scramble cube
        target_update_freq: How often to update target network
        save_freq: How often to save model
        model_path: Path to save/load model
    """
    
    # Initialize environment and agent
    env = RubiksCubeEnv(cube_size=2)
    state_size = len(env.get_state_vector())
    action_size = env.n_actions
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = RubiksDQNAgent(state_size, action_size, device)
    
    # Load existing model if available
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print("Resuming training from saved model")
        except:
            print("Could not load model, starting fresh")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    solved_count = 0
    
    print(f"\nStarting training for {n_episodes} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Scramble depth: {scramble_moves} moves\n")
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Reset and scramble
        env.reset()
        state = env.scramble(scramble_moves)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.memory.push(Experience(state, action, reward, next_state, done))
            
            # Train
            loss = agent.train_step()
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                solved_count += 1
                break
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent.rewards_history.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            solve_rate = solved_count / 100
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Length: {avg_length:.2f} steps")
            print(f"  Solve Rate: {solve_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Memory Size: {len(agent.memory)}")
            
            solved_count = 0
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            agent.save(model_path)
    
    # Final save
    agent.save(model_path)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Total Training Steps: {agent.training_step}")
    print(f"Model saved to: {model_path}")
    print("="*50)
    
    return agent


def test_agent(model_path='rubiks_dqn.pth', n_tests=10, scramble_moves=5, max_steps=50):
    """
    Test the trained agent
    
    Args:
        model_path: Path to saved model
        n_tests: Number of test episodes
        scramble_moves: Scramble depth
        max_steps: Maximum solving steps
    """
    # Initialize
    env = RubiksCubeEnv(cube_size=2)
    state_size = len(env.get_state_vector())
    action_size = env.n_actions
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RubiksDQNAgent(state_size, action_size, device)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    print(f"\nTesting agent on {n_tests} scrambled cubes...")
    print(f"Scramble depth: {scramble_moves} moves\n")
    
    solved = 0
    total_steps = []
    
    for test in range(n_tests):
        env.reset()
        initial_state = env.scramble(scramble_moves)
        
        print(f"\nTest {test + 1}/{n_tests}")
        print("Initial scramble:", end=" ")
        
        state = initial_state
        moves = []
        
        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            move_name = env.action_names[action]
            moves.append(move_name)
            
            next_state, reward, done = env.step(action)
            state = next_state
            
            if done:
                solved += 1
                total_steps.append(step + 1)
                print(f"\n✓ SOLVED in {step + 1} steps!")
                print(f"Solution: {' '.join(moves)}")
                break
        else:
            print(f"\n✗ Not solved within {max_steps} steps")
            print(f"Attempted: {' '.join(moves)}")
    
    print("\n" + "="*50)
    print(f"Results: {solved}/{n_tests} solved ({solved/n_tests*100:.1f}%)")
    if total_steps:
        print(f"Average steps: {np.mean(total_steps):.2f}")
        print(f"Min steps: {min(total_steps)}")
        print(f"Max steps: {max(total_steps)}")
    print("="*50)


def solve_with_ml(cube_state, model_path='rubiks_dqn.pth', max_steps=50):
    """
    Solve a given cube state using trained ML model
    
    Args:
        cube_state: Dictionary with cube state (same format as your cv_solver)
        model_path: Path to trained model
        max_steps: Maximum solving steps
        
    Returns:
        List of moves to solve cube, or None if couldn't solve
    """
    # Initialize
    env = RubiksCubeEnv(cube_size=2)
    state_size = len(env.get_state_vector())
    action_size = env.n_actions
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RubiksDQNAgent(state_size, action_size, device)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found! Train the model first.")
        return None
    
    agent.load(model_path)
    agent.epsilon = 0.0
    
    # Convert cube_state to environment format
    # (You'll need to adapt this based on your cube_state format)
    env.reset()
    # TODO: Set env.state based on cube_state
    
    state = env.get_state_vector()
    moves = []
    
    for step in range(max_steps):
        action = agent.select_action(state, training=False)
        move_name = env.action_names[action]
        moves.append(move_name)
        
        next_state, reward, done = env.step(action)
        state = next_state
        
        if done:
            print(f"✓ ML Solver found solution in {step + 1} moves!")
            return moves
    
    print(f"✗ ML Solver couldn't solve in {max_steps} moves")
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rubik\'s Cube Deep Q-Learning Solver')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--scramble', type=int, default=5,
                        help='Scramble depth (number of moves)')
    parser.add_argument('--model', type=str, default='rubiks_dqn.pth',
                        help='Model save/load path')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n" + "="*50)
        print("RUBIK'S CUBE DEEP Q-LEARNING TRAINER")
        print("="*50)
        train_agent(
            n_episodes=args.episodes,
            scramble_moves=args.scramble,
            model_path=args.model
        )
    else:
        print("\n" + "="*50)
        print("RUBIK'S CUBE ML SOLVER - TESTING")
        print("="*50)
        test_agent(
            model_path=args.model,
            scramble_moves=args.scramble
        )