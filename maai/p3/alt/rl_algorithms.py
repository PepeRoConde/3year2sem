import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
import matplotlib.pyplot as plt


class ReinforcementLearningAlgorithm(ABC):
    """
    Abstract base class for reinforcement learning algorithms.
    
    Attributes:
        env: The OpenAI Gym environment
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy policy
        alpha: Learning rate
        state_discretization: Number of bins for each state dimension
        action_discretization: Number of bins for discretizing actions
        q_table: Q-function represented as a table
        v_table: Value function represented as a table
        state_bins: Bins for discretizing states
        action_bins: Bins for discretizing actions
    """
    
    def __init__(self, env_name: str, gamma: float = 0.99, epsilon: float = 0.1, 
                 alpha: float = 0.1, state_discretization: List[int] = None, 
                 action_discretization: int = 10):
        """
        Initialize the RL algorithm.
        
        Args:
            env_name: Name of the Gym environment
            gamma: Discount factor
            epsilon: Exploration rate for epsilon-greedy policy
            alpha: Learning rate
            state_discretization: Number of bins for each state dimension
            action_discretization: Number of bins for discretizing actions
        """
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        # Set default state discretization if not provided
        if state_discretization is None:
            self.state_discretization = [10, 10, 10]  # Default for Pendulum (3 dimensions)
        else:
            self.state_discretization = state_discretization
        
        self.action_discretization = action_discretization
        
        # Initialize state and action bins
        self._init_state_bins()
        self._init_action_bins()
        
        # Initialize Q-table and V-table
        self.q_table = self._init_q_table()
        self.v_table = self._init_v_table()
        
        # Statistics tracking
        self.episode_rewards = []
    
    def _init_state_bins(self):
        """Initialize discretization bins for state space."""
        # For Pendulum-v1, state is [cos(theta), sin(theta), theta_dot]
        # cos(theta) and sin(theta) are in [-1, 1], theta_dot is in [-8, 8]
        self.state_bins = []
        state_low = [-1.0, -1.0, -8.0]
        state_high = [1.0, 1.0, 8.0]
        
        for i, (low, high, bins) in enumerate(zip(state_low, state_high, self.state_discretization)):
            self.state_bins.append(np.linspace(low, high, bins + 1))
    
    def _init_action_bins(self):
        """Initialize discretization bins for action space."""
        # For Pendulum-v1, action is a single value in [-2, 2]
        action_low = self.env.action_space.low[0]
        action_high = self.env.action_space.high[0]
        self.action_bins = np.linspace(action_low, action_high, self.action_discretization)
    
    def _init_q_table(self) -> np.ndarray:
        """Initialize the Q-table with zeros."""
        q_shape = tuple(self.state_discretization + [self.action_discretization])
        return np.zeros(q_shape)
    
    def _init_v_table(self) -> np.ndarray:
        """Initialize the V-table with zeros."""
        v_shape = tuple(self.state_discretization)
        return np.zeros(v_shape)
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize a continuous state into indices for the Q-table.
        
        Args:
            state: The continuous state vector
            
        Returns:
            Tuple of indices representing the discretized state
        """
        indices = []
        for i, (bins, s) in enumerate(zip(self.state_bins, state)):
            index = np.digitize(s, bins) - 1
            # Clip to valid range
            index = max(0, min(index, self.state_discretization[i] - 1))
            indices.append(index)
        return tuple(indices)
    
    def discretize_action(self, action: np.ndarray) -> int:
        """
        Discretize a continuous action into an index for the Q-table.
        
        Args:
            action: The continuous action value
            
        Returns:
            Index representing the discretized action
        """
        index = np.digitize(action[0], self.action_bins) - 1
        # Clip to valid range
        return max(0, min(index, self.action_discretization - 1))
    
    def continuous_action(self, action_idx: int) -> np.ndarray:
        """
        Convert a discrete action index to a continuous action.
        
        Args:
            action_idx: The discrete action index
            
        Returns:
            The continuous action value
        """
        return np.array([self.action_bins[action_idx]])
    
    def epsilon_greedy_policy(self, state: Tuple[int, ...]) -> int:
        """
        Epsilon-greedy policy for action selection.
        
        Args:
            state: The discretized state
            
        Returns:
            The selected action index
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_discretization)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.q_table[state])
    
    def update_policy(self):
        """Update the policy (V-table) based on the Q-table."""
        for state_indices in np.ndindex(self.v_table.shape):
            self.v_table[state_indices] = np.max(self.q_table[state_indices])
    
    def plot_learning_curve(self):
        """Plot the learning curve (episode rewards)."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title(f"Learning Curve for {self.__class__.__name__}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.show()
    
    @abstractmethod
    def train(self, n_episodes: int, max_steps: int = 200):
        """
        Train the RL algorithm.
        
        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
        """
        pass
    
    @abstractmethod
    def update_q_table(self, *args, **kwargs):
        """
        Update the Q-table based on the specific algorithm.
        Each algorithm will implement this differently.
        """
        pass
    
    def test(self, n_episodes: int = 5, render: bool = True):
        """
        Test the trained policy.
        
        Args:
            n_episodes: Number of test episodes
            render: Whether to render the environment
        """
        total_rewards = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                if render:
                    self.env.render()
                
                # Use the learned policy (no exploration)
                disc_state = self.discretize_state(state)
                action_idx = np.argmax(self.q_table[disc_state])
                action = self.continuous_action(action_idx)
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state
                total_reward += reward
            
            total_rewards.append(total_reward)
            print(f"Test Episode {episode+1}/{n_episodes}, Total Reward: {total_reward:.2f}")
        
        print(f"Average Test Reward: {np.mean(total_rewards):.2f}")
        return np.mean(total_rewards)


class MonteCarloRL(ReinforcementLearningAlgorithm):
    """Monte Carlo reinforcement learning algorithm."""
    
    def __init__(self, env_name: str, gamma: float = 0.99, epsilon: float = 0.1,
                 state_discretization: List[int] = None, action_discretization: int = 10):
        super().__init__(env_name, gamma, epsilon, alpha=None, 
                         state_discretization=state_discretization, 
                         action_discretization=action_discretization)
        # Monte Carlo doesn't use alpha (learning rate)
        
    def update_q_table(self, episode_buffer: List[Tuple]):
        """
        Update Q-table using the Monte Carlo approach.
        
        Args:
            episode_buffer: List of (state, action, reward) tuples from an episode
        """
        # Calculate returns for each step
        G = 0
        returns = {}  # Dictionary to track returns for each state-action pair
        
        # Process the episode backwards
        for t in range(len(episode_buffer) - 1, -1, -1):
            state, action, reward = episode_buffer[t]
            G = reward + self.gamma * G
            
            # First-visit MC: only update if we haven't seen this state-action before
            if (state, action) not in [(s, a) for s, a, _ in episode_buffer[:t]]:
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)
                
                # Update Q-value as the average of all returns
                self.q_table[state + (action,)] = np.mean(returns[(state, action)])
    
    def train(self, n_episodes: int, max_steps: int = 200):
        """
        Train the Monte Carlo algorithm.
        
        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
        """
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_buffer = []
            total_reward = 0
            
            for step in range(max_steps):
                # Discretize state
                disc_state = self.discretize_state(state)
                
                # Select action using epsilon-greedy policy
                action_idx = self.epsilon_greedy_policy(disc_state)
                action = self.continuous_action(action_idx)
                
                # Take action
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                # Store the experience
                episode_buffer.append((disc_state, action_idx, reward))
                
                # Update state
                state = next_state
                
                if done or truncated:
                    break
            
            # Update Q-table using the episode experience
            self.update_q_table(episode_buffer)
            
            # Update policy
            self.update_policy()
            
            # Track progress
            self.episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward:.2f}")
        
        return self.q_table, self.v_table


class SarsaRL(ReinforcementLearningAlgorithm):
    """SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm."""
    
    def update_q_table(self, state: Tuple[int, ...], action: int, 
                      reward: float, next_state: Tuple[int, ...], next_action: int):
        """
        Update Q-table using the SARSA approach.
        
        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
            next_action: Next action
        """
        # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        current_q = self.q_table[state + (action,)]
        next_q = self.q_table[next_state + (next_action,)]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        
        self.q_table[state + (action,)] = current_q + self.alpha * td_error
    
    def train(self, n_episodes: int, max_steps: int = 200):
        """
        Train the SARSA algorithm.
        
        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
        """
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            disc_state = self.discretize_state(state)
            action_idx = self.epsilon_greedy_policy(disc_state)
            
            total_reward = 0
            
            for step in range(max_steps):
                # Take action
                action = self.continuous_action(action_idx)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                # Discretize next state
                next_disc_state = self.discretize_state(next_state)
                
                # Choose next action using epsilon-greedy
                next_action_idx = self.epsilon_greedy_policy(next_disc_state)
                
                # Update Q-table
                self.update_q_table(disc_state, action_idx, reward, next_disc_state, next_action_idx)
                
                # Update state and action
                disc_state = next_disc_state
                action_idx = next_action_idx
                
                if done or truncated:
                    break
            
            # Update policy
            self.update_policy()
            
            # Track progress
            self.episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward:.2f}")
        
        return self.q_table, self.v_table


class QLearningRL(ReinforcementLearningAlgorithm):
    """Q-Learning reinforcement learning algorithm."""
    
    def update_q_table(self, state: Tuple[int, ...], action: int, 
                      reward: float, next_state: Tuple[int, ...]):
        """
        Update Q-table using the Q-Learning approach.
        
        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
        """
        # Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = self.q_table[state + (action,)]
        max_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        self.q_table[state + (action,)] = current_q + self.alpha * td_error
    
    def train(self, n_episodes: int, max_steps: int = 200):
        """
        Train the Q-Learning algorithm.
        
        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
        """
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Discretize state
                disc_state = self.discretize_state(state)
                
                # Select action using epsilon-greedy policy
                action_idx = self.epsilon_greedy_policy(disc_state)
                action = self.continuous_action(action_idx)
                
                # Take action
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                # Discretize next state
                next_disc_state = self.discretize_state(next_state)
                
                # Update Q-table
                self.update_q_table(disc_state, action_idx, reward, next_disc_state)
                
                # Update state
                state = next_state
                
                if done or truncated:
                    break
            
            # Update policy
            self.update_policy()
            
            # Track progress
            self.episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward:.2f}")
        
        return self.q_table, self.v_table


# Example usage
if __name__ == "__main__":
    # Create and train the Monte Carlo agent
    mc_agent = MonteCarloRL(env_name="Pendulum-v1", gamma=0.99, epsilon=0.1)
    mc_agent.train(n_episodes=1000)
    mc_agent.plot_learning_curve()
    mc_agent.test(n_episodes=5)
    
    # Create and train the SARSA agent
    sarsa_agent = SarsaRL(env_name="Pendulum-v1", gamma=0.99, epsilon=0.1, alpha=0.1)
    sarsa_agent.train(n_episodes=1000)
    sarsa_agent.plot_learning_curve()
    sarsa_agent.test(n_episodes=5)
    
    # Create and train the Q-Learning agent
    q_learning_agent = QLearningRL(env_name="Pendulum-v1", gamma=0.99, epsilon=0.1, alpha=0.1)
    q_learning_agent.train(n_episodes=1000)
    q_learning_agent.plot_learning_curve()
    q_learning_agent.test(n_episodes=5)
