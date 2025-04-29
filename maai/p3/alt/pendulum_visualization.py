import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
import time
from typing import List, Optional


class PendulumVisualizer:
    """
    Visualization tools for the Pendulum-v1 environment.
    """
    
    @staticmethod
    def display_frames_as_gif(frames: List[np.ndarray], filename: Optional[str] = None, 
                             fps: int = 30, display_in_notebook: bool = True):
        """
        Displays a list of frames as an animated gif.
        
        Args:
            frames: List of RGB arrays from gym environment
            filename: If specified, saves the animation to this file
            fps: Frames per second
            display_in_notebook: Whether to display the animation in the notebook
        
        Returns:
            Animation object if display_in_notebook is True, else None
        """
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        plt.tight_layout()
        
        def animate(i):
            patch.set_data(frames[i])
            return [patch]
            
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1000/fps)
        
        if filename:
            anim.save(filename, writer='pillow', fps=fps)
            
        if display_in_notebook:
            return HTML(anim.to_jshtml())
        else:
            plt.close()
            return None
    
    @staticmethod
    def visualize_agent(agent, max_steps: int = 200, n_episodes: int = 1, 
                       fps: int = 30, save_filename: Optional[str] = None):
        """
        Visualize an agent in the Pendulum-v1 environment.
        
        Args:
            agent: The RL agent to visualize
            max_steps: Maximum steps per episode
            n_episodes: Number of episodes to visualize
            fps: Frames per second
            save_filename: If specified, saves the animation to this file
        """
        all_frames = []
        total_rewards = []
        
        # Enable rendering to RGB array
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            frames = []
            frames.append(env.render())
            
            total_reward = 0
            done = False
            truncated = False
            
            for step in range(max_steps):
                # Get discretized state
                disc_state = agent.discretize_state(state)
                
                # Select action using the learned policy (no exploration)
                action_idx = np.argmax(agent.q_table[disc_state])
                action = agent.continuous_action(action_idx)
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Render and store frame
                frames.append(env.render())
                
                state = next_state
                total_reward += reward
                
                if done or truncated:
                    break
            
            all_frames.extend(frames)
            total_rewards.append(total_reward)
            print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward:.2f}")
            
            # Add some blank frames between episodes
            if episode < n_episodes - 1:
                for _ in range(10):
                    all_frames.append(np.ones_like(frames[0]) * 255)
        
        env.close()
        
        # Display the animation
        return PendulumVisualizer.display_frames_as_gif(
            all_frames, filename=save_filename, fps=fps)
    
    @staticmethod
    def compare_agents(agents: List, labels: Optional[List[str]] = None, max_steps: int = 200, 
                      fps: int = 30, save_filename: Optional[str] = None):
        """
        Compare multiple agents in the Pendulum-v1 environment side by side.
        
        Args:
            agents: List of RL agents to compare
            labels: Labels for each agent (if None, uses class names)
            max_steps: Maximum steps per episode
            fps: Frames per second
            save_filename: If specified, saves the animation to this file
        """
        if labels is None:
            labels = [agent.__class__.__name__ for agent in agents]
        
        all_frames = []
        total_rewards = {label: 0 for label in labels}
        
        # Create environments for each agent
        envs = [gym.make("Pendulum-v1", render_mode="rgb_array") for _ in agents]
        
        # Reset all environments
        states = [env.reset()[0] for env in envs]
        
        # Get initial frames
        frames = [env.render() for env in envs]
        
        # Combine frames side by side
        combined_frame = np.hstack(frames)
        all_frames.append(combined_frame)
        
        done = [False] * len(agents)
        truncated = [False] * len(agents)
        
        for step in range(max_steps):
            new_frames = []
            
            for i, (agent, env, state) in enumerate(zip(agents, envs, states)):
                if not (done[i] or truncated[i]):
                    # Get discretized state
                    disc_state = agent.discretize_state(state)
                    
                    # Select action using the learned policy
                    action_idx = np.argmax(agent.q_table[disc_state])
                    action = agent.continuous_action(action_idx)
                    
                    # Take action
                    next_state, reward, done[i], truncated[i], _ = env.step(action)
                    
                    states[i] = next_state
                    total_rewards[labels[i]] += reward
                
                # Render and store frame
                new_frames.append(env.render())
            
            # Combine frames side by side
            combined_frame = np.hstack(new_frames)
            all_frames.append(combined_frame)
            
            # Add agent labels to the combined frame
            if step == 0:
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.imshow(combined_frame)
                
                # Add labels
                for i, label in enumerate(labels):
                    x_center = combined_frame.shape[1] // len(labels) * (i + 0.5)
                    ax.text(x_center, 20, label, color='white', fontsize=16, 
                           horizontalalignment='center', verticalalignment='top',
                           bbox=dict(facecolor='black', alpha=0.7))
                
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('agent_labels.png', bbox_inches='tight', pad_inches=0)
                plt.close()
            
            if all(d or t for d, t in zip(done, truncated)):
                break
        
        # Close all environments
        for env in envs:
            env.close()
        
        # Display rewards
        for label, reward in total_rewards.items():
            print(f"{label} Total Reward: {reward:.2f}")
        
        # Display the animation
        return PendulumVisualizer.display_frames_as_gif(
            all_frames, filename=save_filename, fps=fps)


# Example integration with RL algorithms
def run_pendulum_demo():
    """Run a demo of the pendulum visualization with multiple RL algorithms."""
    # Import needed only if running this as a standalone script
    from rl_algorithms import MonteCarloRL, SarsaRL, QLearningRL
    
    # Train agents with fewer episodes for quick demonstration
    mc_agent = MonteCarloRL(env_name="Pendulum-v1", gamma=0.99, epsilon=0.1)
    mc_agent.train(n_episodes=300)
    
    sarsa_agent = SarsaRL(env_name="Pendulum-v1", gamma=0.99, epsilon=0.1, alpha=0.1)
    sarsa_agent.train(n_episodes=300)
    
    q_learning_agent = QLearningRL(env_name="Pendulum-v1", gamma=0.99, epsilon=0.1, alpha=0.1)
    q_learning_agent.train(n_episodes=300)
    
    # Visualize a single agent
    print("Visualizing Monte Carlo agent:")
    PendulumVisualizer.visualize_agent(mc_agent, max_steps=200, n_episodes=1)
    
    # Compare agents
    print("\nComparing all agents:")
    agents = [mc_agent, sarsa_agent, q_learning_agent]
    labels = ["Monte Carlo", "SARSA", "Q-Learning"]
    PendulumVisualizer.compare_agents(agents, labels, max_steps=200)


if __name__ == "__main__":
    run_pendulum_demo()