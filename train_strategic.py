"""
Training script for the Strategic Swarm Agent

This script trains a Q-learning agent on the strategic swarm environment,
which makes fleet-wide decisions about cryptographic policies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from strategic_simulator import StrategicSwarmEnv
from rl_agent import QLearningAgent

def train_strategic_agent(episodes=10000, eval_frequency=100, output_dir="output"):
    """
    Train the strategic agent
    
    Args:
        episodes: Number of training episodes
        eval_frequency: Frequency of evaluation
        output_dir: Directory to save outputs
    
    Returns:
        agent: Trained agent
    """
    # Create the environment
    env = StrategicSwarmEnv(num_drones=5)  # Simulate a swarm of 5 drones
    
    # Create the agent
    state_dims = [3, 3, 4]  # Swarm Threat, Fleet Battery, Mission Phase
    action_dim = 4  # 4 cryptographic algorithms
    agent = QLearningAgent(
        state_dims=state_dims,
        action_dim=action_dim,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01
    )
    
    # Training metrics
    all_episode_rewards = []
    evaluation_rewards = []
    epsilon_values = []
    best_eval_reward = -float('inf')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    start_time = time.time()
    for episode in tqdm(range(episodes), desc="Training Strategic Agent"):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Let agent learn
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Store metrics
        all_episode_rewards.append(episode_reward)
        epsilon_values.append(agent.epsilon)
        agent.training_episodes += 1
        
        # Evaluate periodically
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate_agent(agent, env, episodes=10)
            evaluation_rewards.append(eval_reward)
            
            # Save if best
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_policy(f"{output_dir}/strategic_q_table_best.npy")
            
            # Log progress
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {episode_reward:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final policy
    agent.save_policy(f"{output_dir}/strategic_q_table.npy")
    
    # Plot and save training curves
    plot_training_curves(all_episode_rewards, evaluation_rewards, epsilon_values, output_dir)
    
    return agent

def evaluate_agent(agent, env, episodes=10):
    """
    Evaluate the agent's performance
    
    Args:
        agent: RL agent
        env: Environment
        episodes: Number of evaluation episodes
    
    Returns:
        avg_reward: Average reward over evaluation episodes
    """
    total_reward = 0
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose action (greedy policy)
            action = agent.choose_action(state, training=False)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / episodes

def plot_training_curves(rewards, eval_rewards, epsilons, output_dir):
    """
    Plot and save training curves
    
    Args:
        rewards: List of episode rewards
        eval_rewards: List of evaluation rewards
        epsilons: List of epsilon values
        output_dir: Directory to save plots
    """
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Plot episode rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot smoothed rewards
    plt.subplot(3, 1, 2)
    window_size = 100
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    
    # Plot evaluation rewards
    plt.subplot(3, 1, 3)
    eval_episodes = np.arange(0, len(rewards), len(rewards)//len(eval_rewards))[:len(eval_rewards)]
    plt.plot(eval_episodes, eval_rewards, 'r-')
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategic_training_curves.png")
    plt.close()
    
    # Plot epsilon values
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.savefig(f"{output_dir}/strategic_epsilon.png")
    plt.close()

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Train the agent
    agent = train_strategic_agent(episodes=10000, output_dir="output")
    
    # Print policy summary
    summary = agent.get_policy_summary()
    print("\nPolicy Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")