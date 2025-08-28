import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_rl.crypto_simulator import CryptoEnv
from crypto_rl.rl_agent import CryptoDQNAgent

def run_training(episodes=15000, eval_frequency=300, output_dir='../output'):
    """
    Train the cryptographic RL agent
    
    Args:
        episodes (int): Number of training episodes
        eval_frequency (int): Frequency of evaluation episodes
        output_dir (str): Directory to save results
    """
    # Create the environment
    env = CryptoEnv()
    
    # Create the DQN agent
    agent = CryptoDQNAgent(state_dim=4, action_dim=4)
    
    # Lists to store training metrics
    all_episode_rewards = []
    evaluation_rewards = []
    epsilons = []
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training Crypto Agent"):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store and learn
            agent.remember(state, int(action), float(reward), next_state, bool(done))
            agent.learn()
            
            state = next_state
            episode_reward += reward
        
        # Store training metrics
        all_episode_rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        
        # Evaluate the agent periodically
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=10)
            evaluation_rewards.append(eval_reward)
            print(f"Episode: {episode+1}, Evaluation Reward: {eval_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final policy
    agent.save_policy(f"{output_dir}/crypto_dqn.pt")
    
    # Plot training progress
    plot_training_results(all_episode_rewards, evaluation_rewards, epsilons, output_dir)
    
    return agent

def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate the agent's performance without exploration
    
    Args:
        agent: The RL agent
        env: The environment
        num_episodes: Number of evaluation episodes
    
    Returns:
        float: Average reward over evaluation episodes
    """
    eval_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.choose_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)

def plot_training_results(all_rewards, eval_rewards, epsilons, output_dir):
    """Plot the training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(3, 1, 1)
    plt.plot(all_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot smoothed episode rewards
    plt.subplot(3, 1, 2)
    window_size = 100
    smoothed_rewards = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Episode Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')
    
    # Plot evaluation rewards
    eval_episodes = np.arange(0, len(all_rewards), len(all_rewards)//len(eval_rewards))[:len(eval_rewards)]
    plt.subplot(3, 1, 3)
    plt.plot(eval_episodes, eval_rewards, 'r-')
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Evaluation Reward')
    
    # Save the figure
    plt.savefig(f'{output_dir}/crypto_training_rewards.png')
    plt.close()
    
    # Plot epsilon values
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    # Save the figure
    plt.savefig(f'{output_dir}/crypto_training_epsilon.png')
    plt.close()

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the agent
    agent = run_training(episodes=15000, output_dir=output_dir)
    
    print("Training completed! Policy saved to output/crypto_q_table.npy")
