"""
Training utilities (moved from top-level train_tactical.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from .env import TacticalUAVEnv
from .agent import QLearningAgent


def train_tactical_agent(episodes=10000, eval_frequency=100, output_dir="output"):
	env = TacticalUAVEnv()
	state_dims = [4, 4, 3, 3]
	action_dim = env.action_space.n
	agent = QLearningAgent(
		state_dims=state_dims,
		action_dim=action_dim,
		learning_rate=0.1,
		discount_factor=0.99,
		exploration_rate=1.0,
		exploration_decay=0.995,
		min_exploration_rate=0.01,
	)

	all_episode_rewards = []
	evaluation_rewards = []
	epsilon_values = []
	best_eval_reward = -float('inf')

	os.makedirs(output_dir, exist_ok=True)
	start_time = time.time()
	for episode in tqdm(range(episodes), desc="Training Tactical Agent"):
		state = env.reset()
		done = False
		episode_reward = 0
		while not done:
			action = agent.choose_action(state)
			next_state, reward, done, _ = env.step(action)
			agent.learn(state, action, reward, next_state, done)
			state = next_state
			episode_reward += reward
		all_episode_rewards.append(episode_reward)
		epsilon_values.append(agent.epsilon)
		agent.training_episodes += 1
		if (episode + 1) % eval_frequency == 0:
			eval_reward = evaluate_agent(agent, env, episodes=10)
			evaluation_rewards.append(eval_reward)
			if eval_reward > best_eval_reward:
				best_eval_reward = eval_reward
				agent.save_policy(f"{output_dir}/tactical_q_table_best.npy")
			print(f"Episode {episode+1}/{episodes}, Avg Reward: {episode_reward:.2f}, "
				  f"Eval Reward: {eval_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

	training_time = time.time() - start_time
	print(f"Training completed in {training_time:.2f} seconds")
	agent.save_policy(f"{output_dir}/tactical_q_table.npy")
	plot_training_curves(all_episode_rewards, evaluation_rewards, epsilon_values, output_dir)
	return agent


def evaluate_agent(agent, env, episodes=10):
	total_reward = 0
	for _ in range(episodes):
		state = env.reset()
		done = False
		episode_reward = 0
		while not done:
			action = agent.choose_action(state, training=False)
			next_state, reward, done, _ = env.step(action)
			state = next_state
			episode_reward += reward
		total_reward += episode_reward
	return total_reward / episodes


def plot_training_curves(rewards, eval_rewards, epsilons, output_dir):
	plt.figure(figsize=(15, 15))
	plt.subplot(3, 1, 1)
	plt.plot(rewards)
	plt.title('Episode Rewards')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.subplot(3, 1, 2)
	window_size = 100
	smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
	plt.plot(smoothed_rewards)
	plt.title(f'Smoothed Rewards (Window Size: {window_size})')
	plt.xlabel('Episode')
	plt.ylabel('Smoothed Reward')
	plt.subplot(3, 1, 3)
	if eval_rewards:
		eval_episodes = np.arange(0, len(rewards), max(1, len(rewards)//max(1,len(eval_rewards))))[:len(eval_rewards)]
		plt.plot(eval_episodes, eval_rewards, 'r-')
	plt.title('Evaluation Rewards')
	plt.xlabel('Episode')
	plt.ylabel('Evaluation Reward')
	plt.tight_layout()
	plt.savefig(f"{output_dir}/tactical_training_curves.png")
	plt.close()
	plt.figure(figsize=(10, 5))
	plt.plot(epsilons)
	plt.title('Exploration Rate (Epsilon)')
	plt.xlabel('Episode')
	plt.ylabel('Epsilon')
	plt.savefig(f"{output_dir}/tactical_epsilon.png")
	plt.close()

__all__ = ["train_tactical_agent", "evaluate_agent"]
