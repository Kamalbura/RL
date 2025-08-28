"""
Training utilities (moved from top-level train_tactical.py)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Add utils to path for reproducibility
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.reproducibility import setup_tactical_training_metadata, add_model_hashes

from .env import TacticalUAVEnv
from .agent import TacticalAgent, unflatten_action


def train_tactical_agent(episodes=10000, eval_frequency=100, output_dir="output", seed: int | None = 123, checkpoint_every: int = 500):
	"""Train the tactical (UAV-side) DQN agent on continuous state and tuple actions.

	Adds:
		- Deterministic seeding
		- CSV logging of rewards / eval scores
		- Periodic checkpointing
	"""
	if seed is not None:
		np.random.seed(seed)

	# Setup reproducibility metadata
	metadata = setup_tactical_training_metadata(episodes, seed, output_dir)
	print(f"Training metadata saved to {output_dir}/run_metadata.json")

	env = TacticalUAVEnv()
	agent = TacticalAgent(state_dim=int(env.observation_space.shape[0]), action_dim=12)

	all_episode_rewards: list[float] = []
	evaluation_rewards: list[float] = []
	epsilon_values: list[float] = []
	best_eval_reward = -float('inf')

	os.makedirs(output_dir, exist_ok=True)
	log_csv = os.path.join(output_dir, "tactical_training_log.csv")
	with open(log_csv, "w", encoding="utf-8") as f:
		f.write("episode,reward,eval_reward,epsilon\n")

	start_time = time.time()
	for episode in tqdm(range(episodes), desc="Training Tactical Agent"):
		state = env.reset()
		done = False
		episode_reward = 0.0
		while not done:
			action_idx = agent.choose_action(state, training=True)
			action = unflatten_action(action_idx)
			next_state, reward, done, _ = env.step(action)
			agent.remember(state, action_idx, reward, next_state, done)
			agent.learn()
			state = next_state
			episode_reward += reward
		all_episode_rewards.append(episode_reward)
		epsilon_values.append(agent.epsilon)
		# progress tracking handled via logs

		eval_reward_str = ""
		if (episode + 1) % eval_frequency == 0:
			eval_reward = evaluate_agent(agent, env, episodes=10)
			evaluation_rewards.append(eval_reward)
			eval_reward_str = f"{eval_reward:.4f}"
			if eval_reward > best_eval_reward:
				best_eval_reward = eval_reward
				agent.save_policy(f"{output_dir}/tactical_dqn_best.pt")
			print(
				f"Episode {episode+1}/{episodes} | EpReward {episode_reward:.2f} | Eval {eval_reward:.2f} | Eps {agent.epsilon:.4f}"
			)

		# Periodic checkpoint independent of eval frequency
		if (episode + 1) % checkpoint_every == 0:
			agent.save_policy(f"{output_dir}/tactical_dqn_ckpt_{episode+1}.pt")

		with open(log_csv, "a", encoding="utf-8") as f:
			f.write(f"{episode+1},{episode_reward:.4f},{eval_reward_str},{agent.epsilon:.6f}\n")

	training_time = time.time() - start_time
	print(f"Training completed in {training_time:.2f} seconds")
	agent.save_policy(f"{output_dir}/tactical_dqn.pt")
	
	# Add model hashes to metadata
	model_files = {
		"tactical_dqn": f"{output_dir}/tactical_dqn.pt",
		"tactical_dqn_best": f"{output_dir}/tactical_dqn_best.pt"
	}
	add_model_hashes(f"{output_dir}/run_metadata.json", model_files)
	
	plot_training_curves(all_episode_rewards, evaluation_rewards, epsilon_values, output_dir)
	return agent


def evaluate_agent(agent, env, episodes=10):
	total_reward = 0
	for _ in range(episodes):
		state = env.reset()
		done = False
		episode_reward = 0
		while not done:
			action_idx = agent.choose_action(state, training=False)
			action = unflatten_action(action_idx)
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
