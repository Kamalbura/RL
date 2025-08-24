import os
from ddos_rl.train import train_tactical_agent, evaluate_agent

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    agent = train_tactical_agent(episodes=10000, output_dir="output")
    summary = agent.get_policy_summary()
    print("\nPolicy Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")