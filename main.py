#!/usr/bin/env python3
"""
Main Entry Point for Dual-Agent RL UAV System
Production-ready implementation with proper error handling and logging
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('uav_rl_system.log')
        ]
    )
    return logging.getLogger(__name__)

def train_tactical_agent(episodes=1000, save_path="tactical_policy.npy"):
    """Train tactical UAV agent"""
    logger = logging.getLogger(__name__)
    logger.info("Starting tactical agent training...")
    
    try:
        from ddos_rl.env import TacticalUAVEnv
        from ddos_rl.agent import QLearningAgent
        from utils.reproducibility import set_random_seeds
        
        # Set reproducible seeds
        set_random_seeds(42)
        
        # Initialize environment and agent
        env = TacticalUAVEnv()
        agent = QLearningAgent(
            state_dims=[4, 4, 3, 3],
            action_dim=9,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.9995,
            min_exploration_rate=0.01
        )
        
        logger.info(f"Training for {episodes} episodes...")
        
        # Training loop
        total_rewards = []
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                agent.update_q_value(state, action, reward, next_state)
                
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = sum(total_rewards[-100:]) / min(100, len(total_rewards))
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save trained policy
        agent.save_policy(save_path)
        logger.info(f"Tactical agent training complete. Policy saved to {save_path}")
        
        return agent, total_rewards
        
    except Exception as e:
        logger.error(f"Tactical training failed: {e}")
        raise

def train_strategic_agent(episodes=1000, save_path="strategic_policy.npy"):
    """Train strategic GCS agent"""
    logger = logging.getLogger(__name__)
    logger.info("Starting strategic agent training...")
    
    try:
        from crypto_rl.strategic_agent import StrategicCryptoEnv, QLearningAgent
        from utils.reproducibility import set_random_seeds
        
        # Set reproducible seeds
        set_random_seeds(42)
        
        # Initialize environment and agent
        env = StrategicCryptoEnv()
        agent = QLearningAgent(
            state_dims=[3, 3, 4],
            action_dim=4,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.9995,
            min_exploration_rate=0.01
        )
        
        logger.info(f"Training for {episodes} episodes...")
        
        # Training loop
        total_rewards = []
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                agent.update_q_value(state, action, reward, next_state)
                
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = sum(total_rewards[-100:]) / min(100, len(total_rewards))
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save trained policy
        agent.save_policy(save_path)
        logger.info(f"Strategic agent training complete. Policy saved to {save_path}")
        
        return agent, total_rewards
        
    except Exception as e:
        logger.error(f"Strategic training failed: {e}")
        raise

def deploy_system(tactical_policy=None, strategic_policy=None, uav_id="UAV_001", gcs_id="GCS_MAIN"):
    """Deploy the complete dual-agent system"""
    logger = logging.getLogger(__name__)
    logger.info("Deploying dual-agent RL UAV system...")
    
    try:
        from integration.system_coordinator import SystemCoordinator
        
        # Create system coordinator
        coordinator = SystemCoordinator(uav_id=uav_id, gcs_id=gcs_id)
        
        # Start system with trained policies
        coordinator.start(
            tactical_policy=tactical_policy,
            strategic_policy=strategic_policy
        )
        
        logger.info("System deployed successfully!")
        
        # Keep system running
        try:
            while True:
                status = coordinator.get_system_status()
                if not status['running']:
                    logger.error("System stopped unexpectedly")
                    break
                
                # Log status every 60 seconds
                import time
                time.sleep(60)
                logger.info(f"System status: {status['tactical_status']['running']} tactical, {status['strategic_status']['running']} strategic")
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            coordinator.stop()
            logger.info("System shutdown complete")
            
    except Exception as e:
        logger.error(f"System deployment failed: {e}")
        raise

def validate_system():
    """Run system validation"""
    logger = logging.getLogger(__name__)
    logger.info("Running system validation...")
    
    try:
        # Test tactical environment
        from ddos_rl.env import TacticalUAVEnv
        from ddos_rl.agent import QLearningAgent
        
        env = TacticalUAVEnv()
        agent = QLearningAgent(state_dims=[4, 4, 3, 3], action_dim=9)
        
        state = env.reset()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        
        logger.info("✅ Tactical environment validation passed")
        
        # Test strategic environment
        from crypto_rl.strategic_agent import StrategicCryptoEnv
        
        env = StrategicCryptoEnv()
        state = env.reset()
        next_state, reward, done, info = env.step(0)
        
        logger.info("✅ Strategic environment validation passed")
        
        # Test configuration
        from config.crypto_config import CRYPTO_ALGORITHMS
        assert len(CRYPTO_ALGORITHMS) == 4
        logger.info("✅ Configuration validation passed")
        
        logger.info("🎉 All validation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Dual-Agent RL UAV System')
    parser.add_argument('command', choices=['train', 'deploy', 'validate'], 
                       help='Command to execute')
    parser.add_argument('--agent', choices=['tactical', 'strategic', 'both'], 
                       default='both', help='Which agent to train')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    parser.add_argument('--tactical-policy', help='Path to tactical policy file')
    parser.add_argument('--strategic-policy', help='Path to strategic policy file')
    parser.add_argument('--uav-id', default='UAV_001', help='UAV identifier')
    parser.add_argument('--gcs-id', default='GCS_MAIN', help='GCS identifier')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Dual-Agent RL UAV System")
    
    try:
        if args.command == 'validate':
            success = validate_system()
            sys.exit(0 if success else 1)
            
        elif args.command == 'train':
            if args.agent in ['tactical', 'both']:
                train_tactical_agent(
                    episodes=args.episodes,
                    save_path=args.tactical_policy or 'tactical_policy.npy'
                )
            
            if args.agent in ['strategic', 'both']:
                train_strategic_agent(
                    episodes=args.episodes,
                    save_path=args.strategic_policy or 'strategic_policy.npy'
                )
                
        elif args.command == 'deploy':
            deploy_system(
                tactical_policy=args.tactical_policy,
                strategic_policy=args.strategic_policy,
                uav_id=args.uav_id,
                gcs_id=args.gcs_id
            )
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
