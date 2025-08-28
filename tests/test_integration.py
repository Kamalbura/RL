"""
Integration Tests for Dual-Agent RL UAV System
Tests system components integration and end-to-end functionality
"""

import unittest
import time
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration consistency across components"""
    
    def test_crypto_algorithm_consistency(self):
        """Test crypto algorithm configuration consistency"""
        from config.crypto_config import CRYPTO_ALGORITHMS
        from ddos_rl.profiles import get_crypto_latency_robust
        
        # Test all algorithms are accessible
        for algo_id in range(4):
            self.assertIn(algo_id, CRYPTO_ALGORITHMS)
            algo_name = CRYPTO_ALGORITHMS[algo_id]['name']
            
            # Test profile functions work with these algorithms
            latency = get_crypto_latency_robust(algo_name)
            self.assertIsInstance(latency, (int, float))
            self.assertGreater(latency, 0)
            
    def test_tactical_action_space_consistency(self):
        """Test tactical agent action space consistency"""
        from ddos_rl.env import TacticalUAVEnv
        from ddos_rl.agent import TacticalAgent

        env = TacticalUAVEnv()
        agent = TacticalAgent(state_dim=int(env.observation_space.shape[0]), action_dim=12)

        # Test action space matches environment
        self.assertEqual(env.action_space.spaces[0].n, 3)
        self.assertEqual(env.action_space.spaces[1].n, 4)

        # Test all actions are valid
        for model in range(3):
            for freq in range(4):
                state = env.reset()
                next_state, reward, done, info = env.step((model, freq))
                self.assertIsInstance(reward, (int, float))
            
    def test_strategic_action_space_consistency(self):
        """Test strategic agent action space consistency"""
        from crypto_rl.strategic_agent import StrategicCryptoEnv
        from config.crypto_config import CRYPTO_ALGORITHMS
        
        env = StrategicCryptoEnv()
        
        # Test action space matches crypto algorithms
        self.assertEqual(env.action_space.n, len(CRYPTO_ALGORITHMS))
        
        # Test all actions are valid
        for action in range(len(CRYPTO_ALGORITHMS)):
            state = env.reset()
            next_state, reward, done, info = env.step(action)
            self.assertIsInstance(reward, (int, float))


class TestHardwareIntegration(unittest.TestCase):
    """Test hardware integration components"""
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    def test_rpi_interface_basic(self, mock_cpu_percent, mock_cpu_count):
        """Test basic RPi interface functionality"""
        mock_cpu_count.return_value = 4
        mock_cpu_percent.return_value = 50.0
        
        from hardware.rpi_interface import RPiHardwareInterface
        
        rpi = RPiHardwareInterface()
        
        # Test frequency validation
        self.assertIn(600000, rpi.available_frequencies)
        self.assertIn(2000000, rpi.available_frequencies)
        
        # Test metrics collection
        metrics = rpi.get_system_metrics()
        self.assertIn('cpu_percent', metrics)
        self.assertIn('cpu_count', metrics)
        
    def test_battery_monitor_basic(self):
        """Test battery monitor functionality"""
        from hardware.rpi_interface import BatteryMonitor
        
        battery = BatteryMonitor(voltage_nominal=22.2, capacity_mah=5200)
        
        # Test configuration
        self.assertEqual(battery.voltage_nominal, 22.2)
        self.assertEqual(battery.capacity_mah, 5200)
        self.assertAlmostEqual(battery.capacity_wh, 115.44, places=2)


class TestCommunicationIntegration(unittest.TestCase):
    """Test communication system integration"""
    
    def test_mavlink_interface_basic(self):
        """Test MAVLink interface basic functionality"""
        from communication.mavlink_interface import MAVLinkInterface, MessageType
        
        mavlink = MAVLinkInterface("TEST_UAV", "UAV")
        
        # Test initialization
        self.assertEqual(mavlink.node_id, "TEST_UAV")
        self.assertEqual(mavlink.node_type, "UAV")
        self.assertFalse(mavlink.running)
        
        # Test message handler registration
        handler = Mock()
        mavlink.register_handler(MessageType.TACTICAL_STATUS, handler)
        self.assertIn(MessageType.TACTICAL_STATUS, mavlink.message_handlers)
        
    def test_uav_coordinator_basic(self):
        """Test UAV coordinator functionality"""
        from communication.mavlink_interface import UAVCoordinator
        
        coordinator = UAVCoordinator("TEST_GCS")
        
        # Test initialization
        self.assertEqual(coordinator.gcs_id, "TEST_GCS")
        self.assertEqual(len(coordinator.active_uavs), 0)
        
        # Test fleet status
        status = coordinator.get_fleet_status()
        self.assertIn('total_uavs', status)
        self.assertIn('active_uavs', status)


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    @patch('hardware.rpi_interface.get_hardware_interface')
    @patch('communication.mavlink_interface.MAVLinkInterface')
    def test_tactical_controller_initialization(self, mock_mavlink, mock_hardware):
        """Test tactical controller initialization"""
        mock_hardware.return_value = Mock()
        mock_mavlink.return_value = Mock()
        
        from integration.system_coordinator import TacticalAgentController
        
        controller = TacticalAgentController("TEST_UAV")
        
        # Test initialization
        self.assertEqual(controller.uav_id, "TEST_UAV")
        self.assertIsNotNone(controller.env)
        self.assertIsNotNone(controller.agent)
        self.assertFalse(controller.running)
        
    def test_strategic_controller_initialization(self):
        """Test strategic controller initialization"""
        from integration.system_coordinator import StrategicAgentController
        
        controller = StrategicAgentController("TEST_GCS")
        
        # Test initialization
        self.assertEqual(controller.gcs_id, "TEST_GCS")
        self.assertIsNotNone(controller.env)
        self.assertIsNotNone(controller.agent)
        self.assertFalse(controller.running)
        
    @patch('integration.system_coordinator.TacticalAgentController')
    @patch('integration.system_coordinator.StrategicAgentController')
    def test_system_coordinator_initialization(self, mock_strategic, mock_tactical):
        """Test system coordinator initialization"""
        mock_tactical.return_value = Mock()
        mock_strategic.return_value = Mock()
        
        from integration.system_coordinator import SystemCoordinator
        
        coordinator = SystemCoordinator("TEST_UAV", "TEST_GCS")
        
        # Test initialization
        self.assertEqual(coordinator.uav_id, "TEST_UAV")
        self.assertEqual(coordinator.gcs_id, "TEST_GCS")
        self.assertFalse(coordinator.running)


class TestDeploymentIntegration(unittest.TestCase):
    """Test deployment system integration"""
    
    def test_deployment_config_creation(self):
        """Test deployment configuration creation"""
        from deploy.deployment_manager import DeploymentConfig
        
        config = DeploymentConfig(
            uav_id="TEST_UAV",
            gcs_id="TEST_GCS",
            tactical_policy_path="test_tactical.npy",
            strategic_policy_path="test_strategic.npy"
        )
        
        # Test configuration
        self.assertEqual(config.uav_id, "TEST_UAV")
        self.assertEqual(config.gcs_id, "TEST_GCS")
        self.assertTrue(config.hardware_interface)
        self.assertTrue(config.communication_enabled)
        
    def test_deployment_manager_validation(self):
        """Test deployment manager environment validation"""
        from deploy.deployment_manager import DeploymentManager, DeploymentConfig
        
        config = DeploymentConfig(uav_id="TEST_UAV", gcs_id="TEST_GCS")
        manager = DeploymentManager(config)
        
        # Test validation
        checks = manager.validate_environment()
        self.assertIn('python_version', checks)
        self.assertIn('dir_ddos_rl', checks)
        self.assertIn('dir_crypto_rl', checks)


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end system functionality"""
    
    def test_tactical_strategic_state_mapping(self):
        """Test state mapping between tactical and strategic agents"""
        from ddos_rl.env import TacticalUAVEnv
        from crypto_rl.strategic_agent import StrategicCryptoEnv
        
        tactical_env = TacticalUAVEnv()
        strategic_env = StrategicCryptoEnv()
        
        # Test state space compatibility
        tactical_state = tactical_env.reset()
        strategic_state = strategic_env.reset()
        
        # Tactical: (threat, battery, cpu_load, task_priority)
        # Strategic: (threat, avg_battery, mission_phase)
        
        # Test threat level mapping (tactical has 4 levels, strategic has 3)
        tactical_threat = tactical_state[0]  # 0-3
        strategic_threat = min(2, tactical_threat)  # Map to 0-2
        
        self.assertIn(tactical_threat, range(4))
        self.assertIn(strategic_threat, range(3))
        
    def test_performance_profile_integration(self):
        """Test performance profile integration across components"""
        from ddos_rl.profiles import (
            get_power_consumption_robust,
            get_ddos_execution_time_robust,
            get_crypto_latency_robust
        )
        
        # Test tactical profiles
        power = get_power_consumption_robust(1800, 4)
        exec_time = get_ddos_execution_time_robust("TST", 1800, 4)
        
        self.assertIsInstance(power, (int, float))
        self.assertIsInstance(exec_time, (int, float))
        self.assertGreater(power, 0)
        self.assertGreater(exec_time, 0)
        
        # Test strategic profiles
        crypto_latency = get_crypto_latency_robust("KYBER")
        
        self.assertIsInstance(crypto_latency, (int, float))
        self.assertGreater(crypto_latency, 0)
        
    def test_agent_policy_compatibility(self):
        """Test agent policy save/load compatibility"""
        from ddos_rl.agent import QLearningAgent
        from ddos_rl.agent import TacticalAgent
        from ddos_rl.env import TacticalUAVEnv
        
        # Create temporary policy file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            policy_path = tmp.name
            
        try:
            # Test tactical DQN agent policy save/load
            agent1 = TacticalAgent(state_dim=5, action_dim=12)
            env = TacticalUAVEnv()
            for _ in range(5):
                s = env.reset()
                a = agent1.choose_action(s, training=True)
                ns, r, done, _ = env.step((a // 4, a % 4))
                agent1.remember(s, int(a), float(r), ns, bool(done))
                agent1.learn()
            agent1.save_policy(policy_path)
            
            agent2 = TacticalAgent(state_dim=5, action_dim=12)
            loaded = agent2.load_policy(policy_path)
            self.assertTrue(loaded)
            
        finally:
            if os.path.exists(policy_path):
                os.unlink(policy_path)


def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfigurationIntegration,
        TestHardwareIntegration,
        TestCommunicationIntegration,
        TestSystemIntegration,
        TestDeploymentIntegration,
        TestEndToEndIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
