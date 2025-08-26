#!/usr/bin/env python3
"""
Environment Setup and Validation Script for Dual-Agent RL UAV System
Installs dependencies, validates imports, and tests core functionality
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def test_core_imports():
    """Test core Python imports"""
    print("\nTesting core imports...")
    
    core_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'tqdm', 'scipy', 'sklearn', 'psutil'
    ]
    
    failed_imports = []
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_rl_environment():
    """Test RL environment imports"""
    print("\nTesting RL environment...")
    
    try:
        # Test gymnasium/gym
        try:
            import gymnasium as gym
            print("✅ gymnasium available")
        except ImportError:
            import gym
            print("✅ gym available (legacy)")
            
        # Test basic environment creation
        from ddos_rl.env import TacticalUAVEnv
        env = TacticalUAVEnv()
        state = env.reset()
        print(f"✅ TacticalUAVEnv: state shape {len(state) if hasattr(state, '__len__') else 'scalar'}")
        
        # Test agent
        from ddos_rl.agent import QLearningAgent
        agent = QLearningAgent(state_dims=(4, 4, 3, 3), action_dim=9)
        action = agent.choose_action((0, 1, 2, 1))
        print(f"✅ QLearningAgent: action {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ RL environment test failed: {e}")
        return False

def test_configuration():
    """Test configuration consistency"""
    print("\nTesting configuration...")
    
    try:
        # Test crypto config
        from config.crypto_config import CRYPTO_ALGORITHMS
        print(f"✅ Crypto config: {len(CRYPTO_ALGORITHMS)} algorithms")
        
        # Test profile functions
        from ddos_rl.profiles import get_crypto_latency_robust
        latency = get_crypto_latency_robust("KYBER")
        print(f"✅ Profile functions: KYBER latency {latency}ms")
        
        # Test strategic environment
        from crypto_rl.strategic_agent import StrategicCryptoEnv
        env = StrategicCryptoEnv()
        state = env.reset()
        print(f"✅ StrategicCryptoEnv: state shape {len(state) if hasattr(state, '__len__') else 'scalar'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_hardware_interface():
    """Test hardware interface (mock on non-RPi systems)"""
    print("\nTesting hardware interface...")
    
    try:
        from hardware.rpi_interface import RPiHardwareInterface, BatteryMonitor
        
        # Create interfaces
        rpi = RPiHardwareInterface()
        battery = BatteryMonitor()
        
        print("✅ Hardware interfaces created")
        
        # Test basic functionality (will use mocks on non-RPi)
        metrics = rpi.get_system_metrics()
        print(f"✅ System metrics: {len(metrics)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware interface test failed: {e}")
        return False

def test_communication():
    """Test communication interface"""
    print("\nTesting communication interface...")
    
    try:
        from communication.mavlink_interface import MAVLinkInterface, MessageType
        
        # Create interface
        mavlink = MAVLinkInterface("TEST_UAV", "UAV")
        print("✅ MAVLink interface created")
        
        # Test message types
        print(f"✅ Message types: {len(MessageType)} types available")
        
        return True
        
    except Exception as e:
        print(f"❌ Communication test failed: {e}")
        return False

def test_system_integration():
    """Test system integration"""
    print("\nTesting system integration...")
    
    try:
        from integration.system_coordinator import SystemCoordinator
        
        # Create coordinator (will handle import errors gracefully)
        coordinator = SystemCoordinator("TEST_UAV", "TEST_GCS")
        print("✅ System coordinator created")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def run_validation():
    """Run complete system validation"""
    print("🔍 Dual-Agent RL UAV System - Environment Validation")
    print("=" * 60)
    
    tests = [
        ("Core Dependencies", test_core_imports),
        ("RL Environment", test_rl_environment),
        ("Configuration", test_configuration),
        ("Hardware Interface", test_hardware_interface),
        ("Communication", test_communication),
        ("System Integration", test_system_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. System needs fixes before deployment.")
        return False

def main():
    """Main setup and validation"""
    print("Setting up Dual-Agent RL UAV System environment...")
    
    # Install dependencies
    if not install_requirements():
        print("❌ Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Run validation
    success = run_validation()
    
    if success:
        print("\n✅ Environment setup complete!")
        sys.exit(0)
    else:
        print("\n❌ Environment setup incomplete. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
