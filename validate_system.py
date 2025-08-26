"""
Simple System Validation Script
Tests core components without complex dependencies
"""

import sys
import os

def test_basic_imports():
    """Test basic Python functionality"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import json
        print("✅ JSON available")
    except ImportError:
        print("❌ JSON not available")
        return False
        
    return True

def test_project_structure():
    """Test project directory structure"""
    print("\nTesting project structure...")
    
    required_dirs = [
        'ddos_rl', 'crypto_rl', 'config', 'hardware', 
        'communication', 'integration', 'deploy', 'tests'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ missing")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def test_core_files():
    """Test core configuration files"""
    print("\nTesting core files...")
    
    core_files = [
        'config/crypto_config.py',
        'ddos_rl/env.py',
        'ddos_rl/agent.py',
        'crypto_rl/strategic_agent.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in core_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_crypto_config():
    """Test crypto configuration loading"""
    print("\nTesting crypto configuration...")
    
    try:
        sys.path.insert(0, os.getcwd())
        from config.crypto_config import CRYPTO_ALGORITHMS
        
        print(f"✅ Loaded {len(CRYPTO_ALGORITHMS)} crypto algorithms")
        
        # Check algorithm names
        expected_algos = ['KYBER', 'DILITHIUM', 'SPHINCS', 'FALCON']
        for i, expected in enumerate(expected_algos):
            if i in CRYPTO_ALGORITHMS and CRYPTO_ALGORITHMS[i]['name'] == expected:
                print(f"✅ {expected} configured correctly")
            else:
                print(f"❌ {expected} configuration issue")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Crypto config test failed: {e}")
        return False

def test_tactical_env():
    """Test tactical environment basic loading"""
    print("\nTesting tactical environment...")
    
    try:
        from ddos_rl.env import TacticalUAVEnv
        env = TacticalUAVEnv()
        print("✅ TacticalUAVEnv created")
        
        # Test basic functionality
        state = env.reset()
        print(f"✅ Environment reset: state {state}")
        
        # Test action space
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
            print(f"✅ Action space: {env.action_space.n} actions")
        
        return True
        
    except Exception as e:
        print(f"❌ Tactical environment test failed: {e}")
        return False

def test_strategic_env():
    """Test strategic environment basic loading"""
    print("\nTesting strategic environment...")
    
    try:
        from crypto_rl.strategic_agent import StrategicCryptoEnv
        env = StrategicCryptoEnv()
        print("✅ StrategicCryptoEnv created")
        
        # Test basic functionality
        state = env.reset()
        print(f"✅ Environment reset: state {state}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategic environment test failed: {e}")
        return False

def main():
    """Run validation tests"""
    print("🔍 Dual-Agent RL UAV System - Basic Validation")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Structure", test_project_structure),
        ("Core Files", test_core_files),
        ("Crypto Configuration", test_crypto_config),
        ("Tactical Environment", test_tactical_env),
        ("Strategic Environment", test_strategic_env)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    score = (passed / total) * 10
    print(f"\nValidation Score: {score:.1f}/10")
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All basic tests passed!")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Minor issues to resolve.")
    else:
        print("❌ Major issues found. System needs significant fixes.")

if __name__ == "__main__":
    main()
