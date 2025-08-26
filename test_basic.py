"""
Basic test to verify Python environment is working
"""

print("🔍 Testing Python Environment")
print("=" * 40)

# Test 1: Basic Python
print("✅ Python is running")

# Test 2: Basic imports
try:
    import sys
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
except Exception as e:
    print(f"❌ Python version check failed: {e}")

# Test 3: NumPy
try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError:
    print("❌ NumPy not available")

# Test 4: Project structure
import os
dirs = ['ddos_rl', 'crypto_rl', 'config']
for d in dirs:
    if os.path.exists(d):
        print(f"✅ Directory {d}/ exists")
    else:
        print(f"❌ Directory {d}/ missing")

# Test 5: Config import
try:
    sys.path.insert(0, os.getcwd())
    from config.crypto_config import CRYPTO_ALGORITHMS
    print(f"✅ Crypto config loaded: {len(CRYPTO_ALGORITHMS)} algorithms")
    
    # Show algorithm names
    for i, algo in CRYPTO_ALGORITHMS.items():
        print(f"   {i}: {algo['name']} ({algo['latency_ms']}ms)")
        
except Exception as e:
    print(f"❌ Config import failed: {e}")

# Test 6: Tactical environment
try:
    from ddos_rl.env import TacticalUAVEnv
    env = TacticalUAVEnv()
    state = env.reset()
    print(f"✅ TacticalUAVEnv works: state {state}")
except Exception as e:
    print(f"❌ TacticalUAVEnv failed: {e}")

print("\n" + "=" * 40)
print("Basic test complete!")
