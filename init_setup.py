"""
Initialization script to set up the folder structure and dependencies
"""

import os
import sys
import shutil

def create_folder_structure():
    """Create the folder structure for the project"""
    folders = [
        "config",
        "crypto_rl",
        "strategic_rl",
        "integration",
        "output",
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    
    # Create __init__.py files to make modules importable
    for folder in folders:
        init_file = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {folder} module\n")
            print(f"Created {init_file}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "numpy",
        "matplotlib",
        "gym",
        "tqdm",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall using:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("All required dependencies are installed")

if __name__ == "__main__":
    print("Setting up Crypto RL project folder structure...")
    create_folder_structure()
    print("\nChecking dependencies...")
    check_dependencies()
    print("\nSetup complete!")
