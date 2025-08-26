"""
Reproducibility utilities for RL training runs.
Captures environment state, hyperparameters, and system info for full reproducibility.
"""

import os
import json
import time
import hashlib
import platform
import subprocess
from typing import Dict, Any, Optional
import numpy as np


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_info() -> Dict[str, str]:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                       stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(dirty),
            "dirty_files": dirty.split('\n') if dirty else []
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def get_system_info() -> Dict[str, Any]:
    """Get system and environment information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.time(),
        "iso_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    try:
        import numpy
        packages["numpy"] = numpy.__version__
    except ImportError:
        packages["numpy"] = "not_installed"
    
    try:
        import torch
        packages["torch"] = torch.__version__
        packages["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        packages["torch"] = "not_installed"
    
    try:
        import gymnasium
        packages["gymnasium"] = gymnasium.__version__
    except ImportError:
        try:
            import gym
            packages["gym"] = gym.__version__
        except ImportError:
            packages["gym"] = "not_installed"
    
    return packages


def hash_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return "file_not_found"
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def create_run_metadata(
    experiment_name: str,
    hyperparameters: Dict[str, Any],
    output_dir: str,
    seed: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create comprehensive metadata for a training run."""
    
    metadata = {
        "experiment_name": experiment_name,
        "seed": seed,
        "hyperparameters": hyperparameters,
        "git_info": get_git_info(),
        "system_info": get_system_info(),
        "package_versions": get_package_versions(),
        "output_directory": output_dir,
    }
    
    if additional_info:
        metadata["additional_info"] = additional_info
    
    # Save metadata
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata


def add_model_hashes(metadata_path: str, model_files: Dict[str, str]):
    """Add model file hashes to existing metadata."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    metadata["model_hashes"] = {}
    for name, filepath in model_files.items():
        metadata["model_hashes"][name] = hash_file(filepath)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def verify_reproducibility(metadata_path: str) -> Dict[str, bool]:
    """Verify if a run can be reproduced based on metadata."""
    if not os.path.exists(metadata_path):
        return {"metadata_exists": False}
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    checks = {
        "metadata_exists": True,
        "seed_specified": metadata.get("seed") is not None,
        "git_clean": not metadata.get("git_info", {}).get("dirty", True),
        "hyperparameters_recorded": bool(metadata.get("hyperparameters")),
        "system_info_recorded": bool(metadata.get("system_info")),
    }
    
    # Check if model files still exist and match hashes
    if "model_hashes" in metadata:
        checks["model_files_intact"] = True
        for name, expected_hash in metadata["model_hashes"].items():
            # Assume model files are in same directory as metadata
            model_path = os.path.join(os.path.dirname(metadata_path), f"{name}.npy")
            if hash_file(model_path) != expected_hash:
                checks["model_files_intact"] = False
                break
    else:
        checks["model_files_intact"] = False
    
    return checks


# Example usage functions
def setup_tactical_training_metadata(episodes: int, seed: int, output_dir: str) -> Dict[str, Any]:
    """Setup metadata for tactical agent training."""
    hyperparams = {
        "episodes": episodes,
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "exploration_rate": 1.0,
        "exploration_decay": 0.995,
        "min_exploration_rate": 0.01,
        "eval_frequency": 100,
        "checkpoint_every": 500,
    }
    
    return create_run_metadata(
        experiment_name="tactical_agent_training",
        hyperparameters=hyperparams,
        output_dir=output_dir,
        seed=seed,
        additional_info={
            "agent_type": "tactical",
            "state_dims": [4, 4, 3, 3],
            "action_dim": 9,
            "environment": "TacticalUAVEnv"
        }
    )


def setup_strategic_training_metadata(episodes: int, seed: int, output_dir: str) -> Dict[str, Any]:
    """Setup metadata for strategic agent training."""
    hyperparams = {
        "episodes": episodes,
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "exploration_rate": 1.0,
        "exploration_decay": 0.9995,
        "min_exploration_rate": 0.01,
        "eval_every": 250,
    }
    
    return create_run_metadata(
        experiment_name="strategic_agent_training",
        hyperparameters=hyperparams,
        output_dir=output_dir,
        seed=seed,
        additional_info={
            "agent_type": "strategic",
            "state_dims": [3, 3, 4],
            "action_dim": 4,
            "environment": "StrategicCryptoEnv"
        }
    )
