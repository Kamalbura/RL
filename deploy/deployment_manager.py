"""
Deployment Manager for Dual-Agent RL UAV System
Handles system deployment, configuration, and monitoring
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    uav_id: str
    gcs_id: str
    tactical_policy_path: Optional[str] = None
    strategic_policy_path: Optional[str] = None
    hardware_interface: bool = True
    communication_enabled: bool = True
    logging_level: str = "INFO"
    data_directory: str = "./deployment_data"
    backup_enabled: bool = True

class DeploymentManager:
    """Manages system deployment and configuration"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.deployment_status = "STOPPED"
        
    def _setup_logging(self) -> logging.Logger:
        """Setup deployment logging"""
        logger = logging.getLogger("deployment_manager")
        logger.setLevel(getattr(logging, self.config.logging_level))
        
        # Create deployment data directory
        os.makedirs(self.config.data_directory, exist_ok=True)
        
        # File handler
        log_file = os.path.join(self.config.data_directory, "deployment.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.logging_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def validate_environment(self) -> Dict[str, bool]:
        """Validate deployment environment"""
        checks = {}
        
        # Python version check
        checks['python_version'] = sys.version_info >= (3, 8)
        
        # Required directories
        required_dirs = ['ddos_rl', 'crypto_rl', 'hardware', 'communication', 'integration']
        for dir_name in required_dirs:
            checks[f'dir_{dir_name}'] = os.path.exists(dir_name)
            
        # Policy files
        if self.config.tactical_policy_path:
            checks['tactical_policy'] = os.path.exists(self.config.tactical_policy_path)
        if self.config.strategic_policy_path:
            checks['strategic_policy'] = os.path.exists(self.config.strategic_policy_path)
            
        # Hardware dependencies (for RPi deployment)
        try:
            import psutil
            checks['psutil'] = True
        except ImportError:
            checks['psutil'] = False
            
        # Configuration files
        checks['crypto_config'] = os.path.exists('config/crypto_config.py')
        checks['main_config'] = os.path.exists('config.py')
        
        return checks
        
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            requirements = [
                "numpy>=1.21.0",
                "psutil>=5.8.0",
                "gymnasium>=0.26.0",
                "tqdm>=4.62.0",
                "pandas>=1.3.0",
                "matplotlib>=3.4.0",
                "seaborn>=0.11.0"
            ]
            
            for requirement in requirements:
                self.logger.info(f"Installing {requirement}")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to install {requirement}: {result.stderr}")
                    return False
                    
            self.logger.info("All dependencies installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing dependencies: {e}")
            return False
            
    def setup_system_services(self) -> bool:
        """Setup system services for deployment"""
        try:
            # Create systemd service files (for Linux deployment)
            service_template = """
[Unit]
Description=UAV RL System - {service_name}
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={working_dir}
ExecStart={python_path} -m {module_name}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            services = [
                {
                    'name': 'uav-tactical-agent',
                    'module': 'integration.tactical_service',
                    'description': 'Tactical UAV Agent Service'
                },
                {
                    'name': 'gcs-strategic-agent', 
                    'module': 'integration.strategic_service',
                    'description': 'Strategic GCS Agent Service'
                }
            ]
            
            for service in services:
                service_content = service_template.format(
                    service_name=service['description'],
                    working_dir=os.getcwd(),
                    python_path=sys.executable,
                    module_name=service['module']
                )
                
                service_file = f"/etc/systemd/system/{service['name']}.service"
                self.logger.info(f"Would create service file: {service_file}")
                # In practice: write service_content to service_file
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up system services: {e}")
            return False
            
    def deploy_system(self) -> bool:
        """Deploy the complete system"""
        try:
            self.deployment_status = "DEPLOYING"
            self.logger.info("Starting system deployment")
            
            # Validate environment
            validation_results = self.validate_environment()
            failed_checks = [k for k, v in validation_results.items() if not v]
            
            if failed_checks:
                self.logger.error(f"Environment validation failed: {failed_checks}")
                self.deployment_status = "FAILED"
                return False
                
            # Install dependencies
            if not self.install_dependencies():
                self.deployment_status = "FAILED"
                return False
                
            # Setup configuration
            self._create_deployment_config()
            
            # Initialize system coordinator
            from integration.system_coordinator import SystemCoordinator
            
            coordinator = SystemCoordinator(
                uav_id=self.config.uav_id,
                gcs_id=self.config.gcs_id
            )
            
            # Start system
            coordinator.start(
                tactical_policy=self.config.tactical_policy_path,
                strategic_policy=self.config.strategic_policy_path
            )
            
            # Verify deployment
            time.sleep(5.0)  # Allow startup time
            status = coordinator.get_system_status()
            
            if status['running']:
                self.deployment_status = "RUNNING"
                self.logger.info("System deployment successful")
                
                # Save deployment info
                self._save_deployment_info(status)
                return True
            else:
                self.deployment_status = "FAILED"
                self.logger.error("System failed to start properly")
                return False
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.deployment_status = "FAILED"
            return False
            
    def _create_deployment_config(self):
        """Create deployment configuration files"""
        config_data = {
            'deployment': asdict(self.config),
            'timestamp': time.time(),
            'version': '1.0.0'
        }
        
        config_file = os.path.join(self.config.data_directory, "deployment_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        self.logger.info(f"Deployment config saved to {config_file}")
        
    def _save_deployment_info(self, status: Dict):
        """Save deployment information"""
        deployment_info = {
            'config': asdict(self.config),
            'status': status,
            'deployment_time': time.time(),
            'deployment_status': self.deployment_status
        }
        
        info_file = os.path.join(self.config.data_directory, "deployment_info.json")
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
        self.logger.info(f"Deployment info saved to {info_file}")
        
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            'status': self.deployment_status,
            'config': asdict(self.config),
            'timestamp': time.time()
        }
        
    def stop_deployment(self) -> bool:
        """Stop deployed system"""
        try:
            self.logger.info("Stopping system deployment")
            
            # Stop system coordinator
            from integration.system_coordinator import get_system_coordinator
            coordinator = get_system_coordinator()
            coordinator.stop()
            
            self.deployment_status = "STOPPED"
            self.logger.info("System deployment stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping deployment: {e}")
            return False


def create_deployment_script():
    """Create deployment script"""
    script_content = '''#!/usr/bin/env python3
"""
UAV RL System Deployment Script
Usage: python deploy_system.py [--uav-id UAV_001] [--gcs-id GCS_MAIN]
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from deploy.deployment_manager import DeploymentManager, DeploymentConfig

def main():
    parser = argparse.ArgumentParser(description='Deploy UAV RL System')
    parser.add_argument('--uav-id', default='UAV_001', help='UAV identifier')
    parser.add_argument('--gcs-id', default='GCS_MAIN', help='GCS identifier')
    parser.add_argument('--tactical-policy', help='Path to tactical agent policy')
    parser.add_argument('--strategic-policy', help='Path to strategic agent policy')
    parser.add_argument('--no-hardware', action='store_true', help='Disable hardware interface')
    parser.add_argument('--no-comm', action='store_true', help='Disable communication')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Create deployment configuration
    config = DeploymentConfig(
        uav_id=args.uav_id,
        gcs_id=args.gcs_id,
        tactical_policy_path=args.tactical_policy,
        strategic_policy_path=args.strategic_policy,
        hardware_interface=not args.no_hardware,
        communication_enabled=not args.no_comm,
        logging_level=args.log_level
    )
    
    # Create deployment manager
    manager = DeploymentManager(config)
    
    try:
        print(f"Deploying UAV RL System...")
        print(f"UAV ID: {config.uav_id}")
        print(f"GCS ID: {config.gcs_id}")
        print(f"Hardware Interface: {config.hardware_interface}")
        print(f"Communication: {config.communication_enabled}")
        print()
        
        # Deploy system
        success = manager.deploy_system()
        
        if success:
            print("✅ Deployment successful!")
            print("System is running. Check logs for details.")
            
            # Keep running
            try:
                while True:
                    status = manager.get_deployment_status()
                    print(f"Status: {status['status']}")
                    time.sleep(30)
            except KeyboardInterrupt:
                print("\\nShutting down...")
                manager.stop_deployment()
                
        else:
            print("❌ Deployment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    script_path = "deploy_system.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
        
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
        
    return script_path
