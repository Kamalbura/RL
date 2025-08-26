# Hierarchical RL UAV Cybersecurity System

A production-ready dual-agent reinforcement learning system for UAV cybersecurity operations with tactical (UAV-side) and strategic (GCS-side) intelligent decision making.

## 📚 **Quick Navigation**

### 🎯 **Essential Documentation**
- **[📖 Expert Model Summary](EXPERT_MODEL_SUMMARY.md)** - Complete technical guide with diagrams and educational content
- **[🏗️ System Documentation](HIERARCHICAL_RL_SYSTEM_DOCUMENTATION.md)** - Architecture, deployment, and operational guidelines
- **[🔍 Expert Evaluation Guide](EXPERT_EVALUATION_GUIDE.md)** - RL practitioner audit and evaluation framework
- **[📋 Training Guide](DETAILED_TRAINING_GUIDE.md)** - Comprehensive training procedures and best practices
- **[👥 Team Onboarding](TEAM_ONBOARDING_GUIDE.md)** - Getting started guide for new team members

### 🚀 **Core Applications**
- **[🛡️ UAV Scheduler](uav-scheduler.py)** - Main tactical RL agent with thermal-aware control
- **[🎛️ GCS Scheduler](gcs-scheduler.py)** - Strategic RL agent with human-in-the-loop interface
- **[⚙️ Main Controller](main.py)** - System entry point and orchestration
- **[🔧 System Validation](validation.py)** - Comprehensive system testing and validation

### 📊 **Data & Configuration**
- **[📈 System Profiles](shared/system_profiles.py)** - Central knowledge base with empirical performance data
- **[🔐 Crypto Config](config/crypto_config.py)** - Post-quantum cryptography configuration
- **[🔋 Performance Profiles](performance_profiles.py)** - Hardware performance characteristics

### 🧪 **RL Components**
- **[🤖 Tactical Agent](ddos_rl/agent.py)** - Q-learning agent for UAV-side decisions
- **[🎯 Strategic Agent](crypto_rl/strategic_agent.py)** - Fleet-wide crypto policy agent
- **[🌍 Tactical Environment](ddos_rl/env.py)** - UAV cybersecurity environment
- **[🏛️ Strategic Environment](crypto_rl/strategic_agent.py)** - GCS crypto selection environment

### 🔧 **Integration & Deployment**
- **[🔗 System Coordinator](integration/system_coordinator.py)** - Dual-agent coordination
- **[📡 GCS Integration](integration/gcs_integration.py)** - Strategic RL UI integration
- **[🚀 Deployment Manager](deploy/deployment_manager.py)** - Automated deployment system
- **[🔌 Hardware Interface](hardware/rpi_interface.py)** - Raspberry Pi 4B control
- **[📞 MAVLink Interface](communication/mavlink_interface.py)** - UAV-GCS communication

### 🧰 **Utilities & Testing**
- **[🔄 Reproducibility](utils/reproducibility.py)** - Random seed management
- **[⏹️ Early Stopping](utils/early_stopping.py)** - Training optimization
- **[📊 Reward Monitor](utils/reward_monitor.py)** - Training analytics
- **[✅ Integration Tests](tests/test_integration.py)** - System validation

### 📁 **Archive & Reference**
- **[📦 Dump Folder](dump/)** - Archived files and old versions
- **[📝 Context Data](dump/context.txt)** - Original empirical measurements
- **[📋 Progress Notes](dump/progress.txt)** - Development history

## System Overview

### Architecture
- **Tactical Agent** (Raspberry Pi 4B): DDoS detection model selection + CPU frequency management
- **Strategic Agent** (Ground Control Station): Fleet-wide cryptographic algorithm selection
- **Communication**: MAVLink-based UAV-GCS coordination
- **Hardware Integration**: Real-time RPi 4B control with thermal monitoring

### Key Features
- **Empirical Performance Models**: Based on real hardware measurements
- **Post-Quantum Cryptography**: KYBER, DILITHIUM, SPHINCS, FALCON algorithms
- **Multi-Objective Optimization**: Security-Energy-Performance trade-offs
- **Production Deployment**: Automated setup and monitoring

## Quick Start

### 1. Environment Setup
```bash
# Activate conda environment
conda activate rl_env

# Install dependencies
pip install -r requirements.txt

# Validate system
python main.py validate
```

### 2. Training Agents
```bash
# Train both agents
python main.py train --episodes 1000

# Train specific agent
python main.py train --agent tactical --episodes 500
python main.py train --agent strategic --episodes 500
```

### 3. System Deployment
```bash
# Deploy with trained policies
python main.py deploy --tactical-policy tactical_policy.npy --strategic-policy strategic_policy.npy

# Deploy with custom IDs
python main.py deploy --uav-id UAV_001 --gcs-id GCS_MAIN
```

## System Components

### Tactical Agent (UAV)
- **State Space**: Threat(4) × Battery(4) × CPU Load(3) × Task Priority(3) = 144 states
- **Action Space**: 9 actions (2 DDoS models × 4 CPU frequencies + de-escalate)
- **Hardware**: Raspberry Pi 4B with 600-2000 MHz CPU control
- **Power Model**: 4.53-8.38W based on empirical measurements

### Strategic Agent (GCS)
- **State Space**: Threat(3) × Fleet Battery(3) × Mission Phase(4) = 36 states
- **Action Space**: 4 cryptographic algorithms
- **Algorithms**: KYBER (209.6ms), DILITHIUM (264.0ms), SPHINCS (687.4ms), FALCON (445.4ms)
- **Security Ratings**: 8.5-9.5/10 based on NIST standards

### Performance Profiles
All performance data based on empirical measurements:
- **Power Consumption**: RPi 4B @ 22.2V with 6S LiPo battery
- **DDoS Detection**: TST (F1: 0.999) vs XGBoost (F1: 0.67-0.89)
- **Crypto Latency**: Real implementation timing data
- **Thermal Limits**: 80°C throttling, 85°C critical

## 🏗️ **Clean Project Structure** (Presentation Ready)

```
RL/
├── 📋 README.md                           # Navigation & Quick Start
├── 📦 requirements.txt                    # Python dependencies  
├── 🚫 .gitignore                         # Git ignore rules
│
├── 📚 **DOCUMENTATION**
│   ├── 📖 EXPERT_MODEL_SUMMARY.md         # Complete technical guide
│   ├── 🏗️ HIERARCHICAL_RL_SYSTEM_DOCUMENTATION.md
│   ├── 🔍 EXPERT_EVALUATION_GUIDE.md     # RL audit framework
│   ├── 📋 DETAILED_TRAINING_GUIDE.md     # Training procedures
│   └── 👥 TEAM_ONBOARDING_GUIDE.md       # Getting started
│
├── 🚀 **CORE APPLICATIONS**
│   ├── ⚙️ main.py                        # System orchestration
│   ├── 🛡️ uav-scheduler.py               # Tactical RL agent (UAV)
│   ├── 🎛️ gcs-scheduler.py               # Strategic RL agent (GCS)
│   └── 🔧 validation.py                  # System testing
│
├── 🤖 **RL AGENTS**
│   ├── ddos_rl/                          # Tactical (UAV-side)
│   │   ├── agent.py                      # Q-learning implementation
│   │   ├── env.py                        # UAV environment
│   │   └── config.py                     # Tactical configuration
│   └── crypto_rl/                        # Strategic (GCS-side)
│       ├── strategic_agent.py            # Strategic RL agent
│       ├── rl_agent.py                   # Base Q-learning
│       └── consensus.py                  # Multi-agent consensus
│
├── 📊 **DATA & CONFIGURATION**
│   ├── shared/
│   │   └── 📈 system_profiles.py         # Central knowledge base
│   └── config/
│       ├── 🔐 crypto_config.py           # Post-quantum algorithms
│       └── __init__.py
│
├── 🔧 **SYSTEM INTEGRATION**
│   ├── integration/
│   │   ├── 🔗 system_coordinator.py      # Dual-agent coordination
│   │   └── 📡 gcs_integration.py         # Strategic UI integration
│   ├── hardware/
│   │   └── 🔌 rpi_interface.py           # Raspberry Pi 4B control
│   └── communication/
│       └── 📞 mavlink_interface.py       # UAV-GCS messaging
│
├── 🚀 **DEPLOYMENT**
│   └── deploy/
│       └── 🚀 deployment_manager.py      # Automated deployment
│
├── 🧰 **UTILITIES**
│   └── utils/
│       ├── 🔄 reproducibility.py        # Random seed management
│       ├── ⏹️ early_stopping.py          # Training optimization
│       └── 📊 reward_monitor.py          # Performance analytics
│
├── ✅ **TESTING**
│   └── tests/
│       └── ✅ test_integration.py        # System validation
│
└── 📁 **ARCHIVE** (Hidden from Git)
    └── dump/                             # Old files & references
```

## Configuration

### Tactical Agent Configuration
```python
# Battery: 6S LiPo, 22.2V, 5200mAh
BATTERY_SPECS = {
    "VOLTAGE": 22.2,
    "CAPACITY_MAH": 5200,
    "CAPACITY_WH": 115.44
}

# CPU Frequencies (MHz)
CPU_FREQUENCY_PRESETS = {
    "POWERSAVE": 600,
    "BALANCED": 1200,
    "PERFORMANCE": 1800,
    "TURBO": 2000
}
```

### Strategic Agent Configuration
```python
# Post-quantum cryptographic algorithms
CRYPTO_ALGORITHMS = {
    0: {"name": "KYBER", "latency_ms": 209.6, "security_rating": 8.5},
    1: {"name": "DILITHIUM", "latency_ms": 264.0, "security_rating": 9.0},
    2: {"name": "SPHINCS", "latency_ms": 687.4, "security_rating": 9.5},
    3: {"name": "FALCON", "latency_ms": 445.4, "security_rating": 8.8}
}
```

## Hardware Requirements

### UAV (Tactical Agent)
- **Platform**: Raspberry Pi 4B (8GB recommended)
- **Battery**: 6S LiPo, 22.2V, 5200mAh (115.44Wh)
- **Storage**: 32GB+ microSD card
- **Communication**: WiFi 802.11ac
- **Sensors**: Temperature monitoring, current sensing

### GCS (Strategic Agent)
- **Platform**: Windows/Linux workstation
- **RAM**: 8GB+ recommended
- **Network**: WiFi/Ethernet for fleet communication
- **Storage**: SSD recommended for logging

## Training Parameters

### Default Hyperparameters
```python
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.9995
MIN_EXPLORATION_RATE = 0.01
```

### Training Recommendations
- **Tactical Agent**: 1000+ episodes for convergence
- **Strategic Agent**: 500+ episodes sufficient
- **Batch Size**: Single-step Q-learning (tabular)
- **Evaluation**: Every 100 episodes

## Deployment

### Production Deployment
```bash
# Full system deployment
python deploy/deployment_manager.py --uav-id UAV_001 --gcs-id GCS_MAIN

# Hardware validation
python hardware/rpi_interface.py --test

# Communication testing
python communication/mavlink_interface.py --test
```

### System Monitoring
- **Logs**: `uav_rl_system.log`
- **Metrics**: Real-time power, temperature, battery
- **Alerts**: Thermal throttling, low battery, communication loss

## Validation

### System Tests
```bash
# Basic validation
python main.py validate

# Integration tests
python tests/test_integration.py

# Hardware tests (on RPi)
python hardware/rpi_interface.py --validate
```

### Performance Benchmarks
- **Tactical Decision Time**: <100ms
- **Strategic Decision Time**: <1s
- **Communication Latency**: <50ms
- **Power Efficiency**: 4.53-8.38W operational range

## Research Integration

### Empirical Data Sources
- **IEEE Paper**: DDoS detection F1 scores and timing
- **Context.txt**: Real hardware measurements and crypto benchmarks
- **NIST Standards**: Post-quantum cryptography security ratings

### Publications
System designed for research reproducibility with comprehensive logging and metrics collection suitable for academic publication.

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure conda environment activated
2. **Hardware Access**: Run with sudo on RPi for frequency control
3. **Communication**: Check WiFi connectivity and firewall settings
4. **Memory**: Increase swap on RPi for large Q-tables

### Support
- Check logs in `uav_rl_system.log`
- Run validation tests for component diagnosis
- Monitor system health via integration coordinator

## License

Research and educational use. See individual component licenses for commercial deployment.
