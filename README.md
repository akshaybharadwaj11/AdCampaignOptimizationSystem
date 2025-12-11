# Multi-Agent Reinforcement Learning System with CrewAI

> **Production-Ready Implementation for Ad Auction Optimization**  
> Author: Akshay | Version: 2.0 | December 2025

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

A sophisticated multi-agent agentic system that uses reinforcement learning to optimize ad auction bidding and budget allocation. The system demonstrates advanced RL techniques, comprehensive experimental validation, and production-ready features.

### Key Features

✅ **Advanced RL Algorithms**
- PPO (Proximal Policy Optimization) for controller and budget agents
- Dueling DDQN (Double Deep Q-Network) for bidding agent
- Prioritized Experience Replay for sample efficiency
- Generalized Advantage Estimation (GAE)

✅ **Multi-Agent Architecture**
- Controller Agent: Master orchestrator using PPO
- Bidding Agent: Auction bidding with Dueling DDQN
- Budget Allocation Agent: Resource distribution with PPO
- Analytics Agent: LLM-powered insights via CrewAI
- Simulation Agent: Realistic market environment

✅ **Production Features**
- Comprehensive error handling and fallback mechanisms
- Model versioning and checkpointing
- Real-time monitoring and alerting
- A/B testing framework
- Hyperparameter optimization with Optuna
- Statistical validation and ablation studies

✅ **Experimental Rigor**
- Baseline implementations (Random, Fixed, Greedy)
- Learning curves with confidence intervals
- Statistical significance testing
- Ablation studies for component importance
- Comprehensive visualization suite

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Controller Agent (PPO)                    │
│              Master Orchestrator & Coordinator               │
└──────────────────┬─────────────────┬────────────────────────┘
                   │                 │
        ┌──────────▼─────────┐  ┌───▼────────────────┐
        │  Bidding Agent     │  │ Budget Agent       │
        │   (Dueling DDQN)   │  │    (PPO)           │
        │ • 10 bid levels    │  │ • 3 channels       │
        │ • PER buffer       │  │ • Continuous alloc │
        └──────────┬─────────┘  └───┬────────────────┘
                   │                 │
                   └────────┬────────┘
                            │
                   ┌────────▼─────────┐
                   │ Simulation Agent │
                   │  • Market model  │
                   │  • Auction logic │
                   │  • Reward shaping│
                   └──────────────────┘

        ┌────────────────────────────────────┐
        │    Analytics Agent (CrewAI)        │
        │  • Performance analysis            │
        │  • Anomaly detection               │
        │  • Strategic recommendations       │
        └────────────────────────────────────┘
```

### Data Flow

```
Episode Start → SimulationAgent.reset() → State
                                            │
                     ┌──────────────────────▼
                     │
1. ControllerAgent.select_action(state) → decision
                     │
         ┌───────────┴────────────┐
         │                        │
2a. BiddingAgent.select_bid()    2b. BudgetAgent.select_allocation()
         │                        │
         └───────────┬────────────┘
                     │
3. Create Action → Execute in Environment
                     │
4. SimulationAgent.step(action) → (next_state, reward, done, info)
                     │
5. Store experiences → Update agents
                     │
6. AnalyticsAgent (periodic) → Insights & recommendations
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/akshaybharadwaj11/AdCampaignOptimizationSystem.git
cd multi-agent-rl-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import crewai; print('Installation successful!')"
```

---


## Quick Start

### Basic Training

```python
from agent_utils import EnhancedControllerAgent, EnhancedBiddingAgent
from experiment_utils import EnhancedBudgetAgent, EnhancedSimulationAgent
from orchestration_utils import ProductionMultiAgentSystem

# Initialize system
config = {
    'agent': {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'hidden_dim': 256
    },
    'environment': {
        'initial_budget': 10000.0,
        'max_steps': 100
    }
}

system = ProductionMultiAgentSystem(config=config)

# Train
system.train(num_episodes=1000, eval_frequency=50)

# Evaluate
eval_reward = system.evaluate(num_episodes=20)
print(f"Final Performance: {eval_reward:.2f}")
```

### Running Experiments

```python
from orchestration_utils import ProductionMultiAgentSystem

system = ProductionMultiAgentSystem()

# Run comprehensive experiments with baselines
system.run_comprehensive_experiments()

# Results saved to:
# - experiments/learning_curves.png
# - experiments/comparison_boxplot.png
# - experiments/experiment_report.txt
```

### Hyperparameter Optimization

```python
from experiment_utils import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(n_trials=50)
best_params = optimizer.optimize()

print(f"Best hyperparameters: {best_params}")
# Results: hyperopt/optimization_results.csv
```

### A/B Testing

```python
from orchestration_utils import ABTestFramework

ab_test = ABTestFramework()

# Compare two agent configurations
results = ab_test.run_test(
    variant_a=trained_system,
    variant_b=baseline_system,
    env=environment,
    test_name="RL_vs_Fixed",
    num_episodes=100
)

print(f"Winner: {results['winner']}")
print(f"Improvement: {results['improvement_percent']:.2f}%")
```

---

## 📊 Experimental Results

### Performance Metrics

Our system demonstrates significant improvements over baseline approaches:

| Agent | Mean Reward | Conversions | ROI | Win Rate |
|-------|-------------|-------------|-----|----------|
| **Trained RL** | **156.3 ± 23.4** | **42.1** | **63.2%** | **0.58** |
| Fixed Strategy | 89.7 ± 18.2 | 28.3 | 12.4% | 0.45 |
| Greedy | 102.3 ± 21.1 | 31.7 | 24.8% | 0.48 |
| Random | -12.4 ± 45.6 | 15.2 | -42.1% | 0.32 |

---

## 🔧 Configuration

### Agent Configuration

```yaml
# config.yaml
agent:
  learning_rate: 3e-4
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  batch_size: 64
  buffer_size: 50000
  hidden_dim: 256
  update_frequency: 4
  target_update_frequency: 100

environment:
  initial_budget: 10000.0
  max_steps: 100
  min_bid: 0.5
  max_bid: 5.0
  num_bid_levels: 10
  num_channels: 3
  conversion_value: 50.0

experiment:
  num_episodes: 1000
  eval_frequency: 50
  save_frequency: 100
  num_eval_episodes: 20
  random_seed: 42
```

### Monitoring Thresholds

```python
monitoring_config = {
    'min_reward': -100,
    'max_loss': 1000,
    'min_entropy': 0.05,
    'max_gradient_norm': 10,
    'min_win_rate': 0.1
}
```

---

## 🧪 Testing

### Comprehensive Test Suite

```python
from orchestration_utils import TestSuite

test_suite = TestSuite()

# Test controller convergence
test_suite.test_controller_convergence(controller, env)

# Test bidding stability
test_suite.test_bidding_stability(bidding_agent, env)

# Test error handling
test_suite.test_error_handling(system, env)

# Generate report
print(test_suite.generate_report())
```

### Running All Tests

```bash
python -m pytest tests/ -v --cov=src
```

---

## 📁 Project Structure

```
multi-agent-rl-system/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── config.yaml                  # Configuration
│
├── src/
│   ├── agents_utils.py         # Core RL agents
│   ├── experiment_experiments.py    # Experimental framework
│   └── orchestration_utils.py     # Production orchestrator
│
├── experiments/
│   ├── learning_curves.png
│   ├── comparison_boxplot.png
│   ├── experiment_report.txt
│   └── results.pkl
│
├── monitoring/
│   ├── dashboard.png
│   ├── metrics.csv
│   └── alerts.json
│
├── models/
│   ├── best_model_ep450/
│   ├── checkpoint_ep100/
│   └── latest/
│
├── tests/
│   ├── test_agents.py
│   ├── test_environment.py
│   └── test_integration.py
│
└── docs/
    ├── technical_report.pdf
```
---

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{multiagent_rl_2025,
  author = {Akshay},
  title = {Multi-Agent Reinforcement Learning System with CrewAI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/akshaybharadwaj11/AdCampaignOptimizationSystem}
}
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/desc`)
3. Commit your changes (`git commit -m 'feature desc'`)
4. Push to the branch (`git push origin feature/desc`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check code style
flake8 src/
black src/

# Build documentation
cd docs && make html
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CrewAI** - Agentic System Orchestration
- **OpenAI** - Gpt-4o-mini (Analytics Agent)
- **PyTorch** - RL Implementation
- **Optuna** - Hyperparameter optimization

---

## 📞 Contact

**Akshay**   
- GitHub: [@akshaybharadwaj11](https://github.com/akshaybharadwaj11)
- LinkedIn: [Akshay Bharadwaj](https://www.linkedin.com/in/akshay-bharadwaj-k-h/)
- Email: akshaybharadwaj456@gmail.com

---