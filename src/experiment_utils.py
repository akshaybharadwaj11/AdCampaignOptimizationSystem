"""
Multi-Agent RL System - : Experimental Framework & Analysis
==================================================================

This module contains:
- Baseline implementations
- Experimental framework
- Statistical validation
- Visualization utilities
- Hyperparameter optimization
- A/B testing framework
- Model versioning
- Comprehensive analytics

"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import pandas as pd
from datetime import datetime
import logging
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from collections import defaultdict
import optuna
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from collections import defaultdict, deque 

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# Baseline Implementations
# ============================================================================

class RandomAgent:
    """Random baseline for comparison"""
    def __init__(self, state_dim: int, action_dim: int = 3):
        self.action_dim = action_dim
        self.bid_levels = np.linspace(0.5, 5.0, 10)
        logger.info("Random Agent initialized")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select random action"""
        return np.random.randint(0, self.action_dim)
    
    def select_bid(self, state: np.ndarray) -> Tuple[float, int]:
        """Select random bid"""
        bid_idx = np.random.randint(0, len(self.bid_levels))
        return self.bid_levels[bid_idx], bid_idx
    
    def select_allocation(self, state: np.ndarray) -> Dict[str, float]:
        """Select random allocation"""
        allocation = np.random.dirichlet(np.ones(3))
        return {f"Channel_{i+1}": float(allocation[i]) for i in range(3)}
    
    def store_reward(self, reward: float, done: bool):
        pass
    
    def update(self):
        return {}


class FixedStrategyAgent:
    """Fixed strategy baseline"""
    def __init__(self, state_dim: int, fixed_bid: float = 2.5):
        self.fixed_bid = fixed_bid
        self.bid_idx = 5
        self.equal_allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
        logger.info(f"Fixed Strategy Agent initialized - Bid: ${fixed_bid}")
    
    def select_action(self, state: np.ndarray) -> int:
        """Always use both agents"""
        return 2
    
    def select_bid(self, state: np.ndarray) -> Tuple[float, int]:
        """Fixed bid"""
        return self.fixed_bid, self.bid_idx
    
    def select_allocation(self, state: np.ndarray) -> Dict[str, float]:
        """Equal allocation"""
        return self.equal_allocation
    
    def store_reward(self, reward: float, done: bool):
        pass
    
    def update(self):
        return {}


class GreedyAgent:
    """Greedy baseline that adjusts based on immediate feedback"""
    def __init__(self, state_dim: int):
        self.bid_levels = np.linspace(0.5, 5.0, 10)
        self.current_bid_idx = 5
        self.recent_rewards = deque(maxlen=10)
        self.allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
        logger.info("Greedy Agent initialized")
    
    def select_action(self, state: np.ndarray) -> int:
        """Use both agents"""
        return 2
    
    def select_bid(self, state: np.ndarray) -> Tuple[float, int]:
        """Adjust bid based on recent performance"""
        if len(self.recent_rewards) > 5:
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward < 0:
                self.current_bid_idx = max(0, self.current_bid_idx - 1)
            elif avg_reward > 10:
                self.current_bid_idx = min(len(self.bid_levels) - 1, self.current_bid_idx + 1)
        
        return self.bid_levels[self.current_bid_idx], self.current_bid_idx
    
    def select_allocation(self, state: np.ndarray) -> Dict[str, float]:
        """Fixed allocation"""
        return self.allocation
    
    def store_reward(self, reward: float, done: bool):
        """Store reward for greedy updates"""
        self.recent_rewards.append(reward)
    
    def update(self):
        return {}


# ============================================================================
# Experimental Framework
# ============================================================================

@dataclass
class ExperimentResults:
    """Store experimental results"""
    agent_name: str
    episode_rewards: List[float]
    episode_conversions: List[int]
    episode_spend: List[float]
    episode_roi: List[float]
    training_time: float
    config: Dict[str, Any]
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute summary statistics"""
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_conversions': np.mean(self.episode_conversions),
            'std_conversions': np.std(self.episode_conversions),
            'mean_spend': np.mean(self.episode_spend),
            'mean_roi': np.mean(self.episode_roi),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'final_50_mean': np.mean(self.episode_rewards[-50:])
        }


class ExperimentRunner:
    """Run and compare different agent configurations"""
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
        
        logger.info(f"Experiment Runner initialized - Output: {output_dir}")
    
    def run_baseline(self, agent, env, num_episodes: int = 100, 
                    agent_name: str = "baseline") -> ExperimentResults:
        """Run baseline agent"""
        import time
        start_time = time.time()
        
        episode_rewards = []
        episode_conversions = []
        episode_spend = []
        episode_roi = []
        
        logger.info(f"Running {agent_name} for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_array = state.to_array()
                
                # Get actions from baseline agent
                controller_action = agent.select_action(state_array)
                bid, bid_idx = agent.select_bid(state_array)
                allocation = agent.select_allocation(state_array)
                
                from src.agent_utils import Action  # Import from main module
                action = Action(
                    bid_amount=bid,
                    budget_allocation=allocation,
                    agent_type="both"
                )
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                
                agent.store_reward(reward, done)
            
            episode_rewards.append(episode_reward)
            episode_conversions.append(env.total_conversions)
            episode_spend.append(env.total_spend)
            
            if env.total_spend > 0:
                roi = (env.total_conversions * env.conversion_value - env.total_spend) / env.total_spend
                episode_roi.append(roi)
            else:
                episode_roi.append(0)
            
            agent.update()
            
            if episode % 20 == 0:
                logger.info(f"{agent_name} Episode {episode}: Reward={episode_reward:.2f}")
        
        training_time = time.time() - start_time
        
        results = ExperimentResults(
            agent_name=agent_name,
            episode_rewards=episode_rewards,
            episode_conversions=episode_conversions,
            episode_spend=episode_spend,
            episode_roi=episode_roi,
            training_time=training_time,
            config={"type": agent_name}
        )
        
        self.results[agent_name] = results
        logger.info(f"{agent_name} completed - Time: {training_time:.2f}s")
        
        return results
    
    def compare_agents(self, significance_level: float = 0.05) -> pd.DataFrame:
        """Statistical comparison of agents"""
        comparison_data = []
        
        for name, results in self.results.items():
            stats = results.compute_statistics()
            stats['agent'] = name
            comparison_data.append(stats)
        
        df = pd.DataFrame(comparison_data)
        
        # Perform pairwise t-tests
        agent_names = list(self.results.keys())
        if len(agent_names) >= 2:
            logger.info("\nPairwise Statistical Tests:")
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):
                    agent1 = agent_names[i]
                    agent2 = agent_names[j]
                    
                    rewards1 = self.results[agent1].episode_rewards
                    rewards2 = self.results[agent2].episode_rewards
                    
                    t_stat, p_value = scipy_stats.ttest_ind(rewards1, rewards2)
                    
                    logger.info(f"{agent1} vs {agent2}: t={t_stat:.3f}, p={p_value:.4f}")
                    if p_value < significance_level:
                        better = agent1 if np.mean(rewards1) > np.mean(rewards2) else agent2
                        logger.info(f"  → {better} is significantly better (p < {significance_level})")
        
        return df
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves for all agents"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        ax = axes[0, 0]
        for name, results in self.results.items():
            smoothed = pd.Series(results.episode_rewards).rolling(10, min_periods=1).mean()
            ax.plot(smoothed, label=name, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curves: Reward over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Conversions
        ax = axes[0, 1]
        for name, results in self.results.items():
            smoothed = pd.Series(results.episode_conversions).rolling(10, min_periods=1).mean()
            ax.plot(smoothed, label=name, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Conversions')
        ax.set_title('Conversions over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ROI
        ax = axes[1, 0]
        for name, results in self.results.items():
            smoothed = pd.Series(results.episode_roi).rolling(10, min_periods=1).mean()
            ax.plot(smoothed, label=name, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('ROI')
        ax.set_title('ROI over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Spend
        ax = axes[1, 1]
        for name, results in self.results.items():
            smoothed = pd.Series(results.episode_spend).rolling(10, min_periods=1).mean()
            ax.plot(smoothed, label=name, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Spend ($)')
        ax.set_title('Spend over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        else:
            plt.savefig(self.output_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_comparison_boxplot(self, save_path: Optional[str] = None):
        """Create box plots for agent comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare data
        reward_data = {name: results.episode_rewards[-50:] 
                      for name, results in self.results.items()}
        conversion_data = {name: results.episode_conversions[-50:] 
                          for name, results in self.results.items()}
        roi_data = {name: results.episode_roi[-50:] 
                   for name, results in self.results.items()}
        
        # Rewards
        ax = axes[0]
        ax.boxplot(reward_data.values(), labels=reward_data.keys())
        ax.set_ylabel('Reward')
        ax.set_title('Final 50 Episodes: Reward Distribution')
        ax.grid(True, alpha=0.3)
        
        # Conversions
        ax = axes[1]
        ax.boxplot(conversion_data.values(), labels=conversion_data.keys())
        ax.set_ylabel('Conversions')
        ax.set_title('Final 50 Episodes: Conversions Distribution')
        ax.grid(True, alpha=0.3)
        
        # ROI
        ax = axes[2]
        ax.boxplot(roi_data.values(), labels=roi_data.keys())
        ax.set_ylabel('ROI')
        ax.set_title('Final 50 Episodes: ROI Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "comparison_boxplot.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        report = ["=" * 80]
        report.append("EXPERIMENTAL RESULTS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        comparison_df = self.compare_agents()
        report.append("Summary Statistics:")
        report.append("-" * 80)
        report.append(comparison_df.to_string())
        report.append("")
        
        # Best agent
        best_agent = comparison_df.loc[comparison_df['mean_reward'].idxmax(), 'agent']
        best_reward = comparison_df.loc[comparison_df['mean_reward'].idxmax(), 'mean_reward']
        
        report.append(f"Best Performing Agent: {best_agent}")
        report.append(f"Mean Reward: {best_reward:.2f}")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "experiment_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def save_results(self):
        """Save all results to disk"""
        results_path = self.output_dir / "results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Results saved to {results_path}")


# ============================================================================
# Ablation Study Framework
# ============================================================================

class AblationStudy:
    """Conduct ablation studies to understand component importance"""
    def __init__(self, output_dir: str = "ablation_studies"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
        
        logger.info("Ablation Study Framework initialized")
    
    def test_without_controller(self, system, env, num_episodes: int = 100):
        """Test system without controller (fixed strategy)"""
        logger.info("Testing without controller...")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_array = state.to_array()
                
                # Fixed strategy: always use both agents
                bid, _ = system.bidding_agent.select_bid(state_array)
                allocation = system.budget_agent.select_allocation(state_array)
                
                from src.agent_utils import Action
                action = Action(bid_amount=bid, budget_allocation=allocation, agent_type="both")
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        self.results['no_controller'] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }
        
        logger.info(f"Without Controller: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    def test_without_bidding_agent(self, system, env, num_episodes: int = 100):
        """Test system with fixed bidding"""
        logger.info("Testing without bidding agent...")
        
        episode_rewards = []
        fixed_bid = 2.5
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_array = state.to_array()
                
                allocation = system.budget_agent.select_allocation(state_array)
                
                from src.agent_utils import Action
                action = Action(bid_amount=fixed_bid, budget_allocation=allocation, agent_type="budget")
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        self.results['no_bidding'] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }
        
        logger.info(f"Without Bidding Agent: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    def test_without_budget_agent(self, system, env, num_episodes: int = 100):
        """Test system with fixed budget allocation"""
        logger.info("Testing without budget agent...")
        
        episode_rewards = []
        fixed_allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_array = state.to_array()
                
                bid, _ = system.bidding_agent.select_bid(state_array)
                
                from src.agent_utils import Action
                action = Action(bid_amount=bid, budget_allocation=fixed_allocation, agent_type="bidding")
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        self.results['no_budget'] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }
        
        logger.info(f"Without Budget Agent: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    def plot_results(self, full_system_mean: float):
        """Plot ablation study results"""
        components = ['Full System'] + list(self.results.keys())
        means = [full_system_mean] + [r['mean_reward'] for r in self.results.values()]
        stds = [0] + [r['std_reward'] for r in self.results.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(components))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Color the full system bar differently
        bars[0].set_color('green')
        bars[0].set_alpha(0.8)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Ablation Study: Component Importance')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 5, f'{mean:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "ablation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Ablation plot saved")


# ============================================================================
# Hyperparameter Optimization with Optuna
# ============================================================================

class HyperparameterOptimizer:
    """Optimize hyperparameters using Optuna"""
    def __init__(self, n_trials: int = 50, output_dir: str = "hyperopt"):
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Hyperparameter Optimizer initialized - {n_trials} trials")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        # Suggest hyperparameters
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.99, 0.999)
        
        # Create config
        from src.agent_utils import AgentConfig, EnhancedControllerAgent, EnhancedSimulationAgent
        
        config = AgentConfig(
            learning_rate=lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epsilon_decay=epsilon_decay
        )
        
        # Train agent
        try:
            controller = EnhancedControllerAgent(state_dim=8, config=config)
            env = EnhancedSimulationAgent()
            
            episode_rewards = []
            
            for episode in range(50):  # Shortened for optimization
                state = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    state_array = state.to_array()
                    action, _, _ = controller.select_action(state_array)
                    
                    # Simplified action execution
                    from src.agent_utils import Action
                    act = Action(bid_amount=2.0, budget_allocation={}, agent_type="both")
                    
                    next_state, reward, done, info = env.step(act)
                    controller.store_reward(reward, done)
                    episode_reward += reward
                    state = next_state
                
                episode_rewards.append(episode_reward)
                
                if episode % 10 == 0:
                    controller.update()
            
            # Return mean of last 20 episodes
            return np.mean(episode_rewards[-20:])
        
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -1000.0
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.2f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Save results
        results_df = study.trials_dataframe()
        results_df.to_csv(self.output_dir / "optimization_results.csv", index=False)
        
        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(self.output_dir / "optimization_history.html"))
        
        # Plot parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(self.output_dir / "param_importances.html"))
        
        return study.best_params


# ============================================================================
# Enhanced CrewAI Analytics Agents
# ============================================================================

@tool("analyze_training_metrics")
def analyze_training_metrics(metrics_json: str) -> str:
    """Analyze comprehensive training metrics"""
    metrics = json.loads(metrics_json)
    
    analysis = f"""
TRAINING METRICS ANALYSIS
========================

Performance Metrics:
- Average Reward (last 50): {metrics['avg_reward']:.2f}
- Total Conversions: {metrics['total_conversions']}
- Average ROI: {metrics['avg_roi']:.2f}%
- Win Rate: {metrics['win_rate']:.1f}%

Learning Progress:
- Policy Loss: {metrics['policy_loss']:.4f}
- Value Loss: {metrics['value_loss']:.4f}
- Entropy: {metrics['entropy']:.4f}
- Epsilon: {metrics['epsilon']:.4f}

Assessment:
"""
    
    # Evaluate learning stability
    if metrics['policy_loss'] < 0.1:
        analysis += "✓ Policy has converged well\n"
    else:
        analysis += "⚠ Policy may need more training\n"
    
    if metrics['entropy'] > 0.5:
        analysis += "✓ Good exploration maintained\n"
    elif metrics['entropy'] < 0.1:
        analysis += "⚠ Low entropy - may be exploiting too much\n"
    
    if metrics['avg_roi'] > 50:
        analysis += "✓ Excellent ROI - profitable strategy\n"
    elif metrics['avg_roi'] > 0:
        analysis += "✓ Positive ROI - decent performance\n"
    else:
        analysis += "✗ Negative ROI - needs improvement\n"
    
    return analysis


@tool("generate_recommendations")
def generate_recommendations(performance_data: str) -> str:
    """Generate actionable recommendations"""
    data = json.loads(performance_data)
    
    recommendations = ["STRATEGIC RECOMMENDATIONS", "=" * 40, ""]
    
    if data['roi'] < 0:
        recommendations.append("1. CRITICAL: Reduce bid amounts to improve cost efficiency")
        recommendations.append("   Suggested action: Lower bid levels by 20-30%")
    
    if data['conversion_rate'] < 0.05:
        recommendations.append("2. Optimize conversion funnel")
        recommendations.append("   Suggested action: Improve landing page quality or targeting")
    
    if data['win_rate'] < 0.3:
        recommendations.append("3. Increase bid competitiveness")
        recommendations.append("   Suggested action: Raise bid levels by 10-15%")
    
    if data['budget_utilization'] < 0.5:
        recommendations.append("4. Improve budget pacing")
        recommendations.append("   Suggested action: Be more aggressive in early episodes")
    
    if len(recommendations) == 3:  # Only header
        recommendations.append("✓ System performing well - maintain current strategy")
    
    return "\n".join(recommendations)


@tool("detect_training_issues")
def detect_training_issues(training_log: str) -> str:
    """Detect common training issues"""
    log = json.loads(training_log)
    
    issues = []
    
    # Check for divergence
    if log['reward_variance'] > 10000:
        issues.append("CRITICAL: High reward variance indicates training instability")
    
    # Check for policy collapse
    if log['entropy'] < 0.05:
        issues.append("WARNING: Very low entropy - policy may have collapsed")
    
    # Check for gradient issues
    if 'gradient_norm' in log and log['gradient_norm'] > 10:
        issues.append("WARNING: Large gradients detected - consider gradient clipping")
    
    # Check for Q-value explosion
    if 'max_q_value' in log and abs(log['max_q_value']) > 1000:
        issues.append("CRITICAL: Q-values exploding - reduce learning rate")
    
    if not issues:
        return "No critical training issues detected. System healthy."
    
    return "TRAINING ISSUES DETECTED:\n" + "\n".join(f"- {issue}" for issue in issues)


def create_analytics_crew() -> Crew:
    """Create comprehensive analytics crew"""
    
    # Performance Analyst
    performance_analyst = Agent(
        role='RL Performance Analyst',
        goal='Analyze agent performance and identify optimization opportunities',
        backstory="""Expert in reinforcement learning with 10 years of experience
        optimizing multi-agent systems for marketing applications.""",
        tools=[analyze_training_metrics, generate_recommendations],
        verbose=True,
        allow_delegation=False
    )
    
    # System Health Monitor
    system_monitor = Agent(
        role='System Health Monitor',
        goal='Detect training issues and ensure system stability',
        backstory="""Specialist in debugging RL systems and identifying common
        failure modes like policy collapse, reward hacking, and training divergence.""",
        tools=[detect_training_issues],
        verbose=True,
        allow_delegation=False
    )
    
    # Tasks
    analysis_task = Task(
        description="Analyze current training metrics and provide detailed performance assessment",
        expected_output="Comprehensive analysis of agent performance with key insights",
        agent=performance_analyst
    )
    
    monitoring_task = Task(
        description="Monitor system health and detect any training issues",
        expected_output="Report on system health with any warnings or critical issues",
        agent=system_monitor
    )
    
    recommendation_task = Task(
        description="Generate actionable recommendations based on analysis",
        expected_output="Prioritized list of recommendations for improvement",
        agent=performance_analyst
    )
    
    # Create crew
    crew = Crew(
        agents=[performance_analyst, system_monitor],
        tasks=[analysis_task, monitoring_task, recommendation_task],
        process=Process.sequential,
        verbose=False
    )
    
    return crew


# ============================================================================
# Model Versioning and Management
# ============================================================================

class ModelRegistry:
    """Manage model versions and checkpoints"""
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_file = self.registry_dir / "registry.json"
        
        # Load existing registry
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': []}
        
        logger.info(f"Model Registry initialized - {len(self.metadata['models'])} models tracked")
    
    def save_model(self, model, model_name: str, metrics: Dict[str, float],
                   config: Dict[str, Any], tags: List[str] = None) -> str:
        """Save model with metadata"""
        version = len([m for m in self.metadata['models'] if m['name'] == model_name]) + 1
        version_str = f"v{version}"
        
        # Create model directory
        model_dir = self.registry_dir / model_name / version_str
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_path = model_dir / "model.pt"
        model.save(str(model_path))
        
        # Save metadata
        metadata = {
            'name': model_name,
            'version': version_str,
            'path': str(model_path),
            'metrics': metrics,
            'config': config,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit()
        }
        
        self.metadata['models'].append(metadata)
        self._save_metadata()
        
        logger.info(f"Model saved: {model_name} {version_str}")
        
        return version_str
    
    def load_model(self, model_name: str, version: str = "latest"):
        """Load model by name and version"""
        models = [m for m in self.metadata['models'] if m['name'] == model_name]
        
        if not models:
            raise ValueError(f"No models found with name: {model_name}")
        
        if version == "latest":
            model_meta = models[-1]
        else:
            model_meta = next((m for m in models if m['version'] == version), None)
            if not model_meta:
                raise ValueError(f"Version {version} not found for {model_name}")
        
        logger.info(f"Loading {model_name} {model_meta['version']}")
        
        return model_meta['path'], model_meta
    
    def list_models(self) -> pd.DataFrame:
        """List all registered models"""
        if not self.metadata['models']:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metadata['models'])
        return df[['name', 'version', 'timestamp'] + 
                 [col for col in df.columns if col not in ['name', 'version', 'timestamp', 'path', 'config']]]
    
    def compare_versions(self, model_name: str, metric: str = 'mean_reward'):
        """Compare versions of a model"""
        models = [m for m in self.metadata['models'] if m['name'] == model_name]
        
        versions = [m['version'] for m in models]
        values = [m['metrics'].get(metric, 0) for m in models]
        
        plt.figure(figsize=(10, 6))
        plt.plot(versions, values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Version')
        plt.ylabel(metric)
        plt.title(f'{model_name}: {metric} across versions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.registry_dir / f"{model_name}_version_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Version comparison saved to {save_path}")
    
    def _save_metadata(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return None
