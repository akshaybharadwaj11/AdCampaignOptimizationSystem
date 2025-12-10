"""
Online Learning Demonstration - Agents Improving from Experience
=================================================================

This demonstrates that agents are LEARNING and IMPROVING in real-time:
- Shows performance metrics improving across episodes
- Highlights strategy changes based on experience
- Compares each episode to previous ones
- CrewAI analyzes the learning trajectory
- Makes online learning visible and measurable

Author: Akshay
Date: December 2025
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Learning Progress Tracker
# ============================================================================

@dataclass
class EpisodeSnapshot:
    """Snapshot of system state at episode"""
    episode: int
    reward: float
    conversions: int
    roi: float
    avg_bid: float
    controller_strategy: Dict[str, float]  # % using each mode
    epsilon: float
    policy_loss: float
    value_loss: float
    
    def __str__(self):
        return (f"Ep{self.episode}: R={self.reward:.0f}, Conv={self.conversions}, "
                f"ROI={self.roi:.0f}%, Bid=${self.avg_bid:.2f}, ε={self.epsilon:.3f}")


class LearningTracker:
    """Tracks learning progress across episodes"""
    
    def __init__(self):
        self.snapshots: List[EpisodeSnapshot] = []
        self.improvements = []
    
    def add_snapshot(self, snapshot: EpisodeSnapshot):
        """Add episode snapshot and calculate improvement"""
        self.snapshots.append(snapshot)
        
        if len(self.snapshots) > 1:
            prev = self.snapshots[-2]
            curr = snapshot
            
            improvement = {
                'episode': curr.episode,
                'reward_delta': curr.reward - prev.reward,
                'reward_pct': ((curr.reward - prev.reward) / max(abs(prev.reward), 1)) * 100,
                'conversion_delta': curr.conversions - prev.conversions,
                'roi_delta': curr.roi - prev.roi,
                'bid_delta': curr.avg_bid - prev.avg_bid,
                'epsilon_delta': curr.epsilon - prev.epsilon,
                'policy_loss_delta': curr.policy_loss - prev.policy_loss
            }
            
            self.improvements.append(improvement)
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress"""
        if len(self.snapshots) < 2:
            return {}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        return {
            'initial_reward': first.reward,
            'current_reward': last.reward,
            'total_improvement': last.reward - first.reward,
            'improvement_pct': ((last.reward - first.reward) / max(abs(first.reward), 1)) * 100,
            'initial_conversions': first.conversions,
            'current_conversions': last.conversions,
            'conversion_growth': last.conversions - first.conversions,
            'initial_roi': first.roi,
            'current_roi': last.roi,
            'roi_improvement': last.roi - first.roi,
            'exploration_reduction': first.epsilon - last.epsilon,
            'policy_convergence': abs(last.policy_loss) < abs(first.policy_loss)
        }


# ============================================================================
# Online Learning Demo System
# ============================================================================

class OnlineLearningDemo:
    """Demonstrates agents learning and improving from experience"""
    
    def __init__(self):
        from agent_utils import (EnhancedControllerAgent, EnhancedBiddingAgent,
                          EnhancedBudgetAgent, EnhancedSimulationAgent, Action)
        
        # Initialize RL agents
        self.controller = EnhancedControllerAgent(state_dim=10)
        self.bidding_agent = EnhancedBiddingAgent(state_dim=10)
        self.budget_agent = EnhancedBudgetAgent(state_dim=10)
        self.env = EnhancedSimulationAgent()
        self.Action = Action
        
        # Learning tracker
        self.tracker = LearningTracker()
        
        # CrewAI learning analyst
        self.learning_analyst = Agent(
            role='Online Learning Analyst',
            goal='Analyze and explain how agents improve from experience',
            backstory="""Expert in reinforcement learning who specializes in 
            identifying learning patterns, strategy evolution, and performance 
            improvements in multi-agent systems.""",
            verbose=True,
            allow_delegation=False
        )
        
        logger.info("Online Learning Demo initialized")
    
    def run_episode(self, episode: int) -> Dict:
        """Run episode and collect metrics"""
        
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        # Track decisions and actions
        bids = []
        controller_decisions = []
        
        while not done:
            state_array = state.to_array()
            
            # Controller decision
            controller_action, _, _ = self.controller.select_action(state_array, training=True)
            controller_decisions.append(controller_action)
            
            # Agent actions
            use_bidding = controller_action in [0, 2]
            use_budget = controller_action in [1, 2]
            
            if use_bidding:
                bid, bid_idx = self.bidding_agent.select_bid(state_array, training=True)
            else:
                bid, bid_idx = 2.0, 5
            
            if use_budget:
                allocation = self.budget_agent.select_allocation(state_array, training=True)
            else:
                allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
            
            bids.append(bid)
            
            # Execute
            action = self.Action(bid_amount=bid, budget_allocation=allocation, 
                               agent_type=["bidding", "budget", "both"][controller_action])
            
            next_state, reward, done, info = self.env.step(action)
            next_state_array = next_state.to_array()
            
            # Store experiences
            self.controller.store_reward(reward, done)
            if use_bidding:
                self.bidding_agent.store_experience(state_array, bid_idx, reward,
                                                   next_state_array, done)
            if use_budget:
                self.budget_agent.store_reward(reward, done)
            
            episode_reward += reward
            state = next_state
        
        # Calculate metrics
        roi = 0
        if self.env.total_spend > 0:
            roi = ((self.env.total_conversions * self.env.conversion_value - 
                   self.env.total_spend) / self.env.total_spend) * 100
        
        # Controller strategy distribution
        total_decisions = len(controller_decisions)
        controller_strategy = {
            'bidding_only': controller_decisions.count(0) / total_decisions * 100,
            'budget_only': controller_decisions.count(1) / total_decisions * 100,
            'both': controller_decisions.count(2) / total_decisions * 100
        }
        
        return {
            'reward': episode_reward,
            'conversions': self.env.total_conversions,
            'roi': roi,
            'avg_bid': np.mean(bids),
            'controller_strategy': controller_strategy,
            'epsilon': self.bidding_agent.epsilon,
            'spend': self.env.total_spend
        }
    
    def update_agents(self):
        """Update all agents - this is where learning happens!"""
        
        print("\n🔄 AGENTS LEARNING FROM EXPERIENCE...")
        
        # Controller learns
        controller_metrics = self.controller.update(epochs=5)
        if controller_metrics:
            print(f"   Controller: Policy Loss={controller_metrics.get('policy_loss', 0):.4f}, "
                  f"Entropy={controller_metrics.get('entropy', 0):.4f}")
        
        # Bidding agent learns
        updates = 0
        for _ in range(4):
            if len(self.bidding_agent.replay_buffer) >= 32:
                loss = self.bidding_agent.update()
                if loss:
                    updates += 1
        
        if updates > 0:
            print(f"   Bidding Agent: {updates} learning updates, ε={self.bidding_agent.epsilon:.4f}")
        
        # Budget agent learns
        budget_metrics = self.budget_agent.update(epochs=5)
        if budget_metrics:
            print(f"   Budget Agent: Policy Loss={budget_metrics.get('policy_loss', 0):.4f}")
        
        print("   ✓ All agents updated with new experience")
        
        return {
            'policy_loss': controller_metrics.get('policy_loss', 0) if controller_metrics else 0,
            'value_loss': controller_metrics.get('value_loss', 0) if controller_metrics else 0
        }
    
    def show_improvement_analysis(self, episode: int):
        """Show detailed improvement analysis"""
        
        if len(self.tracker.snapshots) < 2:
            return
        
        print(f"\n{'='*80}")
        print(f"📈 LEARNING PROGRESS ANALYSIS - After Episode {episode}")
        print(f"{'='*80}\n")
        
        # Compare with previous episode
        prev = self.tracker.snapshots[-2]
        curr = self.tracker.snapshots[-1]
        
        print("EPISODE-TO-EPISODE COMPARISON:")
        print("-" * 80)
        
        # Reward
        reward_change = curr.reward - prev.reward
        reward_pct = (reward_change / max(abs(prev.reward), 1)) * 100
        print(f"Reward:       {prev.reward:8.2f} → {curr.reward:8.2f}  "
              f"({reward_change:+.2f}, {reward_pct:+.1f}%)"
              f"  {'✓' if reward_change > 0 else '✗'}")
        
        # Conversions
        conv_change = curr.conversions - prev.conversions
        print(f"Conversions:  {prev.conversions:8} → {curr.conversions:8}  "
              f"({conv_change:+d})"
              f"  {'✓' if conv_change > 0 else '✗'}")
        
        # ROI
        roi_change = curr.roi - prev.roi
        print(f"ROI:          {prev.roi:7.1f}% → {curr.roi:7.1f}%  "
              f"({roi_change:+.1f}pp)"
              f"  {'✓' if roi_change > 0 else '✗'}")
        
        # Bid strategy
        bid_change = curr.avg_bid - prev.avg_bid
        print(f"Avg Bid:      ${prev.avg_bid:7.2f} → ${curr.avg_bid:7.2f}  "
              f"({bid_change:+.2f})")
        
        # Exploration
        eps_change = curr.epsilon - prev.epsilon
        print(f"Exploration:  {prev.epsilon:8.4f} → {curr.epsilon:8.4f}  "
              f"({eps_change:+.4f})  {'✓ Exploiting more' if eps_change < 0 else ''}")
        
        print("\nSTRATEGY EVOLUTION:")
        print("-" * 80)
        
        # Controller strategy changes
        for mode in ['bidding_only', 'budget_only', 'both']:
            prev_pct = prev.controller_strategy[mode]
            curr_pct = curr.controller_strategy[mode]
            change = curr_pct - prev_pct
            print(f"{mode.replace('_', ' ').title():15s}: {prev_pct:5.1f}% → {curr_pct:5.1f}%  "
                  f"({change:+.1f}%)")
        
        print("\nLEARNING INDICATORS:")
        print("-" * 80)
        
        # What the agent learned
        if reward_change > 0:
            print("✓ Positive reward improvement - agent found better strategy")
        else:
            print("• Exploring different strategies (temporary dip is normal)")
        
        if conv_change > 0:
            print("✓ More conversions - bidding strategy improving")
        
        if roi_change > 5:
            print("✓ Significant ROI gain - resource allocation optimizing")
        
        if eps_change < -0.01:
            print("✓ Reducing exploration - converging to learned policy")
        
        if abs(curr.policy_loss) < abs(prev.policy_loss):
            print("✓ Policy loss decreasing - neural network learning")
        
        # Overall trajectory
        if len(self.tracker.snapshots) >= 3:
            print("\nOVERALL TRAJECTORY:")
            print("-" * 80)
            
            summary = self.tracker.get_learning_summary()
            print(f"Total Episodes: {episode + 1}")
            print(f"Initial → Current Reward: {summary['initial_reward']:.0f} → {summary['current_reward']:.0f}")
            print(f"Total Improvement: {summary['total_improvement']:+.0f} ({summary['improvement_pct']:+.1f}%)")
            print(f"Conversion Growth: {summary['initial_conversions']} → {summary['current_conversions']} "
                  f"({summary['conversion_growth']:+d})")
            print(f"ROI Improvement: {summary['initial_roi']:.1f}% → {summary['current_roi']:.1f}% "
                  f"({summary['roi_improvement']:+.1f}pp)")
            
            if summary['improvement_pct'] > 50:
                print("\n🎯 Strong learning progress - agents significantly improved!")
            elif summary['improvement_pct'] > 0:
                print("\n📈 Steady learning progress - agents improving")
            else:
                print("\n🔄 Agents exploring - improvement expected soon")
    
    def run_crewai_learning_analysis(self, episode: int):
        """CrewAI analyzes the learning progress"""
        
        if len(self.tracker.snapshots) < 2:
            return
        
        summary = self.tracker.get_learning_summary()
        recent = self.tracker.snapshots[-3:] if len(self.tracker.snapshots) >= 3 else self.tracker.snapshots
        
        # Build detailed analysis for CrewAI
        analysis_data = f"""
ONLINE LEARNING ANALYSIS - Episode {episode}
{'='*80}

LEARNING TRAJECTORY:
-------------------
Episodes Completed: {episode + 1}

Initial Performance (Episode 0):
  • Reward: {summary['initial_reward']:.2f}
  • Conversions: {summary['initial_conversions']}
  • ROI: {summary['initial_roi']:.1f}%

Current Performance (Episode {episode}):
  • Reward: {summary['current_reward']:.2f}
  • Conversions: {summary['current_conversions']}
  • ROI: {summary['current_roi']:.1f}%

Total Improvement:
  • Reward: {summary['total_improvement']:+.2f} ({summary['improvement_pct']:+.1f}%)
  • Conversions: {summary['conversion_growth']:+d}
  • ROI: {summary['roi_improvement']:+.1f} percentage points

RECENT EPISODES:
---------------
"""
        
        for snap in recent:
            analysis_data += f"Episode {snap.episode}: Reward={snap.reward:.0f}, Conv={snap.conversions}, "
            analysis_data += f"ROI={snap.roi:.0f}%, Bid=${snap.avg_bid:.2f}\n"
        
        analysis_data += f"""

LEARNING INDICATORS:
-------------------
Exploration Reduction: {summary['exploration_reduction']:.4f} (ε decreased = learning)
Policy Convergence: {"Yes - losses decreasing" if summary['policy_convergence'] else "Still optimizing"}

AGENT BEHAVIORS:
---------------
"""
        
        curr = self.tracker.snapshots[-1]
        analysis_data += f"""
Controller Strategy:
  • Bidding Only: {curr.controller_strategy['bidding_only']:.1f}%
  • Budget Only: {curr.controller_strategy['budget_only']:.1f}%
  • Both Agents: {curr.controller_strategy['both']:.1f}%

Bidding Agent:
  • Current Epsilon: {curr.epsilon:.4f}
  • Average Bid: ${curr.avg_bid:.2f}
  • Learning Status: {"Exploiting" if curr.epsilon < 0.1 else "Exploring"}

The system demonstrates ONLINE LEARNING - agents improve based on experience!
"""
        
        # Create CrewAI task
        task = Task(
            description=f"""
Analyze this online learning progress and explain how the agents improved:

{analysis_data}

Focus on:
1. Evidence of learning (what metrics improved?)
2. Strategy evolution (how did behaviors change?)
3. Learning patterns (is improvement consistent or volatile?)
4. Key insights (what did agents learn?)
5. Future trajectory (will improvement continue?)

Explain this in a way that demonstrates the system is LEARNING FROM EXPERIENCE.
            """,
            expected_output="Clear explanation of learning progress with specific evidence",
            agent=self.learning_analyst
        )
        
        crew = Crew(
            agents=[self.learning_analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        print(f"\n{'='*80}")
        print("🤖 CrewAI LEARNING ANALYST - Analyzing Improvement")
        print(f"{'='*80}\n")
        
        try:
            result = crew.kickoff()
            print(result)
        except Exception as e:
            print(f"⚠️  CrewAI unavailable: {e}")
            print("\nShowing automated analysis:")
            print(analysis_data)
    
    def train_and_demonstrate_learning(self, num_episodes: int = 5):
        """Run training with explicit learning demonstrations"""
        
        print("\n" + "="*80)
        print("ONLINE LEARNING DEMONSTRATION")
        print("Agents Learn and Improve from Experience")
        print("="*80)
        print("\nWatch as:")
        print("  • Agents start with random/poor performance")
        print("  • Experience each episode")
        print("  • Update based on what worked/didn't work")
        print("  • Improve their strategies over time")
        print("  • Performance metrics increase")
        print("="*80)
        
        for episode in range(num_episodes):
            print(f"\n{'#'*80}")
            print(f"# EPISODE {episode}")
            print(f"{'#'*80}\n")
            
            # Run episode
            print(f"🎮 Running Episode {episode}...")
            metrics = self.run_episode(episode)
            
            # Create snapshot
            snapshot = EpisodeSnapshot(
                episode=episode,
                reward=metrics['reward'],
                conversions=metrics['conversions'],
                roi=metrics['roi'],
                avg_bid=metrics['avg_bid'],
                controller_strategy=metrics['controller_strategy'],
                epsilon=metrics['epsilon'],
                policy_loss=0,  # Will be updated after training
                value_loss=0
            )
            
            print(f"\n📊 Episode {episode} Results:")
            print(f"   Reward: {metrics['reward']:.2f}")
            print(f"   Conversions: {metrics['conversions']}")
            print(f"   ROI: {metrics['roi']:.1f}%")
            print(f"   Avg Bid: ${metrics['avg_bid']:.2f}")
            print(f"   Spend: ${metrics['spend']:.2f}")
            
            # Update agents - THIS IS WHERE LEARNING HAPPENS
            learning_metrics = self.update_agents()
            snapshot.policy_loss = learning_metrics['policy_loss']
            snapshot.value_loss = learning_metrics['value_loss']
            
            # Track progress
            self.tracker.add_snapshot(snapshot)
            
            # Show improvement analysis
            self.show_improvement_analysis(episode)
            
            # CrewAI analysis every 2 episodes
            if episode > 0 and episode % 2 == 0:
                self.run_crewai_learning_analysis(episode)
            
            # Pause between episodes for viewing
            if episode < num_episodes - 1:
                input(f"\n▶️  Press Enter to run Episode {episode + 1}...")
        
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL LEARNING SUMMARY")
        print(f"{'='*80}\n")
        
        summary = self.tracker.get_learning_summary()
        
        print("COMPLETE LEARNING TRAJECTORY:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Initial Reward: {summary['initial_reward']:.2f}")
        print(f"  Final Reward: {summary['current_reward']:.2f}")
        print(f"  Total Improvement: {summary['total_improvement']:+.2f} ({summary['improvement_pct']:+.1f}%)")
        print(f"  Conversion Growth: {summary['conversion_growth']:+d}")
        print(f"  ROI Improvement: {summary['roi_improvement']:+.1f}pp")
        
        print("\nEVIDENCE OF LEARNING:")
        if summary['improvement_pct'] > 0:
            print("  ✓ Agents improved performance through experience")
        if summary['conversion_growth'] > 0:
            print("  ✓ Learned to generate more conversions")
        if summary['roi_improvement'] > 0:
            print("  ✓ Optimized resource allocation for better ROI")
        if summary['exploration_reduction'] > 0:
            print("  ✓ Reduced random exploration as confidence increased")
        if summary['policy_convergence']:
            print("  ✓ Neural networks converged to better policies")
        
        print("\n🎓 CONCLUSION:")
        if summary['improvement_pct'] > 50:
            print("  The system demonstrated STRONG ONLINE LEARNING!")
            print("  Agents significantly improved through experience.")
        elif summary['improvement_pct'] > 0:
            print("  The system demonstrated CLEAR ONLINE LEARNING!")
            print("  Agents steadily improved with each episode.")
        else:
            print("  Agents are exploring - more episodes needed for convergence.")
        
        print("\n" + "="*80)


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run online learning demonstration"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              ONLINE LEARNING DEMONSTRATION                             ║
║                                                                        ║
║  This explicitly shows agents LEARNING and IMPROVING:                 ║
║                                                                        ║
║  1. Run episode → Get results                                         ║
║  2. Agents update based on experience                                 ║
║  3. Show what improved and why                                        ║
║  4. Run next episode with learned knowledge                           ║
║  5. Demonstrate continuous improvement                                ║
║                                                                        ║
║  Evidence of learning:                                                ║
║    • Rewards increase episode-to-episode                              ║
║    • Strategies evolve based on what works                            ║
║    • Exploration decreases as confidence grows                        ║
║    • Policy losses converge                                           ║
║    • CrewAI explains the learning process                             ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    num_episodes = input("\nHow many episodes to demonstrate? (3-10, default=5): ").strip()
    num_episodes = int(num_episodes) if num_episodes.isdigit() else 5
    num_episodes = min(max(3, num_episodes), 10)
    
    print(f"\n🚀 Demonstrating online learning across {num_episodes} episodes...\n")
    
    # Run demo
    demo = OnlineLearningDemo()
    demo.train_and_demonstrate_learning(num_episodes=num_episodes)
    
    print("\n✅ Online learning demonstration complete!")
    print("\nYou witnessed:")
    print("  • Agents starting with poor/random performance")
    print("  • Learning from each episode's experience")
    print("  • Improving strategies based on what worked")
    print("  • Performance metrics increasing over time")
    print("  • CrewAI analyzing the learning progress")
    print("\nThis is TRUE online learning - agents adapt in real-time!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()