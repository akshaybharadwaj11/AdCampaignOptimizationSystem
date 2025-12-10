"""
Complete Multi-Agent Flow with CrewAI Orchestration
====================================================

This implements the full cycle:
1. SimulationAgent → ControllerAgent (state)
2. ControllerAgent → BiddingAgent & BudgetAgent (action queries)
3. BiddingAgent → ControllerAgent (bid proposal)
4. BudgetAgent → ControllerAgent (allocation proposal)
5. ControllerAgent → SimulationAgent (combined action)
6. SimulationAgent → All RL Agents (reward, next state)
7. AnalyticsAgent (CrewAI) → Reads logs, generates insights

Author: Akshay
Date: December 2025
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from crewai import Agent, Task, Crew, Process
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Step Logging System
# ============================================================================

class StepLogger:
    """Logs detailed step-by-step interactions for CrewAI analysis"""
    
    def __init__(self):
        self.episode_logs = []
        self.current_episode = []
        
    def log_step(self, step_num: int, event: str, data: Dict):
        """Log a single step event"""
        log_entry = {
            'step': step_num,
            'event': event,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self.current_episode.append(log_entry)
        
    def finish_episode(self, episode_num: int, metrics: Dict):
        """Finish current episode and prepare for CrewAI analysis"""
        episode_summary = {
            'episode': episode_num,
            'steps': self.current_episode,
            'metrics': metrics,
            'total_steps': len(self.current_episode)
        }
        self.episode_logs.append(episode_summary)
        self.current_episode = []
        return episode_summary
    
    def get_recent_episodes(self, n: int = 5):
        """Get last N episodes for analysis"""
        return self.episode_logs[-n:]


# ============================================================================
# CrewAI Analytics Agent
# ============================================================================

def create_analytics_agent():
    """Create the CrewAI analytics agent that monitors RL agents"""
    
    analyst = Agent(
        role='Real-Time RL System Monitor',
        goal='Monitor multi-agent RL system and provide actionable insights',
        backstory="""You are an expert AI system analyst specializing in multi-agent 
        reinforcement learning. You monitor the interaction between Controller, Bidding, 
        and Budget agents in real-time, identifying patterns, issues, and opportunities 
        for optimization. Your insights help improve the system's performance.""",
        verbose=True,
        allow_delegation=False
    )
    
    return analyst


def analyze_episode_flow(episode_data: Dict) -> str:
    """Analyze the complete flow of an episode"""
    
    steps = episode_data['steps']
    metrics = episode_data['metrics']
    episode = episode_data['episode']
    
    # Extract key information
    controller_decisions = [s['data'].get('controller_action') for s in steps 
                           if s['event'] == 'controller_decision']
    bids = [s['data'].get('bid') for s in steps 
            if s['event'] == 'bidding_proposal']
    allocations = [s['data'].get('allocation') for s in steps 
                  if s['event'] == 'budget_proposal']
    rewards = [s['data'].get('reward') for s in steps 
              if s['event'] == 'reward_received']
    
    analysis = f"""
EPISODE {episode} FLOW ANALYSIS
{'='*80}

STEP-BY-STEP ORCHESTRATION:
---------------------------
Total Steps: {len(steps)}
Controller Decisions: {len(controller_decisions)}
Bidding Proposals: {len(bids)}
Budget Proposals: {len(allocations)}

CONTROLLER BEHAVIOR:
-------------------
"""
    
    # Analyze controller decisions
    if controller_decisions:
        decision_dist = {
            'bidding_only': controller_decisions.count(0),
            'budget_only': controller_decisions.count(1),
            'both': controller_decisions.count(2)
        }
        total = len(controller_decisions)
        
        analysis += f"""
Decision Distribution:
  • Bidding Only: {decision_dist['bidding_only']/total*100:.1f}%
  • Budget Only: {decision_dist['budget_only']/total*100:.1f}%
  • Both Agents: {decision_dist['both']/total*100:.1f}%

Strategy: """
        
        if decision_dist['both'] / total > 0.6:
            analysis += "Controller prefers coordinated multi-agent approach\n"
        elif decision_dist['bidding_only'] / total > 0.5:
            analysis += "Controller relies heavily on bidding optimization\n"
        else:
            analysis += "Controller uses mixed strategy\n"
    
    analysis += f"""
BIDDING AGENT PERFORMANCE:
-------------------------
"""
    
    if bids:
        valid_bids = [b for b in bids if b is not None]
        if valid_bids:
            analysis += f"""
Average Bid: ${np.mean(valid_bids):.2f}
Bid Range: ${min(valid_bids):.2f} - ${max(valid_bids):.2f}
Bid Std Dev: ${np.std(valid_bids):.2f}

Bidding Pattern: """
            
            avg_bid = np.mean(valid_bids)
            if avg_bid > 3.5:
                analysis += "Aggressive (high bids for maximum reach)\n"
            elif avg_bid > 2.0:
                analysis += "Moderate (balanced cost-benefit approach)\n"
            else:
                analysis += "Conservative (cost-focused strategy)\n"
    
    analysis += f"""
BUDGET AGENT PERFORMANCE:
------------------------
"""
    
    if allocations:
        # Aggregate allocations
        all_channel_1 = []
        all_channel_2 = []
        all_channel_3 = []
        
        for alloc in allocations:
            if alloc:
                all_channel_1.append(alloc.get('Channel_1', 0))
                all_channel_2.append(alloc.get('Channel_2', 0))
                all_channel_3.append(alloc.get('Channel_3', 0))
        
        if all_channel_1:
            analysis += f"""
Average Allocation:
  • Channel 1: {np.mean(all_channel_1)*100:.1f}%
  • Channel 2: {np.mean(all_channel_2)*100:.1f}%
  • Channel 3: {np.mean(all_channel_3)*100:.1f}%

Strategy: """
            
            max_channel = max(np.mean(all_channel_1), 
                            np.mean(all_channel_2), 
                            np.mean(all_channel_3))
            
            if max_channel > 0.5:
                analysis += "Concentrated (focuses on best performer)\n"
            elif max_channel > 0.4:
                analysis += "Moderately focused (2-channel emphasis)\n"
            else:
                analysis += "Diversified (balanced across channels)\n"
    
    analysis += f"""
EPISODE RESULTS:
---------------
Total Reward: {metrics['reward']:.2f}
Conversions: {metrics['conversions']}
Clicks: {metrics['clicks']}
Spend: ${metrics['spend']:.2f}
ROI: {metrics['roi']:.1f}%

REWARD FLOW:
-----------
"""
    
    if rewards:
        valid_rewards = [r for r in rewards if r is not None]
        if valid_rewards:
            positive_steps = sum(1 for r in valid_rewards if r > 0)
            negative_steps = sum(1 for r in valid_rewards if r < 0)
            
            analysis += f"""
Positive Reward Steps: {positive_steps} ({positive_steps/len(valid_rewards)*100:.1f}%)
Negative Reward Steps: {negative_steps} ({negative_steps/len(valid_rewards)*100:.1f}%)
Average Step Reward: {np.mean(valid_rewards):.2f}

Reward Pattern: """
            
            if positive_steps / len(valid_rewards) > 0.6:
                analysis += "Predominantly positive - good strategy\n"
            elif positive_steps / len(valid_rewards) > 0.4:
                analysis += "Mixed rewards - learning in progress\n"
            else:
                analysis += "Needs improvement - too many negative steps\n"
    
    analysis += f"""
MULTI-AGENT COORDINATION:
------------------------
The episode demonstrates:
"""
    
    if metrics['reward'] > 500:
        analysis += "✓ Excellent coordination - agents working effectively together\n"
    elif metrics['reward'] > 0:
        analysis += "✓ Good coordination - positive synergy between agents\n"
    else:
        analysis += "⚠ Coordination needs work - agents not yet aligned\n"
    
    if metrics['conversions'] > 30:
        analysis += "✓ High conversion success - bidding agent performing well\n"
    
    if metrics['roi'] > 100:
        analysis += "✓ Strong ROI - budget allocation is effective\n"
    
    analysis += f"""
RECOMMENDATIONS:
---------------
"""
    
    # Generate specific recommendations
    if metrics['reward'] < 0:
        analysis += "1. Focus on reducing negative reward steps\n"
        analysis += "2. Controller should experiment with different agent combinations\n"
    
    if metrics['roi'] < 50:
        analysis += "1. Bidding agent should reduce bid amounts\n"
        analysis += "2. Budget agent should concentrate on best-performing channel\n"
    
    if metrics['conversions'] < 20:
        analysis += "1. Increase bid competitiveness for better auction wins\n"
        analysis += "2. Review targeting and quality score factors\n"
    
    if metrics['reward'] > 1000:
        analysis += "1. Excellent performance - maintain current strategy\n"
        analysis += "2. Consider scaling budget to capitalize on success\n"
        analysis += "3. Document successful patterns for replication\n"
    
    analysis += "\n" + "="*80
    
    return analysis


# ============================================================================
# Integrated Training Loop
# ============================================================================

class IntegratedMultiAgentSystem:
    """Multi-agent system with integrated CrewAI orchestration"""
    
    def __init__(self):
        from agent_utils import (EnhancedControllerAgent, EnhancedBiddingAgent,
                          EnhancedBudgetAgent, EnhancedSimulationAgent, Action)
        
        # Initialize RL agents
        self.controller = EnhancedControllerAgent(state_dim=10)
        self.bidding_agent = EnhancedBiddingAgent(state_dim=10)
        self.budget_agent = EnhancedBudgetAgent(state_dim=10)
        self.env = EnhancedSimulationAgent()
        self.Action = Action
        
        # Initialize logging and analytics
        self.step_logger = StepLogger()
        self.analytics_agent = create_analytics_agent()
        
        # Metrics
        self.episode_metrics = []
        
        logger.info("Integrated Multi-Agent System initialized")
    
    def run_episode_with_logging(self, episode: int, training: bool = True):
        """Run single episode with detailed logging for CrewAI analysis"""
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode} - DETAILED FLOW")
        print(f"{'='*80}\n")
        
        state = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Episode start
        print(f"[Step {step}] Episode Start")
        print(f"  Initial State: Budget=${state.remaining_budget:.2f}, CPC=${state.current_cpc:.2f}")
        
        while not done:
            state_array = state.to_array()
            
            # STEP 1: SimulationAgent → ControllerAgent
            print(f"\n[Step {step}] SimulationAgent → ControllerAgent")
            print(f"  State: {state_array[:4]}...")  # Show first 4 features
            
            self.step_logger.log_step(step, 'state_sent', {
                'state': state_array.tolist(),
                'remaining_budget': state.remaining_budget
            })
            
            # STEP 2: ControllerAgent decides and queries specialists
            controller_action, _, entropy = self.controller.select_action(
                state_array, training=training
            )
            
            print(f"[Step {step}] ControllerAgent Decision")
            print(f"  Action: {['Bidding Only', 'Budget Only', 'Both'][controller_action]}")
            print(f"  Entropy: {entropy:.3f}")
            
            self.step_logger.log_step(step, 'controller_decision', {
                'controller_action': controller_action,
                'entropy': entropy
            })
            
            # Track which agents to use
            use_bidding = controller_action in [0, 2]
            use_budget = controller_action in [1, 2]
            
            # STEP 3: BiddingAgent → ControllerAgent (if needed)
            if use_bidding:
                bid, bid_idx = self.bidding_agent.select_bid(state_array, training=training)
                print(f"[Step {step}] 💰 BiddingAgent → ControllerAgent")
                print(f"  Proposed Bid: ${bid:.2f}")
                
                self.step_logger.log_step(step, 'bidding_proposal', {
                    'bid': bid,
                    'bid_idx': bid_idx
                })
            else:
                bid = 2.0
                bid_idx = 5
                print(f"[Step {step}] 💰 BiddingAgent: Using default bid ${bid:.2f}")
            
            # STEP 4: BudgetAgent → ControllerAgent (if needed)
            if use_budget:
                allocation = self.budget_agent.select_allocation(state_array, training=training)
                print(f"[Step {step}] 📊 BudgetAgent → ControllerAgent")
                print(f"  Proposed Allocation: {', '.join([f'{k}={v:.1%}' for k, v in list(allocation.items())[:2]])}")
                
                self.step_logger.log_step(step, 'budget_proposal', {
                    'allocation': allocation
                })
            else:
                allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
                print(f"[Step {step}] 📊 BudgetAgent: Using default allocation")
            
            # STEP 5: ControllerAgent → SimulationAgent (combined action)
            action = self.Action(
                bid_amount=bid,
                budget_allocation=allocation,
                agent_type=["bidding", "budget", "both"][controller_action]
            )
            
            print(f"[Step {step}] 🎯 ControllerAgent → SimulationAgent")
            print(f"  Combined Action: Bid=${bid:.2f}, Type={action.agent_type}")
            
            self.step_logger.log_step(step, 'action_executed', {
                'bid': bid,
                'allocation': allocation,
                'agent_type': action.agent_type
            })
            
            # STEP 6: SimulationAgent → All RL Agents (results)
            next_state, reward, done, info = self.env.step(action)
            next_state_array = next_state.to_array()
            
            print(f"[Step {step}] 📥 SimulationAgent → All Agents")
            print(f"  Reward: {reward:+.2f} | Conv: {info['conversions']} | "
                  f"Clicks: {info['clicks']} | Cost: ${info['cost']:.2f}")
            
            self.step_logger.log_step(step, 'reward_received', {
                'reward': reward,
                'conversions': info['conversions'],
                'clicks': info['clicks'],
                'cost': info['cost'],
                'done': done
            })
            
            # Store experiences for agents that acted
            self.controller.store_reward(reward, done)
            
            if use_bidding:
                self.bidding_agent.store_experience(
                    state_array, bid_idx, reward, next_state_array, done
                )
            
            if use_budget:
                self.budget_agent.store_reward(reward, done)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if step >= 5 and training:  # Show first 5 steps in detail
                print(f"\n[Steps {step}-{self.env.max_steps}] ⚡ Continuing... (fast mode)")
                break  
        
        # Complete episode silently
        while not done:
            state_array = state.to_array()
            controller_action, _, _ = self.controller.select_action(state_array, training=training)
            
            use_bidding = controller_action in [0, 2]
            use_budget = controller_action in [1, 2]
            
            if use_bidding:
                bid, bid_idx = self.bidding_agent.select_bid(state_array, training=training)
            else:
                bid, bid_idx = 2.0, 5
            
            if use_budget:
                allocation = self.budget_agent.select_allocation(state_array, training=training)
            else:
                allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
            
            action = self.Action(bid_amount=bid, budget_allocation=allocation, 
                               agent_type=["bidding", "budget", "both"][controller_action])
            
            next_state, reward, done, info = self.env.step(action)
            next_state_array = next_state.to_array()
            
            self.controller.store_reward(reward, done)
            if use_bidding:
                self.bidding_agent.store_experience(state_array, bid_idx, reward, 
                                                   next_state_array, done)
            if use_budget:
                self.budget_agent.store_reward(reward, done)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        # Episode complete
        print(f"\n[Step {step}] 🏁 Episode Complete")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Total Steps: {step}")
        
        # Calculate metrics
        roi = 0
        if self.env.total_spend > 0:
            roi = ((self.env.total_conversions * self.env.conversion_value - 
                   self.env.total_spend) / self.env.total_spend) * 100
        
        metrics = {
            'reward': episode_reward,
            'conversions': self.env.total_conversions,
            'clicks': self.env.total_clicks,
            'spend': self.env.total_spend,
            'roi': roi,
            'steps': step
        }
        
        self.episode_metrics.append(metrics)
        
        # Finish logging
        episode_data = self.step_logger.finish_episode(episode, metrics)
        
        # STEP 7: AnalyticsAgent → Analyzes episode flow
        print(f"\n{'='*80}")
        print(f"STEP 7: CrewAI ANALYTICS AGENT ANALYSIS")
        print(f"{'='*80}\n")
        
        analysis = self.run_crewai_analysis(episode_data)
        
        return metrics, analysis
    
    def run_crewai_analysis(self, episode_data: Dict):
        """Run CrewAI analytics on episode data"""
        
        # Generate analysis
        analysis_text = analyze_episode_flow(episode_data)
        
        # Create task for CrewAI agent
        task = Task(
            description=f"""
Review the following episode flow analysis and provide strategic insights:

{analysis_text}

Based on this detailed episode flow, provide:
1. Overall assessment of multi-agent coordination
2. Key strengths in the current strategy
3. Critical areas needing improvement
4. Specific next steps for optimization

Focus on actionable insights that can improve the system's performance.
            """,
            expected_output="Strategic insights with specific recommendations",
            agent=self.analytics_agent
        )
        
        # Create and run crew
        crew = Crew(
            agents=[self.analytics_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False  # Set to True to see CrewAI thinking
        )
        
        try:
            print("🤖 CrewAI Analytics Agent analyzing episode flow...\n")
            result = crew.kickoff()
            
            print("="*80)
            print("CREWAI STRATEGIC INSIGHTS:")
            print("="*80)
            print(result)
            print("="*80)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"CrewAI analysis failed: {e}")
            print(f"⚠️  CrewAI analysis unavailable: {e}")
            print("\nShowing automated analysis instead:")
            print(analysis_text)
            return analysis_text
    
    def train_with_crewai(self, num_episodes: int = 3):
        """Train system with CrewAI providing insights"""
        
        print("\n" + "="*80)
        print("INTEGRATED MULTI-AGENT TRAINING WITH CREWAI ORCHESTRATION")
        print("="*80)
        print("\nThis demonstrates the complete flow:")
        print("  1. SimulationAgent → ControllerAgent (state)")
        print("  2. ControllerAgent → BiddingAgent & BudgetAgent (queries)")
        print("  3. BiddingAgent → ControllerAgent (bid proposal)")
        print("  4. BudgetAgent → ControllerAgent (allocation proposal)")
        print("  5. ControllerAgent → SimulationAgent (combined action)")
        print("  6. SimulationAgent → All RL Agents (reward, next state)")
        print("  7. AnalyticsAgent (CrewAI) → Insights & recommendations")
        print("="*80)
        
        for episode in range(num_episodes):
            metrics, analysis = self.run_episode_with_logging(
                episode, training=True
            )
            
            # Update agents
            if episode % 1 == 0:  # Update every episode for demo
                print(f"\n🔄 Updating RL agents...")
                self.controller.update(epochs=3)
                self.budget_agent.update(epochs=3)
                
                for _ in range(2):
                    if len(self.bidding_agent.replay_buffer) >= 32:
                        self.bidding_agent.update()
                
                print("✓ Agents updated")
            
            if episode < num_episodes - 1:
                input(f"\n  Press Enter to continue to Episode {episode + 1}...")
        

# ============================================================================
# Demo Runner
# ============================================================================

def main():
    """Run integrated demo"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║         COMPLETE MULTI-AGENT ORCHESTRATION WITH CREWAI                ║
║                                                                        ║
║  This demonstrates the full cycle per timestep:                       ║
║    1. SimulationAgent → ControllerAgent                               ║
║    2. ControllerAgent → BiddingAgent & BudgetAgent                    ║
║    3. BiddingAgent → ControllerAgent                                  ║
║    4. BudgetAgent → ControllerAgent                                   ║
║    5. ControllerAgent → SimulationAgent                               ║
║    6. SimulationAgent → All RL Agents                                 ║
║    7. AnalyticsAgent (CrewAI) → Insights                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    num_episodes = input("\nHow many episodes to demonstrate? (1-5, default=2): ").strip()
    num_episodes = int(num_episodes) if num_episodes.isdigit() else 2
    num_episodes = min(max(1, num_episodes), 50)
    
    print(f"\n🚀 Running {num_episodes} episode(s) with complete flow...\n")
    
    # Initialize system
    system = IntegratedMultiAgentSystem()
    
    # Run training with CrewAI
    system.train_with_crewai(num_episodes=num_episodes)
 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()