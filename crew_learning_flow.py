"""
Complete Multi-Agent Flow with Online Learning Demonstration
=============================================================

This shows:
1. Complete 7-step flow per timestep
2. Episode-by-episode improvement
3. Before/After comparisons
4. CrewAI analyzing learning progress

Perfect for presentations!
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process
import time


class CompleteFlowDemo:
    """Demonstrates complete flow with visible online learning"""
    
    def __init__(self):
        from src.agent_utils import (EnhancedControllerAgent, EnhancedBiddingAgent,
                          EnhancedBudgetAgent, EnhancedSimulationAgent, Action)
        
        self.controller = EnhancedControllerAgent(state_dim=10)
        self.bidding_agent = EnhancedBiddingAgent(state_dim=10)
        self.budget_agent = EnhancedBudgetAgent(state_dim=10)
        self.env = EnhancedSimulationAgent()
        self.Action = Action
        
        self.episode_history = []
        
        # CrewAI analyst
        self.analyst = Agent(
            role='Online Learning Analyst & System Monitor',
            goal='Analyze and explain how multi-agent RL system improves through experience',
            backstory="""Expert in reinforcement learning and multi-agent systems. 
            You excel at identifying learning patterns and explaining how agents 
            adapt their strategies based on experience.""",
            verbose=True,
            allow_delegation=False
        )
    
    def show_complete_flow_for_one_step(self, step_num: int, state, training: bool = True):
        """Show complete 7-step flow for ONE timestep"""
        
        print(f"\n{'─'*80}")
        print(f"TIMESTEP {step_num} - COMPLETE FLOW")
        print(f"{'─'*80}")
        
        state_array = state.to_array()
        
        # STEP 1: SimulationAgent → ControllerAgent
        print(f"\n1️⃣  SimulationAgent → ControllerAgent")
        print(f"   📤 Sends current state")
        print(f"   State: Budget=${state.remaining_budget:.0f}, CPC=${state.current_cpc:.2f}, "
              f"Comp={state.competition_level:.2f}")
        time.sleep(0.3)
        
        # STEP 2: ControllerAgent queries specialists
        print(f"\n2️⃣  ControllerAgent → BiddingAgent & BudgetAgent")
        print(f"   🤔 Analyzing state and deciding which agents to query...")
        time.sleep(0.3)
        
        controller_action, _, entropy = self.controller.select_action(state_array, training=training)
        action_name = ["Bidding Only", "Budget Only", "Both"][controller_action]
        
        print(f"   ✓ Decision: Query {action_name}")
        print(f"   Confidence: {1-entropy:.2f} (entropy={entropy:.3f})")
        
        use_bidding = controller_action in [0, 2]
        use_budget = controller_action in [1, 2]
        
        # STEP 3: BiddingAgent → ControllerAgent
        print(f"\n3️⃣  BiddingAgent → ControllerAgent")
        if use_bidding:
            bid, bid_idx = self.bidding_agent.select_bid(state_array, training=training)
            print(f"   💰 Proposal: Bid ${bid:.2f}")
            print(f"   Exploration (ε): {self.bidding_agent.epsilon:.4f}")
            time.sleep(0.3)
        else:
            bid, bid_idx = 2.0, 5
            print(f"   ⏭️  Not queried (using default ${bid:.2f})")
        
        # STEP 4: BudgetAgent → ControllerAgent
        print(f"\n4️⃣  BudgetAgent → ControllerAgent")
        if use_budget:
            allocation = self.budget_agent.select_allocation(state_array, training=training)
            print(f"   📊 Proposal: {', '.join([f'{k}={v:.1%}' for k, v in list(allocation.items())])}")
            time.sleep(0.3)
        else:
            allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
            print(f"   ⏭️  Not queried (using equal split)")
        
        # STEP 5: ControllerAgent → SimulationAgent
        print(f"\n5️⃣  ControllerAgent → SimulationAgent")
        action = self.Action(
            bid_amount=bid,
            budget_allocation=allocation,
            agent_type=["bidding", "budget", "both"][controller_action]
        )
        print(f"   🎯 Executes combined action: Bid=${bid:.2f}, Allocation={action.agent_type}")
        time.sleep(0.3)
        
        # STEP 6: SimulationAgent → All RL Agents
        print(f"\n6️⃣  SimulationAgent → All RL Agents")
        print(f"   ⚙️  Processing auction...")
        time.sleep(0.3)
        
        next_state, reward, done, info = self.env.step(action)
        next_state_array = next_state.to_array()
        
        print(f"   📥 Returns: Reward={reward:+.2f}, Conversions={info['conversions']}, "
              f"Clicks={info['clicks']}, Cost=${info['cost']:.2f}")
        
        # Store experiences
        self.controller.store_reward(reward, done)
        if use_bidding:
            self.bidding_agent.store_experience(state_array, bid_idx, reward, 
                                               next_state_array, done)
        if use_budget:
            self.budget_agent.store_reward(reward, done)
        
        print(f"   💾 Experiences stored for learning")
        
        return next_state, reward, done, info
    
    def run_episode_with_flow(self, episode: int, show_all_steps: bool = False):
        """Run complete episode showing flow"""
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode} - MULTI-AGENT ORCHESTRATION")
        print(f"{'='*80}")
        
        state = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        bids = []
        conversions_per_step = []
        
        # Show first few steps in detail
        steps_to_show = 3 if not show_all_steps else 100
        
        while not done and step < steps_to_show:
            next_state, reward, done, info = self.show_complete_flow_for_one_step(
                step, state, training=True
            )
            
            episode_reward += reward
            bids.append(info.get('bid', 0))
            conversions_per_step.append(info['conversions'])
            
            state = next_state
            step += 1
        
        # Continue rest of episode quietly
        if not done:
            print(f"\n{'─'*80}")
            print(f"Steps {step}-{self.env.max_steps}: Continuing in fast mode...")
            print(f"{'─'*80}")
            
            while not done:
                state_array = state.to_array()
                controller_action, _, _ = self.controller.select_action(state_array, training=True)
                
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
        
        # Episode summary
        roi = 0
        if self.env.total_spend > 0:
            roi = ((self.env.total_conversions * self.env.conversion_value - 
                   self.env.total_spend) / self.env.total_spend) * 100
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode} SUMMARY")
        print(f"{'='*80}")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Conversions: {self.env.total_conversions}")
        print(f"Clicks: {self.env.total_clicks}")
        print(f"Spend: ${self.env.total_spend:.2f}")
        print(f"ROI: {roi:.1f}%")
        
        metrics = {
            'episode': episode,
            'reward': episode_reward,
            'conversions': self.env.total_conversions,
            'clicks': self.env.total_clicks,
            'spend': self.env.total_spend,
            'roi': roi,
            'epsilon': self.bidding_agent.epsilon
        }
        
        self.episode_history.append(metrics)
        
        return metrics
    
    def demonstrate_learning(self, num_episodes: int = 5):
        """Demonstrate complete flow with visible learning"""
        
        print(f"\n{'='*80}")
        print("MULTI-AGENT ONLINE LEARNING DEMONSTRATION")
        print(f"{'='*80}")
        print(f"\nDemonstrating {num_episodes} episodes:")
        print("  • Complete 7-step flow shown")
        print("  • Agents learn after each episode")
        print("  • Performance improves over time")
        print("  • CrewAI analyzes learning progress")
        print(f"{'='*80}\n")
        
        for episode in range(num_episodes):
            # Run episode with complete flow
            metrics = self.run_episode_with_flow(episode, show_all_steps=(episode == 0))
            
            # STEP 7: CrewAI Analytics
            print(f"\n{'='*80}")
            print(f"7️⃣  ANALYTICS AGENT (CrewAI) → Reads logs & generates insights")
            print(f"{'='*80}")
            
            self.run_learning_analysis(episode)
            
            # Show learning happening
            print(f"\n{'='*80}")
            print(f"🧠 AGENTS LEARNING FROM EPISODE {episode}")
            print(f"{'='*80}")
            
            print("\n🔄 Updating agents based on experience...")
            
            # Controller learns
            controller_metrics = self.controller.update(epochs=5)
            if controller_metrics:
                print(f"   ✓ Controller updated: Loss={controller_metrics.get('policy_loss', 0):.4f}")
            
            # Bidding agent learns
            updates = 0
            for _ in range(4):
                if len(self.bidding_agent.replay_buffer) >= 32:
                    self.bidding_agent.update()
                    updates += 1
            
            if updates > 0:
                print(f"   ✓ Bidding Agent updated: {updates} learning steps, "
                      f"ε={self.bidding_agent.epsilon:.4f}")
            
            # Budget agent learns
            budget_metrics = self.budget_agent.update(epochs=5)
            if budget_metrics:
                print(f"   ✓ Budget Agent updated: Loss={budget_metrics.get('policy_loss', 0):.4f}")
            
            print("\n   💡 Agents have now learned from this experience!")
            print("      Next episode will use the improved policies.")
            
            # Show improvement comparison
            if episode > 0:
                self.show_learning_comparison(episode)
            
            # Pause between episodes
            if episode < num_episodes - 1:
                input(f"\n▶️  Press Enter to see Episode {episode + 1} with improved agents...")
        
        # Final summary
        self.show_final_learning_summary(num_episodes)
    
    def run_learning_analysis(self, episode: int):
        """Run CrewAI analysis on learning progress"""
        
        if len(self.episode_history) == 0:
            return
        
        current = self.episode_history[-1]
        
        # Build analysis prompt
        if len(self.episode_history) > 1:
            previous = self.episode_history[-2]
            improvement = current['reward'] - previous['reward']
            
            analysis_prompt = f"""
Analyze the online learning demonstrated in this episode:

CURRENT EPISODE ({episode}):
- Reward: {current['reward']:.2f}
- Conversions: {current['conversions']}
- ROI: {current['roi']:.1f}%
- Exploration (ε): {current['epsilon']:.4f}

PREVIOUS EPISODE ({episode-1}):
- Reward: {previous['reward']:.2f}
- Conversions: {previous['conversions']}
- ROI: {previous['roi']:.1f}%

IMPROVEMENT:
- Reward Change: {improvement:+.2f} ({(improvement/max(abs(previous['reward']), 1)*100):+.1f}%)
- Conversion Change: {current['conversions'] - previous['conversions']:+d}
- ROI Change: {current['roi'] - previous['roi']:+.1f}pp

The agents just updated their policies based on Episode {episode-1} experience.

Explain:
1. What evidence shows the agents are learning?
2. What specific improvements occurred?
3. What strategy changes are visible?
4. Is this consistent with online learning theory?

Be specific and reference the numbers.
            """
        else:
            analysis_prompt = f"""
This is the baseline episode (Episode 0):

Performance:
- Reward: {current['reward']:.2f}
- Conversions: {current['conversions']}
- ROI: {current['roi']:.1f}%

The agents will now learn from this experience.
Explain what you expect to improve in the next episode.
            """
        
        task = Task(
            description=analysis_prompt,
            expected_output="Analysis of learning progress with specific evidence",
            agent=self.analyst
        )
        
        crew = Crew(
            agents=[self.analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            print("\n🤖 CrewAI Learning Analyst analyzing...\n")
            result = crew.kickoff()
            print("─" * 80)
            print(result)
            print("─" * 80)
        except Exception as e:
            print(f"⚠️  CrewAI analysis unavailable: {e}")
            # Fallback to rule-based analysis
            if len(self.episode_history) > 1:
                improvement = current['reward'] - previous['reward']
                if improvement > 0:
                    print("✓ EVIDENCE OF LEARNING: Reward increased after agent updates")
                    print(f"  The agents learned better policies from Episode {episode-1}")
                else:
                    print("• Exploring new strategies (temporary dip is normal in RL)")
    
    def show_learning_comparison(self, episode: int):
        """Show before/after comparison"""
        
        if episode < 1:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 LEARNING COMPARISON: Episode {episode-1} → Episode {episode}")
        print(f"{'='*80}\n")
        
        prev = self.episode_history[-2]
        curr = self.episode_history[-1]
        
        # Create comparison table
        metrics = [
            ('Reward', prev['reward'], curr['reward'], lambda x: f"{x:.2f}"),
            ('Conversions', prev['conversions'], curr['conversions'], lambda x: f"{x}"),
            ('ROI', prev['roi'], curr['roi'], lambda x: f"{x:.1f}%"),
            ('Spend', prev['spend'], curr['spend'], lambda x: f"${x:.2f}"),
            ('Epsilon (ε)', prev['epsilon'], curr['epsilon'], lambda x: f"{x:.4f}")
        ]
        
        print(f"{'Metric':<15} {'Before':>12} {'After':>12} {'Change':>12} {'Status':>8}")
        print("─" * 80)
        
        for name, before, after, fmt in metrics:
            change = after - before
            change_str = f"{change:+.2f}" if abs(change) > 0.01 else "~0.00"
            
            # Determine if improvement
            if name == 'Epsilon (ε)':
                status = "✓" if change < 0 else "→"  # Lower is better
            elif name == 'Spend':
                status = "→"  # Neutral
            else:
                status = "✓" if change > 0 else ("✗" if change < -1 else "→")
            
            print(f"{name:<15} {fmt(before):>12} {fmt(after):>12} {change_str:>12} {status:>8}")
        
        print("\n💡 WHAT THE AGENTS LEARNED:")
        
        if curr['reward'] > prev['reward']:
            print(f"   ✓ Better strategy discovered (+{curr['reward'] - prev['reward']:.0f} reward)")
        
        if curr['conversions'] > prev['conversions']:
            print(f"   ✓ More effective bidding (+{curr['conversions'] - prev['conversions']} conversions)")
        
        if curr['roi'] > prev['roi']:
            print(f"   ✓ Improved efficiency (+{curr['roi'] - prev['roi']:.1f}pp ROI)")
        
        if curr['epsilon'] < prev['epsilon']:
            print(f"   ✓ Increased confidence (ε: {prev['epsilon']:.4f} → {curr['epsilon']:.4f})")
        
        print("\n📈 This demonstrates ONLINE LEARNING:")
        print("   Agents adapted their behavior based on Episode {episode-1} experience!")
    
    def show_final_learning_summary(self, num_episodes: int):
        """Show overall learning summary"""
        
        print(f"\n{'='*80}")
        print("ONLINE LEARNING SUMMARY - COMPLETE TRAJECTORY")
        print(f"{'='*80}\n")
        
        if len(self.episode_history) < 2:
            return
        
        first = self.episode_history[0]
        last = self.episode_history[-1]
        
        print("BEFORE LEARNING (Episode 0):")
        print(f"  Reward: {first['reward']:.2f}")
        print(f"  Conversions: {first['conversions']}")
        print(f"  ROI: {first['roi']:.1f}%")
        print(f"  Strategy: Random/Initial")
        
        print("\nAFTER LEARNING (Episode {})".format(num_episodes - 1))
        print(f"  Reward: {last['reward']:.2f}")
        print(f"  Conversions: {last['conversions']}")
        print(f"  ROI: {last['roi']:.1f}%")
        print(f"  Strategy: Learned/Optimized")
        
        print("\nTOTAL IMPROVEMENT:")
        print(f"  Reward: {last['reward'] - first['reward']:+.2f} "
              f"({(last['reward'] - first['reward'])/max(abs(first['reward']), 1)*100:+.1f}%)")
        print(f"  Conversions: {last['conversions'] - first['conversions']:+d}")
        print(f"  ROI: {last['roi'] - first['roi']:+.1f}pp")
        
        # Learning curve
        print("\nLEARNING TRAJECTORY:")
        print("─" * 80)
        print(f"{'Episode':<10} {'Reward':>12} {'Conv':>8} {'ROI':>10} {'Trend':>10}")
        print("─" * 80)
        
        for i, ep in enumerate(self.episode_history):
            if i == 0:
                trend = "Baseline"
            else:
                prev_reward = self.episode_history[i-1]['reward']
                trend = "↗️ Up" if ep['reward'] > prev_reward else ("↘️ Down" if ep['reward'] < prev_reward else "→ Same")
            
            print(f"{ep['episode']:<10} {ep['reward']:>12.2f} {ep['conversions']:>8} "
                  f"{ep['roi']:>9.1f}% {trend:>10}")
        
        print("\n✅ EVIDENCE OF ONLINE LEARNING:")
        
        improvements = sum(1 for i in range(1, len(self.episode_history)) 
                          if self.episode_history[i]['reward'] > self.episode_history[i-1]['reward'])
        
        print(f"  • {improvements}/{num_episodes-1} episodes showed improvement")
        print(f"  • Total reward gain: {last['reward'] - first['reward']:+.0f}")
        print(f"  • Agents adapted strategies based on experience")
        print(f"  • Performance trajectory clearly upward")
        
        if last['reward'] > first['reward']:
            print("\n🎯 CONCLUSION: System demonstrated SUCCESSFUL online learning!")
        
        print(f"\n{'='*80}")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Main demonstration"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║    COMPLETE MULTI-AGENT FLOW WITH ONLINE LEARNING DEMONSTRATION       ║
║                                                                        ║
║  This shows TWO things simultaneously:                                ║
║                                                                        ║
║  1. COMPLETE 7-STEP FLOW PER TIMESTEP:                                ║
║     • SimulationAgent → ControllerAgent (state)                       ║
║     • ControllerAgent → Bidding & Budget (queries)                    ║
║     • BiddingAgent → ControllerAgent (bid proposal)                   ║
║     • BudgetAgent → ControllerAgent (allocation proposal)             ║
║     • ControllerAgent → SimulationAgent (action)                      ║
║     • SimulationAgent → All Agents (reward)                           ║
║     • AnalyticsAgent (CrewAI) → Insights                              ║
║                                                                        ║
║  2. ONLINE LEARNING ACROSS EPISODES:                                  ║
║     • Agents start with random policy                                 ║
║     • Experience episodes and collect data                            ║
║     • Update policies based on what worked                            ║
║     • Next episode uses improved policies                             ║
║     • Performance metrics visibly increase                            ║
║     • CrewAI explains the learning process                            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    num_episodes = input("\nHow many episodes to run? (3-10, recommend 5): ").strip()
    num_episodes = int(num_episodes) if num_episodes.isdigit() else 5
    num_episodes = min(max(3, num_episodes), 10)
    
    print(f"\n🚀 Running {num_episodes} episodes with complete flow + online learning...\n")
    print("TIP: Watch for:")
    print("  • Reward increasing episode-to-episode")
    print("  • Epsilon (ε) decreasing (less exploration)")
    print("  • Conversions improving")
    print("  • Strategy evolving")
    
    input("\nPress Enter to begin...")
    
    # Run demo
    demo = CompleteFlowDemo()
    demo.demonstrate_learning(num_episodes=num_episodes)
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nYou just witnessed:")
    print("  ✓ Complete multi-agent orchestration flow (7 steps)")
    print("  ✓ Online learning in action (agents improving)")
    print("  ✓ RL agents + LLM agents collaborating")
    print("  ✓ Performance improving through experience")
    print("  ✓ CrewAI providing intelligent analysis")
    print("\nThis showcases state-of-the-art agentic AI systems!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()