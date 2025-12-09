"""
Multi-Agent Agentic System with RL Capabilities using CrewAI
=============================================================

This system implements a sophisticated multi-agent architecture for ad auction optimization
with reinforcement learning and LLM-based analytics.

Architecture:
- Controller Agent (PPO): Master orchestrator
- Bidding Agent (DQN/DDQN): Auction bidding strategies
- Budget Allocation Agent (PPO/DDPG): Resource distribution
- Analytics Agent (CrewAI): Interpretability and insights
- Simulation Agent: Environment dynamics

Author: Akshay
Date: December 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import json
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class State:
    """System state representation"""
    remaining_budget: float
    current_cpc: float
    competition_level: float
    time_remaining: float
    conversions: int
    clicks: int
    impressions: int
    historical_performance: np.ndarray
    market_volatility: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL agents"""
        return np.array([
            self.remaining_budget,
            self.current_cpc,
            self.competition_level,
            self.time_remaining,
            self.conversions,
            self.clicks,
            self.impressions,
            self.market_volatility
        ], dtype=np.float32)


@dataclass
class Action:
    """Action representation"""
    bid_amount: float
    budget_allocation: Dict[str, float]
    agent_type: str  # 'bidding' or 'budget'


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class EpisodeLog:
    """Episode logging structure"""
    episode_id: int
    total_reward: float
    steps: int
    final_conversions: int
    final_clicks: int
    total_spend: float
    avg_cpc: float
    actions_taken: List[Dict]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Neural Networks
# ============================================================================

class PolicyNetwork(nn.Module):
    """Policy network for PPO agents"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)


class ValueNetwork(nn.Module):
    """Value network for PPO agents"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class QNetwork(nn.Module):
    """Q-Network for DQN/DDQN agents"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# RL Agents
# ============================================================================

class ControllerAgent:
    """Master orchestrator using PPO"""
    def __init__(self, state_dim: int, action_dim: int = 3, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim  # Choose: bidding_only, budget_only, both
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        
        # Storage for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        logger.info("Controller Agent initialized with PPO")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select which specialized agent to use"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
        
        dist = Categorical(action_probs)
        
        if training:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs, dim=1)
        
        log_prob = dist.log_prob(action)
        
        # Store for training
        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        return action.item(), log_prob.item()
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, epochs: int = 10):
        """Update policy and value networks using PPO"""
        if len(self.states) == 0:
            return
        
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            # Policy loss
            action_probs = self.policy_net(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Value loss
            values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        
        # Clear storage
        self.clear_memory()
        
        logger.info(f"Controller updated - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []


class BiddingAgent:
    """Bidding strategy agent using DDQN"""
    def __init__(self, state_dim: int, num_bid_levels: int = 10, lr: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = num_bid_levels
        
        self.q_network = QNetwork(state_dim, num_bid_levels)
        self.target_network = QNetwork(state_dim, num_bid_levels)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=50000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_freq = 100
        self.steps = 0
        
        # Bid levels (e.g., $0.50 to $5.00)
        self.bid_levels = np.linspace(0.5, 5.0, num_bid_levels)
        
        logger.info(f"Bidding Agent initialized with DDQN ({num_bid_levels} bid levels)")
    
    def select_bid(self, state: np.ndarray, training: bool = True) -> Tuple[float, int]:
        """Select bid amount using epsilon-greedy"""
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        bid_amount = self.bid_levels[action_idx]
        return bid_amount, action_idx
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def update(self, training: bool = False):
        """Update Q-network using DDQN"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor([e.done for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network for action selection, target network for evaluation
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info("Target network updated")
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()


class BudgetAllocationAgent:
    """Budget allocation agent using PPO for continuous control"""
    def __init__(self, state_dim: int, num_channels: int = 3, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = num_channels
        
        # Mean and log_std networks for continuous actions
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_channels),
            nn.Softmax(dim=-1)  # Ensure allocation sums to 1
        )
        
        self.value_net = ValueNetwork(state_dim)
        
        self.mean_optimizer = optim.Adam(self.mean_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
        
        logger.info(f"Budget Allocation Agent initialized with PPO ({num_channels} channels)")
    
    def select_allocation(self, state: np.ndarray, training: bool = True) -> Dict[str, float]:
        """Select budget allocation across channels"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            allocation = self.mean_net(state_tensor).squeeze(0)
            value = self.value_net(state_tensor)
        
        allocation_np = allocation.numpy()
        
        # Store for training
        if training:
            self.states.append(state)
            self.actions.append(allocation_np)
            self.values.append(value.item())
        
        return {name: float(alloc) for name, alloc in zip(self.channel_names, allocation_np)}
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, epochs: int = 10):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return
        
        # Compute returns
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        returns = torch.FloatTensor(returns)
        old_values = torch.FloatTensor(self.values)
        
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            # Policy loss (simplified for continuous)
            predicted_actions = self.mean_net(states)
            policy_loss = -torch.mean((predicted_actions * advantages.unsqueeze(1)).sum(dim=1))
            
            # Value loss
            values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Update
            self.mean_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mean_net.parameters(), 0.5)
            self.mean_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        
        self.clear_memory()
        logger.info(f"Budget Agent updated - Policy Loss: {policy_loss.item():.4f}")
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []


# ============================================================================
# Simulation Environment
# ============================================================================

class SimulationAgent:
    """Ad auction simulation environment"""
    def __init__(self, initial_budget: float = 10000.0, max_steps: int = 100):
        self.initial_budget = initial_budget
        self.max_steps = max_steps
        self.reset()
        
        logger.info("Simulation Environment initialized")
    
    def reset(self) -> State:
        """Reset environment to initial state"""
        self.current_step = 0
        self.remaining_budget = self.initial_budget
        self.total_conversions = 0
        self.total_clicks = 0
        self.total_impressions = 0
        self.total_spend = 0.0
        
        # Market dynamics
        self.base_cpc = np.random.uniform(1.0, 3.0)
        self.competition = np.random.uniform(0.5, 1.0)
        self.market_volatility = np.random.uniform(0.1, 0.3)
        
        state = State(
            remaining_budget=self.remaining_budget,
            current_cpc=self.base_cpc,
            competition_level=self.competition,
            time_remaining=1.0,
            conversions=0,
            clicks=0,
            impressions=0,
            historical_performance=np.zeros(10),
            market_volatility=self.market_volatility
        )
        
        return state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        self.current_step += 1
        
        # Simulate auction outcome based on bid
        bid = action.bid_amount
        win_probability = self._compute_win_probability(bid)
        
        impressions = 0
        clicks = 0
        conversions = 0
        cost = 0.0
        
        # Simulate multiple auction opportunities
        num_auctions = np.random.poisson(10)
        
        for _ in range(num_auctions):
            if random.random() < win_probability and self.remaining_budget > 0:
                impressions += 1
                actual_cpc = self.base_cpc * (1 + np.random.normal(0, self.market_volatility))
                
                # Click-through rate depends on bid quality
                ctr = 0.05 * (bid / self.base_cpc)
                if random.random() < ctr:
                    clicks += 1
                    cost += actual_cpc
                    
                    # Conversion rate
                    cvr = 0.1 * (bid / self.base_cpc)
                    if random.random() < cvr:
                        conversions += 1
        
        # Update state
        self.remaining_budget -= cost
        self.total_conversions += conversions
        self.total_clicks += clicks
        self.total_impressions += impressions
        self.total_spend += cost
        
        # Market dynamics evolve
        self.base_cpc *= (1 + np.random.normal(0, 0.05))
        self.competition = np.clip(self.competition + np.random.normal(0, 0.05), 0.3, 1.0)
        
        # Compute reward
        reward = self._compute_reward(conversions, clicks, cost)
        
        # Check if done
        done = (self.current_step >= self.max_steps or 
                self.remaining_budget <= 0)
        
        # Next state
        time_remaining = 1.0 - (self.current_step / self.max_steps)
        next_state = State(
            remaining_budget=max(0, self.remaining_budget),
            current_cpc=self.base_cpc,
            competition_level=self.competition,
            time_remaining=time_remaining,
            conversions=self.total_conversions,
            clicks=self.total_clicks,
            impressions=self.total_impressions,
            historical_performance=np.zeros(10),  # Simplified
            market_volatility=self.market_volatility
        )
        
        info = {
            'conversions': conversions,
            'clicks': clicks,
            'impressions': impressions,
            'cost': cost,
            'win_rate': impressions / max(1, num_auctions)
        }
        
        return next_state, reward, done, info
    
    def _compute_win_probability(self, bid: float) -> float:
        """Compute auction win probability based on bid"""
        # Higher bid relative to market CPC increases win rate
        bid_ratio = bid / self.base_cpc
        win_prob = 1 / (1 + np.exp(-(bid_ratio - 1) * 5))  # Sigmoid
        win_prob *= (1 - 0.5 * self.competition)  # Competition reduces win rate
        return np.clip(win_prob, 0.0, 1.0)
    
    def _compute_reward(self, conversions: int, clicks: int, cost: float) -> float:
        """Compute shaped reward signal"""
        # Reward = Revenue - Cost with efficiency bonus
        conversion_value = 50.0  # $50 per conversion
        revenue = conversions * conversion_value
        
        profit = revenue - cost
        
        # Efficiency bonus
        if clicks > 0:
            cvr = conversions / clicks
            efficiency_bonus = cvr * 10  # Bonus for high conversion rate
        else:
            efficiency_bonus = 0
        
        # Penalty for overspending
        if cost > 0:
            roi = revenue / cost
            roi_bonus = max(0, (roi - 1) * 5)  # Bonus for ROI > 1
        else:
            roi_bonus = 0
        
        total_reward = profit + efficiency_bonus + roi_bonus
        
        return total_reward


# ============================================================================
# CrewAI Analytics Agent
# ============================================================================

@tool("analyze_episode_performance")
def analyze_episode_performance(episode_data: str) -> str:
    """Analyze episode performance and provide insights"""
    data = json.loads(episode_data)
    
    total_reward = data['total_reward']
    conversions = data['final_conversions']
    clicks = data['final_clicks']
    spend = data['total_spend']
    avg_cpc = data['avg_cpc']
    
    analysis = f"""
Episode Performance Analysis:
=============================

Key Metrics:
- Total Reward: ${total_reward:.2f}
- Conversions: {conversions}
- Clicks: {clicks}
- Total Spend: ${spend:.2f}
- Average CPC: ${avg_cpc:.2f}

Performance Assessment:
"""
    
    # ROI calculation
    if spend > 0:
        roi = (conversions * 50 - spend) / spend * 100
        analysis += f"- ROI: {roi:.2f}%"
        
        if roi > 50:
            analysis += " [EXCELLENT]"
        elif roi > 0:
            analysis += " [PROFITABLE]"
        else:
            analysis += " [NEEDS IMPROVEMENT]"
    
    analysis += f"\n- Conversion Rate: {conversions/max(1, clicks)*100:.2f}%"
    
    # Recommendations
    analysis += "\n\nRecommendations:\n"
    if avg_cpc > 3.0:
        analysis += "- Consider reducing bid amounts to improve cost efficiency\n"
    if conversions / max(1, clicks) < 0.05:
        analysis += "- Low conversion rate - optimize landing pages or targeting\n"
    if total_reward < 0:
        analysis += "- Negative rewards indicate unprofitable strategy - revise approach\n"
    
    return analysis


@tool("detect_anomalies")
def detect_anomalies(metrics: str) -> str:
    """Detect anomalies in agent behavior"""
    data = json.loads(metrics)
    anomalies = []
    
    if data.get('reward_variance', 0) > 1000:
        anomalies.append("High reward variance detected - unstable policy")
    
    if data.get('avg_bid', 0) > 10:
        anomalies.append("Unusually high bids - possible agent collapse")
    
    if data.get('budget_utilization', 0) < 0.3:
        anomalies.append("Low budget utilization - agent too conservative")
    
    if len(anomalies) == 0:
        return "No anomalies detected. System operating normally."
    
    return "Anomalies Detected:\n" + "\n".join(f"- {a}" for a in anomalies)


def create_analytics_agent() -> Agent:
    """Create CrewAI analytics agent"""
    return Agent(
        role='RL Analytics Specialist',
        goal='Analyze RL agent performance and provide actionable insights',
        backstory="""You are an expert in reinforcement learning and marketing analytics.
        Your role is to interpret RL training logs, detect issues, and provide strategic
        recommendations to improve agent performance in ad auction optimization.""",
        tools=[analyze_episode_performance, detect_anomalies],
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# Multi-Agent System Orchestrator
# ============================================================================

class MultiAgentRLSystem:
    """Main orchestrator for multi-agent RL system"""
    def __init__(self, state_dim: int = 8):
        # Initialize RL agents
        self.controller = ControllerAgent(state_dim)
        self.bidding_agent = BiddingAgent(state_dim)
        self.budget_agent = BudgetAllocationAgent(state_dim)
        
        # Initialize environment
        self.env = SimulationAgent()
        
        # Initialize analytics
        self.analytics_agent = create_analytics_agent()
        
        # Episode logging
        self.episode_logs = []
        
        logger.info("Multi-Agent RL System initialized")
    
    def train(self, num_episodes: int = 1000, update_freq: int = 10):
        """Train the multi-agent system"""
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_actions = []
            done = False
            step = 0
            
            while not done:
                state_array = state.to_array()
                
                # Controller decides which agent(s) to use
                controller_action, _ = self.controller.select_action(state_array)
                
                # Execute based on controller decision
                if controller_action == 0:  # Bidding only
                    bid, bid_idx = self.bidding_agent.select_bid(state_array)
                    allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}  # Equal split
                elif controller_action == 1:  # Budget only
                    bid = 2.0  # Fixed bid
                    bid_idx = 5
                    allocation = self.budget_agent.select_allocation(state_array)
                else:  # Both
                    bid, bid_idx = self.bidding_agent.select_bid(state_array)
                    allocation = self.budget_agent.select_allocation(state_array)
                
                action = Action(bid_amount=bid, budget_allocation=allocation, 
                              agent_type=["bidding", "budget", "both"][controller_action])
                
                # Execute in environment
                next_state, reward, done, info = self.env.step(action)
                next_state_array = next_state.to_array()
                
                # Store experiences
                self.controller.store_reward(reward, done)
                self.bidding_agent.store_experience(state_array, bid_idx, reward, 
                                                   next_state_array, done)
                self.budget_agent.store_reward(reward, done)
                
                episode_reward += reward
                episode_actions.append({
                    'step': step,
                    'controller_action': controller_action,
                    'bid': bid,
                    'allocation': allocation,
                    'reward': reward,
                    **info
                })
                
                state = next_state
                step += 1
            
            # Log episode
            episode_log = EpisodeLog(
                episode_id=episode,
                total_reward=episode_reward,
                steps=step,
                final_conversions=self.env.total_conversions,
                final_clicks=self.env.total_clicks,
                total_spend=self.env.total_spend,
                avg_cpc=self.env.total_spend / max(1, self.env.total_clicks),
                actions_taken=episode_actions
            )
            self.episode_logs.append(episode_log)
            
            # Update agents
            if episode % update_freq == 0 and episode > 0:
                self.controller.update()
                self.budget_agent.update()
                
            # Update bidding agent more frequently
            if len(self.bidding_agent.replay_buffer) >= self.bidding_agent.batch_size:
                for _ in range(4):
                    self.bidding_agent.update(training=True)
            
            # Analytics every 50 episodes
            if episode % 50 == 0 and episode > 0:
                self._run_analytics(episode)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Conversions={self.env.total_conversions}, "
                          f"Spend=${self.env.total_spend:.2f}")
    
    def _run_analytics(self, episode: int):
        """Run CrewAI analytics on recent episodes"""
        recent_logs = self.episode_logs[-50:]
        
        # Prepare data for analytics
        avg_reward = np.mean([log.total_reward for log in recent_logs])
        total_conversions = sum(log.final_conversions for log in recent_logs)
        
        episode_data = json.dumps({
            'episode': episode,
            'total_reward': recent_logs[-1].total_reward,
            'final_conversions': recent_logs[-1].final_conversions,
            'final_clicks': recent_logs[-1].final_clicks,
            'total_spend': recent_logs[-1].total_spend,
            'avg_cpc': recent_logs[-1].avg_cpc
        })
        
        # Create analysis task
        analysis_task = Task(
            description=f"Analyze the performance of episode {episode} and provide insights",
            expected_output="Detailed performance analysis with recommendations",
            agent=self.analytics_agent,
            tools=[analyze_episode_performance]
        )
        
        # Run analytics
        crew = Crew(
            agents=[self.analytics_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            result = crew.kickoff(inputs={'episode_data': episode_data})
            logger.info(f"\n{'='*60}\nAnalytics Report (Episode {episode}):\n{'='*60}\n{result}\n")
        except Exception as e:
            logger.warning(f"Analytics failed: {e}")
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate trained agents"""
        total_rewards = []
        total_conversions = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_array = state.to_array()
                
                # Greedy action selection
                controller_action, _ = self.controller.select_action(state_array, training=False)
                
                if controller_action == 0:
                    bid, _ = self.bidding_agent.select_bid(state_array, training=False)
                    allocation = {f"Channel_{i+1}": 1.0/3 for i in range(3)}
                elif controller_action == 1:
                    bid = 2.0
                    allocation = self.budget_agent.select_allocation(state_array, training=False)
                else:
                    bid, _ = self.bidding_agent.select_bid(state_array, training=False)
                    allocation = self.budget_agent.select_allocation(state_array, training=False)
                
                action = Action(bid_amount=bid, budget_allocation=allocation, 
                              agent_type=["bidding", "budget", "both"][controller_action])
                
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
            total_conversions.append(self.env.total_conversions)
            
            logger.info(f"Eval Episode {episode}: Reward={episode_reward:.2f}, "
                       f"Conversions={self.env.total_conversions}")
        
        logger.info(f"\n{'='*60}\nEvaluation Results:\n{'='*60}")
        logger.info(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        logger.info(f"Average Conversions: {np.mean(total_conversions):.1f} ± {np.std(total_conversions):.1f}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Multi-Agent Agentic System with RL and CrewAI")
    print("="*80)
    
    # Initialize system
    system = MultiAgentRLSystem(state_dim=8)
    
    # Train
    print("\n[*] Starting training...")
    system.train(num_episodes=200, update_freq=10)
    
    # Evaluate
    print("\n[*] Evaluating trained agents...")
    system.evaluate(num_episodes=20)
    
    print("\n[*] Training complete!")