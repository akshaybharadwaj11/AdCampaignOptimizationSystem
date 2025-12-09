"""
Production-Ready Multi-Agent RL System with CrewAI
===================================================

A sophisticated multi-agent agentic system for ad auction optimization with:
- Advanced RL algorithms (PPO, DDQN, A3C)
- Comprehensive experimental framework
- Statistical validation and ablation studies
- Hyperparameter optimization
- Production monitoring and A/B testing
- Model versioning and checkpointing

Author: Akshay
Version: 2.0
Date: December 2025
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import random
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import json
import pandas as pd
from datetime import datetime
import logging
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for RL agents"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_size: int = 50000
    hidden_dim: int = 256
    update_frequency: int = 4
    target_update_frequency: int = 100


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    num_episodes: int = 1000
    eval_frequency: int = 50
    save_frequency: int = 100
    num_eval_episodes: int = 20
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    initial_budget: float = 10000.0
    max_steps: int = 100
    min_bid: float = 0.5
    max_bid: float = 5.0
    num_bid_levels: int = 10
    num_channels: int = 3
    conversion_value: float = 50.0


# ============================================================================
# Enhanced Data Structures
# ============================================================================

@dataclass
class State:
    """Enhanced state representation with versioning"""
    remaining_budget: float
    current_cpc: float
    competition_level: float
    time_remaining: float
    conversions: int
    clicks: int
    impressions: int
    historical_performance: np.ndarray
    market_volatility: float
    roi_history: List[float] = field(default_factory=list)
    version: int = 1
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array with proper normalization - always returns 10 features"""
        # Always compute ROI features (use 0 if history is empty)
        if self.roi_history and len(self.roi_history) > 0:
            roi_mean = np.mean(self.roi_history[-10:])
            roi_std = np.std(self.roi_history[-10:]) if len(self.roi_history) > 1 else 0.0
        else:
            roi_mean = 0.0
            roi_std = 0.0
        
        features = np.array([
            self.remaining_budget / 10000.0,  # Normalize
            self.current_cpc / 5.0,
            self.competition_level,
            self.time_remaining,
            self.conversions / 100.0,
            self.clicks / 1000.0,
            self.impressions / 10000.0,
            self.market_volatility,
            roi_mean,  # Always include ROI mean
            roi_std    # Always include ROI std
        ], dtype=np.float32)
        
        return features
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return asdict(self)


@dataclass
class Action:
    """Enhanced action with metadata"""
    bid_amount: float
    budget_allocation: Dict[str, float]
    agent_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experience:
    """Experience tuple with priority"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class EpisodeMetrics:
    """Comprehensive episode metrics"""
    episode_id: int
    total_reward: float
    steps: int
    conversions: int
    clicks: int
    impressions: int
    total_spend: float
    avg_cpc: float
    roi: float
    win_rate: float
    conversion_rate: float
    controller_entropy: float
    epsilon: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# Advanced Neural Networks
# ============================================================================

class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture for better value estimation"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using dueling architecture
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value


# ============================================================================
# Prioritized Experience Replay
# ============================================================================

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for better sample efficiency"""
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, experience: Experience):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritization"""
        if self.size < batch_size:
            batch_size = self.size
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return self.size


# ============================================================================
# Enhanced RL Agents with Error Handling
# ============================================================================

class EnhancedControllerAgent:
    """Production-ready Controller Agent with comprehensive features"""
    def __init__(self, state_dim: int = 10, action_dim: int = 3, 
                 config: AgentConfig = AgentConfig()):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim, config.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Training parameters
        self.gamma = config.gamma
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Metrics
        self.training_steps = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        # Fallback mechanism
        self.fallback_action = 2  # Use both agents by default
        
        logger.info(f"Enhanced Controller initialized - State dim: {state_dim}, Actions: {action_dim}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action with proper error handling"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, value = self.actor_critic(state_tensor)
            
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            if training:
                action = dist.sample()
            else:
                action = torch.argmax(action_probs, dim=1)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            if training:
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())
            
            return action.item(), log_prob.item(), entropy.item()
        
        except Exception as e:
            logger.error(f"Controller action selection failed: {e}")
            return self.fallback_action, 0.0, 0.0
    
    def store_reward(self, reward: float, done: bool):
        """Store transition with validation"""
        if not np.isfinite(reward):
            logger.warning(f"Invalid reward: {reward}, clipping to [-1000, 1000]")
            reward = np.clip(reward, -1000, 1000)
        
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
    
    def update(self, epochs: int = 10) -> Dict[str, float]:
        """Update with comprehensive logging"""
        if len(self.states) == 0:
            return {}
        
        # Ensure buffers are synchronized
        min_len = min(len(self.states), len(self.actions), len(self.rewards), 
                     len(self.log_probs), len(self.values), len(self.dones))
        
        if min_len == 0:
            self.clear_memory()
            return {}
        
        # Trim all buffers to minimum length
        self.states = self.states[:min_len]
        self.actions = self.actions[:min_len]
        self.rewards = self.rewards[:min_len]
        self.log_probs = self.log_probs[:min_len]
        self.values = self.values[:min_len]
        self.dones = self.dones[:min_len]
        
        try:
            advantages, returns = self.compute_gae()
            
            states = torch.FloatTensor(np.array(self.states))
            actions = torch.LongTensor(self.actions)
            old_log_probs = torch.FloatTensor(self.log_probs)
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            for epoch in range(epochs):
                action_logits, values = self.actor_critic(states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = values.squeeze()
                value_loss = F.mse_loss(values, returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
            
            self.training_steps += 1
            
            metrics = {
                'policy_loss': total_policy_loss / epochs,
                'value_loss': total_value_loss / epochs,
                'entropy': total_entropy / epochs,
                'training_steps': self.training_steps
            }
            
            self.policy_losses.append(metrics['policy_loss'])
            self.value_losses.append(metrics['value_loss'])
            self.entropies.append(metrics['entropy'])
            
            self.clear_memory()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Controller update failed: {e}")
            self.clear_memory()
            return {}
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']


class EnhancedBiddingAgent:
    """Production-ready Bidding Agent with Dueling DDQN"""
    def __init__(self, state_dim: int = 10, num_bid_levels: int = 10, 
                 config: AgentConfig = AgentConfig()):
        self.state_dim = state_dim
        self.action_dim = num_bid_levels
        self.config = config
        
        # Networks
        self.q_network = DuelingQNetwork(state_dim, num_bid_levels, config.hidden_dim)
        self.target_network = DuelingQNetwork(state_dim, num_bid_levels, config.hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            alpha=0.6,
            beta=0.4
        )
        
        # Training parameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.update_counter = 0
        self.target_update_freq = config.target_update_frequency
        
        # Bid levels
        self.bid_levels = np.linspace(0.5, 5.0, num_bid_levels)
        
        # Metrics
        self.q_losses = []
        self.epsilon_history = []
        
        logger.info(f"Enhanced Bidding Agent initialized - {num_bid_levels} bid levels")
    
    def select_bid(self, state: np.ndarray, training: bool = True) -> Tuple[float, int]:
        """Select bid with epsilon-greedy exploration"""
        try:
            if training and random.random() < self.epsilon:
                action_idx = random.randrange(self.action_dim)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
            
            bid_amount = self.bid_levels[action_idx]
            return bid_amount, action_idx
        
        except Exception as e:
            logger.error(f"Bidding selection failed: {e}")
            return 2.0, self.action_dim // 2  # Fallback to middle bid
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience with validation"""
        if not np.isfinite(reward):
            reward = np.clip(reward, -1000, 1000)
        
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def update(self) -> Optional[float]:
        """Update Q-network with prioritized replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        try:
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            
            states = torch.FloatTensor(np.array([e.state for e in experiences]))
            actions = torch.LongTensor([e.action for e in experiences])
            rewards = torch.FloatTensor([e.reward for e in experiences])
            next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
            dones = torch.FloatTensor([e.done for e in experiences])
            weights = torch.FloatTensor(weights)
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Double DQN
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            
            # Compute TD errors
            td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
            
            # Weighted loss
            loss = (weights.unsqueeze(1) * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities
            self.replay_buffer.update_priorities(indices, td_errors.flatten())
            
            # Update target network
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            self.q_losses.append(loss.item())
            self.epsilon_history.append(self.epsilon)
            
            return loss.item()
        
        except Exception as e:
            logger.error(f"Bidding update failed: {e}")
            return None
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']


class EnhancedBudgetAgent:
    """Production-ready Budget Allocation Agent"""
    def __init__(self, state_dim: int = 10, num_channels: int = 3,
                 config: AgentConfig = AgentConfig()):
        self.state_dim = state_dim
        self.action_dim = num_channels
        self.config = config
        
        # Network
        self.actor_critic = ActorCriticNetwork(state_dim, num_channels, config.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Training parameters
        self.gamma = config.gamma
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
        
        # Metrics
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"Enhanced Budget Agent initialized - {num_channels} channels")
    
    def select_allocation(self, state: np.ndarray, training: bool = True) -> Dict[str, float]:
        """Select budget allocation with fallback"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, value = self.actor_critic(state_tensor)
            
            allocation = F.softmax(action_logits, dim=-1).squeeze(0)
            allocation_np = allocation.numpy()
            
            # Ensure valid allocation
            if not np.isfinite(allocation_np).all():
                logger.warning("Invalid allocation detected, using equal split")
                allocation_np = np.ones(self.action_dim) / self.action_dim
            
            if training:
                self.states.append(state)
                self.actions.append(allocation_np)
                self.values.append(value.item())
            
            return {name: float(alloc) for name, alloc in zip(self.channel_names, allocation_np)}
        
        except Exception as e:
            logger.error(f"Budget allocation failed: {e}")
            equal_split = 1.0 / self.action_dim
            return {name: equal_split for name in self.channel_names}
    
    def store_reward(self, reward: float, done: bool):
        """Store reward with validation"""
        if not np.isfinite(reward):
            reward = np.clip(reward, -1000, 1000)
        
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, epochs: int = 10) -> Dict[str, float]:
        """Update policy"""
        if len(self.states) == 0:
            return {}
        
        try:
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
            
            total_policy_loss = 0
            total_value_loss = 0
            
            for _ in range(epochs):
                action_logits, values = self.actor_critic(states)
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Policy loss
                policy_loss = -torch.mean((action_probs * advantages.unsqueeze(1)).sum(dim=1))
                
                # Value loss
                values = values.squeeze()
                value_loss = F.mse_loss(values, returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
            
            metrics = {
                'policy_loss': total_policy_loss / epochs,
                'value_loss': total_value_loss / epochs
            }
            
            self.policy_losses.append(metrics['policy_loss'])
            self.value_losses.append(metrics['value_loss'])
            
            self.clear_memory()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Budget update failed: {e}")
            self.clear_memory()
            return {}
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# Simulation Environment with Advanced Features
# ============================================================================

class EnhancedSimulationAgent:
    """Production-ready simulation environment"""
    def __init__(self, config: EnvironmentConfig = EnvironmentConfig()):
        self.config = config
        self.initial_budget = config.initial_budget
        self.max_steps = config.max_steps
        self.conversion_value = config.conversion_value
        
        # Metrics tracking
        self.episode_stats = []
        
        self.reset()
        
        logger.info("Enhanced Simulation Environment initialized")
    
    def reset(self) -> State:
        """Reset environment to initial state"""
        self.current_step = 0
        self.remaining_budget = self.initial_budget
        self.total_conversions = 0
        self.total_clicks = 0
        self.total_impressions = 0
        self.total_spend = 0.0
        self.roi_history = []
        
        # Market dynamics
        self.base_cpc = np.random.uniform(1.0, 3.0)
        self.competition = np.random.uniform(0.5, 1.0)
        self.market_volatility = np.random.uniform(0.1, 0.3)
        
        # Seasonal effects
        self.day_of_week_effect = np.random.uniform(0.8, 1.2)
        
        state = State(
            remaining_budget=self.remaining_budget,
            current_cpc=self.base_cpc,
            competition_level=self.competition,
            time_remaining=1.0,
            conversions=0,
            clicks=0,
            impressions=0,
            historical_performance=np.zeros(10),
            market_volatility=self.market_volatility,
            roi_history=[]
        )
        
        return state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """Execute action with realistic market dynamics"""
        self.current_step += 1
        
        try:
            # Simulate auction
            bid = action.bid_amount
            win_probability = self._compute_win_probability(bid)
            
            impressions = 0
            clicks = 0
            conversions = 0
            cost = 0.0
            
            # Multiple auction opportunities
            num_auctions = int(np.random.poisson(10) * self.day_of_week_effect)
            
            for _ in range(num_auctions):
                if random.random() < win_probability and self.remaining_budget > 0:
                    impressions += 1
                    actual_cpc = self.base_cpc * (1 + np.random.normal(0, self.market_volatility))
                    
                    # CTR depends on bid quality
                    ctr = min(0.2, 0.05 * (bid / self.base_cpc))
                    if random.random() < ctr:
                        clicks += 1
                        cost += actual_cpc
                        
                        # CVR depends on bid quality
                        cvr = min(0.3, 0.1 * (bid / self.base_cpc))
                        if random.random() < cvr:
                            conversions += 1
            
            # Update state
            self.remaining_budget = max(0, self.remaining_budget - cost)
            self.total_conversions += conversions
            self.total_clicks += clicks
            self.total_impressions += impressions
            self.total_spend += cost
            
            # Market evolution
            self.base_cpc *= (1 + np.random.normal(0, 0.05))
            self.base_cpc = np.clip(self.base_cpc, 0.5, 10.0)
            self.competition = np.clip(self.competition + np.random.normal(0, 0.05), 0.3, 1.0)
            
            # Compute reward
            reward = self._compute_reward(conversions, clicks, cost)
            
            # ROI tracking
            if cost > 0:
                roi = (conversions * self.conversion_value - cost) / cost
                self.roi_history.append(roi)
            
            # Check termination
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
                historical_performance=np.zeros(10),
                market_volatility=self.market_volatility,
                roi_history=self.roi_history.copy()
            )
            
            info = {
                'conversions': conversions,
                'clicks': clicks,
                'impressions': impressions,
                'cost': cost,
                'win_rate': impressions / max(1, num_auctions),
                'ctr': clicks / max(1, impressions),
                'cvr': conversions / max(1, clicks)
            }
            
            return next_state, reward, done, info
        
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            # Return safe defaults
            return self.reset(), 0.0, True, {}
    
    def _compute_win_probability(self, bid: float) -> float:
        """Compute auction win probability"""
        bid_ratio = bid / self.base_cpc
        win_prob = 1 / (1 + np.exp(-(bid_ratio - 1) * 5))
        win_prob *= (1 - 0.5 * self.competition)
        return np.clip(win_prob, 0.0, 1.0)
    
    def _compute_reward(self, conversions: int, clicks: int, cost: float) -> float:
        """Compute shaped reward signal with proper scaling"""
        
        # Base reward: Simple profit with proper scaling
        revenue = conversions * self.conversion_value
        profit = revenue - cost
        
        # Scale down to per-step range (divide by expected steps)
        # This prevents accumulation of large negative values
        base_reward = profit / 10.0  # Scale factor
        
        # Conversion bonus (encourage getting conversions)
        conversion_bonus = conversions * 5.0  # $5 bonus per conversion
        
        # Efficiency bonus (high CVR is good)
        if clicks > 0:
            cvr = conversions / clicks
            efficiency_bonus = cvr * 20.0  # Reward efficient conversions
        else:
            efficiency_bonus = 0.0
        
        # Win rate consideration (we need to win auctions)
        win_bonus = clicks * 0.5  # Small bonus for getting clicks
        
        # Cost penalty (scaled)
        cost_penalty = -cost * 0.1  # Small penalty for spending
        
        # Combine rewards
        total_reward = (
            base_reward + 
            conversion_bonus + 
            efficiency_bonus + 
            win_bonus + 
            cost_penalty
        )
        
        # Clip to reasonable range per step
        total_reward = np.clip(total_reward, -50, 100)
        
        return total_reward
