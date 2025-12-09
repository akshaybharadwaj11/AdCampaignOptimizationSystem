"""
Multi-Agent Ad Campaign Optimization - Minimal Implementation
Files: single Python module implementing:
- Simple AdEnvironment (simulator)
- DQN BiddingAgent
- PPO ControllerAgent
- BudgetAgent (heuristic)
- AnalyticsAgent (logging)
- Simple CrewAI-like Orchestrator to wire agents

Run: requires Python 3.8+, PyTorch, numpy, matplotlib
Install: pip install torch numpy matplotlib

This is a runnable scaffold for experimentation and fast prototyping.
Do NOT run against real ad platforms without adding safety validators.
"""

"""
Multi-Agent Ad Campaign Optimization - Realistic Simulator
Updated: Added seasonality (time-of-day and day-of-week effects) and multiple campaigns.
"""

import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# ------------------------
# Environment: Multi-Campaign Ad Simulator with Seasonality
# ------------------------
class AdEnvironment:
    """Simulates multiple ad campaigns with time-of-day and day-of-week effects."""
    def __init__(self, episode_length=100, base_budget=1000.0, n_campaigns=3, seed=None):
        self.episode_length = episode_length
        self.base_budget = base_budget
        self.n_campaigns = n_campaigns
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.budget_left = self.base_budget
        # Each campaign has base CTR and CVR
        self.campaigns = []
        for _ in range(self.n_campaigns):
            self.campaigns.append({
                'base_ctr': self.rng.uniform(0.01, 0.05),
                'base_cvr': self.rng.uniform(0.01, 0.1)
            })
        self.last_bid = 1.0
        state = self._get_state()
        return state

    def _get_state(self):
        budget_norm = self.budget_left / self.base_budget
        time_of_day = (self.t % 24) / 24.0
        day_of_week = ((self.t // 24) % 7) / 7.0
        campaign_features = []
        for c in self.campaigns:
            campaign_features.extend([c['base_ctr'], c['base_cvr']])
        return np.array([budget_norm, time_of_day, day_of_week, self.last_bid] + campaign_features, dtype=np.float32)

    def step(self, bid_multiplier, budget_alloc_fraction):
        bid_multiplier = float(np.clip(bid_multiplier, 0.1, 3.0))
        budget_alloc_fraction = float(np.clip(budget_alloc_fraction, 0.0, 1.0))

        spend_budget = self.budget_left * budget_alloc_fraction
        total_reward = 0.0
        total_revenue = 0.0
        total_spend = 0.0
        total_impressions = 0
        total_clicks = 0
        total_conversions = 0

        # Simulate each campaign
        for c in self.campaigns:
            base_cpm = 2.0
            impressions = (spend_budget / (base_cpm * bid_multiplier)) * 1000.0 / self.n_campaigns
            impressions *= self.rng.uniform(0.7, 1.3)

            # Add seasonality effects: peak hours 18-22, weekdays better performance
            hour = self.t % 24
            day = (self.t // 24) % 7
            ctr_factor = 1.0
            if 18 <= hour <= 22:
                ctr_factor += 0.2
            if day < 5:
                ctr_factor += 0.1

            effective_ctr = min(max(c['base_ctr'] * ctr_factor * (1.0 + 0.2 * (bid_multiplier - 1.0)), 1e-6), 0.5)
            clicks = np.random.binomial(n=max(0,int(impressions)), p=effective_ctr)

            conv_prob = min(max(c['base_cvr'] * (1.0 + 0.05 * (bid_multiplier - 1.0)), 1e-6), 0.5)
            conversions = np.random.binomial(n=clicks, p=conv_prob) if clicks > 0 else 0
            avg_order_value = 50.0
            revenue = conversions * avg_order_value

            spend = spend_budget / self.n_campaigns * self.rng.uniform(0.9, 1.1)

            total_reward += revenue - spend
            total_revenue += revenue
            total_spend += spend
            total_impressions += impressions
            total_clicks += clicks
            total_conversions += conversions

        self.budget_left = max(0.0, self.budget_left - total_spend)
        self.last_bid = bid_multiplier
        self.t += 1
        done = (self.t >= self.episode_length) or (self.budget_left <= 1e-3)
        next_state = self._get_state()
        info = {
            'impressions': total_impressions,
            'clicks': total_clicks,
            'conversions': total_conversions,
            'revenue': total_revenue,
            'spend': total_spend
        }
        return next_state.astype(np.float32), float(total_reward), bool(done), info

# ------------------------
# Replay Buffer for DQN
# ------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ------------------------
# Neural network building blocks
# ------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128, 128), activate_final=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        if not activate_final:
            pass
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------
# DQN Bidding Agent
# ------------------------
class DQNBiddingAgent:
    def __init__(self, state_dim, n_actions=7, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=500):
        # discretize bid multipliers into buckets
        self.actions = np.linspace(0.5, 2.0, n_actions)  # e.g., 0.5x ... 2.0x
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        self.q_net = MLP(state_dim, n_actions).to(DEVICE)
        self.target_q = MLP(state_dim, n_actions).to(DEVICE)
        self.target_q.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(20000)

    def select_action(self, state, eval_mode=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1.0 * self.steps / self.epsilon_decay)
        self.steps += 1
        if eval_mode or random.random() > eps:
            with torch.no_grad():
                q = self.q_net(state_t)
                action_idx = int(torch.argmax(q, dim=1).cpu().item())
        else:
            action_idx = random.randrange(self.n_actions)
        return float(self.actions[action_idx]), action_idx

    def push(self, *args):
        self.replay.push(*args)

    def update(self, batch_size=64, sync_every=500):
        if len(self.replay) < batch_size:
            return
        trans = self.replay.sample(batch_size)
        state = torch.tensor(np.array(trans.state), dtype=torch.float32, device=DEVICE)
        action = torch.tensor(np.array(trans.action), dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward = torch.tensor(np.array(trans.reward), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state = torch.tensor(np.array(trans.next_state), dtype=torch.float32, device=DEVICE)
        done = torch.tensor(np.array(trans.done), dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_values = self.q_net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_q(next_state).max(1)[0].unsqueeze(1)
            expected = reward + (1.0 - done) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target
        for param, target_param in zip(self.q_net.parameters(), self.target_q.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

# ------------------------
# PPO Controller Agent
# ------------------------
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128,128)):
        super().__init__()
        self.actor = MLP(state_dim, action_dim, hidden)
        self.critic = MLP(state_dim, 1, hidden)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

class PPOController:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip=0.2, epochs=4, batch_size=64):
        self.net = PPONetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size

        # storage
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self.net(state_t)
        probs = nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.cpu().item(), logprob.cpu().item(), value.cpu().item()

    def store(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def finish_episode(self):
        # compute returns and advantages
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=DEVICE)
        old_logprobs = torch.tensor(np.array(self.logprobs), dtype=torch.float32, device=DEVICE)

        # compute discounted returns
        returns = []
        R = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_logprobs, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_old_logprobs, batch_returns in loader:
                logits, values = self.net(batch_states)
                probs = nn.functional.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # advantages
                advantages = batch_returns - values.detach()

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values, batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # clear buffers
        self.states, self.actions, self.logprobs, self.rewards, self.dones = [], [], [], [], []

# ------------------------
# BudgetAgent (heuristic)
# ------------------------
class BudgetAgent:
    """Simple heuristic budget allocation agent.
    Produces fraction of remaining budget to spend each step based on remaining steps and budget.
    """
    def __init__(self, lookahead=20):
        self.lookahead = lookahead

    def allocate(self, budget_left, steps_left):
        # naive pacing: spend budget evenly over remaining steps
        if steps_left <= 0:
            return 0.0
        fraction = min(1.0, budget_left / (budget_left + 1.0)) * (1.0 / max(1, steps_left))
        # keep fraction reasonable
        return float(np.clip(fraction * 1.0, 0.01, 0.2))

# ------------------------
# AnalyticsAgent (logger / lightweight advisor)
# ------------------------
class AnalyticsAgent:
    def __init__(self):
        self.logs = []

    def log(self, info):
        self.logs.append(info)

    def summarize(self, last_k=50):
        recent = self.logs[-last_k:]
        if not recent:
            return {}
        avg_reward = np.mean([r['reward'] for r in recent])
        avg_revenue = np.mean([r['revenue'] for r in recent])
        avg_spend = np.mean([r['spend'] for r in recent])
        return {'avg_reward': avg_reward, 'avg_revenue': avg_revenue, 'avg_spend': avg_spend}

# ------------------------
# Orchestrator (CrewAI-like minimal)
# ------------------------
class Orchestrator:
    def __init__(self, env, controller, bidding_agent, budget_agent, analytics_agent, max_steps=100):
        self.env = env
        self.controller = controller
        self.bidding_agent = bidding_agent
        self.budget_agent = budget_agent
        self.analytics = analytics_agent
        self.max_steps = max_steps

    def run_episode(self, train=True, render=False):
        state = self.env.reset()
        total_reward = 0.0
        step = 0
        done = False
        # Controller uses discrete action space: select an index -> maps to (bid_selector, budget_scale)
        while not done and step < self.max_steps:
            # Controller observes state
            # For simplicity, controller's action space we'll set to 5 discrete choices meaning different combos
            ctrl_action_idx, logprob, _ = self.controller.select_action(state)

            # map controller action index to a high-level directive
            # choices: 0: conservative, 1: moderate, 2: aggressive, 3: expand audience, 4: reduce spend
            # We'll map these to bid suggestions and budget pacing multipliers
            mapping = {
                0: {'bid_hint': 0.8, 'budget_factor': 0.7},
                1: {'bid_hint': 1.0, 'budget_factor': 1.0},
                2: {'bid_hint': 1.3, 'budget_factor': 1.2},
                3: {'bid_hint': 1.1, 'budget_factor': 1.0},
                4: {'bid_hint': 0.7, 'budget_factor': 0.5},
            }
            directive = mapping.get(ctrl_action_idx, mapping[1])

            # ask bidding agent for precise multiplier given state
            bid_multiplier, bid_idx = self.bidding_agent.select_action(state, eval_mode=not train)
            # combine controller's hint and bidding agent's suggestion (weighted average)
            combined_bid = 0.6 * bid_multiplier + 0.4 * directive['bid_hint']

            # budget agent returns fraction to spend (pacing)
            steps_left = max(1, self.env.episode_length - self.env.t)
            budget_fraction = self.budget_agent.allocate(self.env.budget_left, steps_left) * directive['budget_factor']

            # step environment
            next_state, reward, done, info = self.env.step(combined_bid, budget_fraction)

            # store transitions
            # Controller stores symbolic action (use ctrl_action_idx)
            self.controller.store(state, ctrl_action_idx, logprob, reward, done)
            # Bidding agent stores (use bid_idx)
            self.bidding_agent.push(state, bid_idx, reward, next_state, done)

            # analytics
            analytics_info = {
                't': self.env.t,
                'state': state,
                'ctrl_action': ctrl_action_idx,
                'bid_idx': bid_idx,
                'combined_bid': combined_bid,
                'budget_fraction': budget_fraction,
                'reward': reward,
                'revenue': info['revenue'],
                'spend': info['spend'],
                'impressions': info['impressions'],
                'clicks': info['clicks'],
                'conversions': info['conversions']
            }
            self.analytics.log(analytics_info)

            state = next_state
            total_reward += reward
            step += 1

            # optional: update bidding agent periodically
            if train:
                self.bidding_agent.update(batch_size=64)

        # after episode
        if train:
            self.controller.finish_episode()

        return total_reward

# ------------------------
# Training script
# ------------------------
def train_loop(episodes=200, episode_length=100):
    env = AdEnvironment(episode_length=episode_length)
    state_dim = env.reset().shape[0]

    # Controller: discrete action space 5
    controller = PPOController(state_dim=state_dim, action_dim=5)
    bidding_agent = DQNBiddingAgent(state_dim=state_dim, n_actions=7)
    budget_agent = BudgetAgent()
    analytics = AnalyticsAgent()

    orch = Orchestrator(env, controller, bidding_agent, budget_agent, analytics, max_steps=episode_length)

    rewards = []
    for ep in range(episodes):
        r = orch.run_episode(train=True)
        rewards.append(r)
        if (ep+1) % 10 == 0:
            summary = analytics.summarize(last_k=200)
            print(f"Episode {ep+1}/{episodes} | Reward: {r:.2f} | Summary: {summary}")

    # plot training curve
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()
    return controller, bidding_agent, analytics, budget_agent

# ------------------------
# Quick evaluation routine
# ------------------------
def evaluate(controller, bidding_agent, budget_agent, runs=10, episode_length=100):
    env = AdEnvironment(episode_length=episode_length)
    total = []
    for _ in range(runs):
        state = env.reset()
        done = False
        R = 0
        steps = 0
        while not done and steps < episode_length:
            action_idx, logprob, _ = controller.select_action(state)
            mapping = {
                0: {'bid_hint': 0.8, 'budget_factor': 0.7},
                1: {'bid_hint': 1.0, 'budget_factor': 1.0},
                2: {'bid_hint': 1.3, 'budget_factor': 1.2},
                3: {'bid_hint': 1.1, 'budget_factor': 1.0},
                4: {'bid_hint': 0.7, 'budget_factor': 0.5},
            }
            directive = mapping.get(action_idx, mapping[1])
            bid_multiplier, bid_idx = bidding_agent.select_action(state, eval_mode=True)
            combined_bid = 0.6 * bid_multiplier + 0.4 * directive['bid_hint']
            steps_left = max(1, env.episode_length - env.t)
            budget_fraction = budget_agent.allocate(env.budget_left, steps_left) * directive['budget_factor']
            next_state, reward, done, info = env.step(combined_bid, budget_fraction)
            state = next_state
            R += reward
            steps += 1
        total.append(R)
    print(f"Evaluation over {runs} runs: avg reward = {np.mean(total):.2f}, std = {np.std(total):.2f}")
    return total

# ------------------------
# Entrypoint for quick runs
# ------------------------
if __name__ == '__main__':
    ctrl, bid, analytics, budget = train_loop(episodes=2000, episode_length=100)
    # You can evaluate after training
    evaluate(ctrl, bid, budget, runs=10, episode_length=100)

"""
Notes & Next Steps:
- This scaffold uses a heuristic BudgetAgent. You can replace it with an RL agent (PPO/DDPG).
- CrewAI integration is mimicked by the Orchestrator. To integrate with real CrewAI, implement each agent as a CrewAI agent and expose tools for environment interactions and logging.
- Add safety validators before using on real systems.
- Improve simulator realism with logged data.
"""
