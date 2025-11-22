import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Optional, Tuple

from models.dqn_model import DuelingDQNModel


class SumTree:
    """Efficient data structure for prioritized experience replay sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])
    
    def total(self) -> float:
        return self.tree[0]
    
    def add(self, p: float, data: Tuple):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, Tuple]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.data[data_idx]


class PrioritizedReplayMemory:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.0001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.beta_max = 1.0
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        if batch_size > len(self.tree.data):
            raise ValueError(f"Batch size {batch_size} exceeds memory size {len(self.tree.data)}")
        
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        self.beta = min(self.beta_max, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(self.tree.tree[idx])
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(len(self.tree.data) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, is_weights)
    
    def update_priorities(self, indices: list, td_errors: np.ndarray):
        td_errors = np.abs(td_errors) + 1e-6
        priorities = np.power(td_errors, self.alpha)
        self.max_priority = max(self.max_priority, priorities.max())
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


class ReplayMemory:
    """Standard uniform replay buffer (fallback)."""
    
    def __init__(self, capacity: int = 50000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        if batch_size > len(self.memory):
            raise ValueError(f"Batch size {batch_size} exceeds memory size {len(self.memory)}")
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    
    def __init__(self, state_size: int = 16, action_size: int = 3,
                 learning_rate: float = 0.00025, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.9999,
                 epsilon_min: float = 0.01, memory_size: int = 50000,
                 batch_size: int = 64, target_update: int = 1000,
                 tau: float = 0.005, use_prioritized: bool = True,
                 n_step: int = 3, device: Optional[torch.device] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.use_prioritized = use_prioritized
        self.n_step = n_step
        self.steps = 0
        
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.q_network = DuelingDQNModel(state_size, output_size=action_size).to(self.device)
        self.target_network = DuelingDQNModel(state_size, output_size=action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, eps=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)
        
        if use_prioritized:
            self.memory = PrioritizedReplayMemory(memory_size)
        else:
            self.memory = ReplayMemory(memory_size)
        
        self.warmup_steps = batch_size
        self.n_step_buffer = deque(maxlen=n_step)
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(self.n_step_buffer) >= self.n_step or done:
                n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
                self.memory.push(n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
        else:
            self.memory.push(state, action, reward, next_state, done)
        
        self.steps += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _get_n_step_info(self) -> Tuple:
        n_step_reward = 0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * reward
            if done:
                break
        
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        last_next_state, _, _, _, last_done = self.n_step_buffer[-1]
        
        return first_state, first_action, n_step_reward, last_next_state, last_done
    
    def train(self):
        if len(self.memory) < self.batch_size or self.steps < self.warmup_steps:
            return
        
        if self.use_prioritized:
            states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(self.batch_size)
            is_weights_t = torch.FloatTensor(is_weights).to(self.device)
        else:
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            is_weights_t = None
            indices = None
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(1)
            next_q_values = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            gamma_n = self.gamma ** self.n_step if self.n_step > 1 else self.gamma
            target_q_values = rewards_t + (gamma_n * next_q_values * ~dones_t)
        
        td_errors = target_q_values - q_values
        loss = self.huber_loss(q_values, target_q_values)
        
        if is_weights_t is not None:
            loss = (is_weights_t * loss).mean()
        else:
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.use_prioritized and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_np)
        
        if self.tau > 0:
            self._soft_update_target_network()
        elif self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.steps % 1000 == 0:
            self.scheduler.step()
    
    def _soft_update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def reset_n_step_buffer(self):
        """Clear n-step buffer (call when episode ends)."""
        self.n_step_buffer.clear()
    
    def save(self, filepath: str):
        try:
            dirname = os.path.dirname(filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps
            }, filepath)
        except Exception as e:
            raise IOError(f"Failed to save model to {filepath}: {e}")
    
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.steps = checkpoint.get('steps', 0)
        except Exception as e:
            raise IOError(f"Failed to load model from {filepath}: {e}")
