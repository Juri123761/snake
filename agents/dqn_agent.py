import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Optional

from models.dqn_model import DuelingDQNModel


class ReplayMemory:
    
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
                 device: Optional[torch.device] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.q_network = DuelingDQNModel(state_size, output_size=action_size).to(self.device)
        self.target_network = DuelingDQNModel(state_size, output_size=action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, eps=1e-4)
        
        self.memory = ReplayMemory(memory_size)
        self.warmup_steps = batch_size
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self):
        if len(self.memory) < self.batch_size or self.steps < self.warmup_steps:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(1)
            next_q_values = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards_t + (self.gamma * next_q_values * ~dones_t)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        try:
            dirname = os.path.dirname(filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.steps = checkpoint.get('steps', 0)
        except Exception as e:
            raise IOError(f"Failed to load model from {filepath}: {e}")
