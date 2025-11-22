import numpy as np
import random
import pickle
import os
from typing import Tuple


class QLearningAgent:
    
    def __init__(self, state_size: int = 16, action_size: int = 3,
                 learning_rate: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
    
    def _state_to_key(self, state: np.ndarray) -> Tuple[int, ...]:
        return tuple((state > 0.5).astype(int))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self._state_to_key(state)
        best_action = 0
        best_q = self.q_table.get((state_key, 0), 0.0)
        for a in range(1, self.action_size):
            q_val = self.q_table.get((state_key, a), 0.0)
            if q_val > best_q:
                best_q = q_val
                best_action = a
        return best_action
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        state_key = self._state_to_key(state)
        current_q = self.q_table.get((state_key, action), 0.0)
        
        if done:
            target_q = reward
        else:
            next_state_key = self._state_to_key(next_state)
            max_next_q = max(self.q_table.get((next_state_key, a), 0.0) 
                           for a in range(self.action_size))
            target_q = reward + self.gamma * max_next_q
        
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[(state_key, action)] = new_q
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        try:
            dirname = os.path.dirname(filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            raise IOError(f"Failed to save Q-table to {filepath}: {e}")
    
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Q-table file not found: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load Q-table from {filepath}: {e}")
