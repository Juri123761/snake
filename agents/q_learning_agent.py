import numpy as np
import random
import pickle
import os
from typing import Tuple


class QLearningAgent:
    
    def __init__(self, state_size: int = 11, action_size: int = 3,
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
    
    def _get_q_value(self, state: np.ndarray, action: int) -> float:
        state_key = self._state_to_key(state)
        return self.q_table.get((state_key, action), 0.0)
    
    def _set_q_value(self, state: np.ndarray, action: int, value: float):
        state_key = self._state_to_key(state)
        self.q_table[(state_key, action)] = value
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = [self._get_q_value(state, a) for a in range(self.action_size)]
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        current_q = self._get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            next_q_values = [self._get_q_value(next_state, a) 
                            for a in range(self.action_size)]
            target_q = reward + self.gamma * max(next_q_values)
        
        new_q = current_q + self.lr * (target_q - current_q)
        self._set_q_value(state, action, new_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
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
