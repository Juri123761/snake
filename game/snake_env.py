import numpy as np
import random
from typing import Tuple, List


GRID_WIDTH = 17
GRID_HEIGHT = 15


class SnakeEnv:
    
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        self.width = width
        self.height = height
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.direction_to_idx = {d: i for i, d in enumerate(self.directions)}
        self.reset()
    
    def reset(self) -> np.ndarray:
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = (1, 0)
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        return self.get_state()
    
    def _spawn_food(self) -> Tuple[int, int]:
        max_cells = self.width * self.height
        snake_set = set(self.snake)
        if len(snake_set) >= max_cells:
            raise RuntimeError("Grid is full, cannot spawn food")
        
        free_cells = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in snake_set]
        return random.choice(free_cells)
    
    def _move_direction(self, action: int) -> Tuple[int, int]:
        current_idx = self.direction_to_idx[self.direction]
        if action == 0:
            new_idx = (current_idx - 1) % 4
        elif action == 1:
            new_idx = current_idx
        else:
            new_idx = (current_idx + 1) % 4
        return self.directions[new_idx]
    
    def _check_collision(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if len(self.snake) > 1:
            return pos in self.snake[1:]
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.done:
            return self.get_state(), 0.0, True
        
        self.steps += 1
        self.steps_since_food += 1
        
        self.direction = self._move_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        if self._check_collision(new_head):
            self.done = True
            return self.get_state(), -100.0, True
        
        old_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        self.snake.insert(0, new_head)
        
        reward = 0.05
        
        if new_head == self.food:
            self.score += 1
            reward = 30.0 + (len(self.snake) * 0.5)
            self.steps_since_food = 0
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            if self.steps_since_food > 150:
                self.done = True
                return self.get_state(), -30.0, True
        
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        distance_reward = (old_distance - new_distance) * 3.0
        reward += distance_reward
        
        return self.get_state(), reward, self.done
    
    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        state = np.zeros(16, dtype=np.float32)
        
        current_idx = self.direction_to_idx[self.direction]
        
        straight = (head[0] + self.direction[0], head[1] + self.direction[1])
        left_dir = self.directions[(current_idx - 1) % 4]
        right_dir = self.directions[(current_idx + 1) % 4]
        left = (head[0] + left_dir[0], head[1] + left_dir[1])
        right = (head[0] + right_dir[0], head[1] + right_dir[1])
        
        state[0] = 1.0 if self._check_collision(straight) else 0.0
        state[1] = 1.0 if self._check_collision(left) else 0.0
        state[2] = 1.0 if self._check_collision(right) else 0.0
        
        state[3 + current_idx] = 1.0
        
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        
        if dx == 0 and dy == 0:
            food_dir = current_idx
        elif abs(dx) > abs(dy):
            food_dir = 0 if dx > 0 else 2
        elif abs(dy) > abs(dx):
            food_dir = 1 if dy > 0 else 3
        else:
            food_dir = 0 if dx > 0 else 2
        
        state[7 + food_dir] = 1.0
        
        state[11] = len(self.snake) / (self.width * self.height)
        
        max_dist = self.width + self.height
        state[12] = abs(dx) / max_dist
        state[13] = abs(dy) / max_dist
        
        state[14] = head[0] / self.width
        state[15] = head[1] / self.height
        
        return state
    
    def get_score(self) -> int:
        return self.score
    
    def get_snake(self) -> List[Tuple[int, int]]:
        return self.snake.copy()
    
    def get_food(self) -> Tuple[int, int]:
        return self.food
