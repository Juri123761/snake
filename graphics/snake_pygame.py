import pygame
import sys
from typing import Tuple, List


class SnakeRenderer:
    
    def __init__(self, width: int = 17, height: int = 15, cell_size: int = 30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.window_width = width * cell_size
        self.window_height = height * cell_size
        
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()
        except pygame.error as e:
            raise RuntimeError(f"Failed to initialize Pygame: {e}")
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.DARK_GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        
        try:
            self.font = pygame.font.Font(None, 36)
        except pygame.error:
            self.font = pygame.font.SysFont('arial', 36)
    
    def render(self, snake: List[Tuple[int, int]], food: Tuple[int, int], score: int):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill(self.BLACK)
        
        for i in range(self.width + 1):
            pygame.draw.line(self.screen, self.GRAY, 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.window_height))
        for i in range(self.height + 1):
            pygame.draw.line(self.screen, self.GRAY, 
                           (0, i * self.cell_size), 
                           (self.window_width, i * self.cell_size))
        
        for i, (x, y) in enumerate(snake):
            color = self.GREEN if i == 0 else self.DARK_GREEN
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        food_rect = pygame.Rect(food[0] * self.cell_size, food[1] * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.RED, food_rect)
        pygame.draw.rect(self.screen, self.BLACK, food_rect, 1)
        
        score_text = self.font.render(f"Score: {score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(10)
    
    def close(self):
        pygame.quit()
