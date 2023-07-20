import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium import Wrapper

from snake_ai.envs.game_components.snake import Snake

class GridWorldEnv(gym.Env):

    def __init__(self, window_size: int = 512, grid_size: int = 8):
        """
        """

        self.window_size = window_size
        self.grid_size = grid_size

        self.window = None
        self.clock = None

        self.debug_grid = True

        mid_point = self.grid_size // 2 - 1 if self.grid_size % 2 == 0 else self.grid_size // 2
        # Get Snake Entity
        self.snake = Snake(
            size=5,
            initial_head_position=(mid_point, mid_point)
        )
    
    def step(self, action: any):
        """
        """

        self._render_frame()
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        """
        """

        if self.window is None:
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = (
            self.window_size / self.grid_size
        )

        #Draw objects
        self.snake.render(canvas, pix_square_size)

        if self.debug_grid:
            for x in range(self.grid_size + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size * x),
                    (self.window_size, pix_square_size * x),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(1)

if __name__ == "__main__":

    grid_env = GridWorldEnv(
        window_size=1024,
        grid_size=32
    )
    while True:
        grid_env.step(0)

        
