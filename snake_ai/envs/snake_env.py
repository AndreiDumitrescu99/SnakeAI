import numpy as np
import pygame
from typing import List

import gymnasium as gym
from gymnasium import spaces
from gymnasium import Wrapper

from snake_ai.envs.game_components.snake import Snake
from snake_ai.envs.game_components.wall import Wall
from snake_ai.envs.utils.state_handler import StateHandler
from snake_ai.envs.game_components.food import Food

class SnakeEnv(gym.Env):

    def __init__(self, window_size: int = 512, grid_size: int = 8, number_of_rewards: int = 1, debug_grid: bool = True, render_mode="rgb_array"):
        """
        """

        self.window_size = window_size
        self.grid_size = grid_size + 2 # We add the outer walls.

        self.window = None
        self.clock = None

        self.debug_grid = debug_grid

        mid_point = self.grid_size // 2 - 1 if self.grid_size % 2 == 0 else self.grid_size // 2

        # Get Snake Entity.
        self.snake = Snake(
            size=5,
            initial_head_position=(mid_point, mid_point)
        )

        # Get walls.
        self.walls: List[Wall] = []
        self._init_outer_walls()

        # Define rewards.
        self.rewards: List[Food] = []
        self.number_of_rewards = number_of_rewards

        # Define actions space.
        self.action_space = gym.spaces.Discrete(5)

        # Define observation space.
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.grid_size, self.grid_size), dtype=np.uint8)

        # Define render modes.
        self.render_modes = ["rgb_array_list", "rgb_array"]

        # Define metadata.
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

        # Define render mode.
        self.render_mode = render_mode

        self.tick = 0
        self.action = 0

        # Init State Handler.
        self.state_handler = StateHandler(self.grid_size, self.walls, self.snake.body_parts, self.number_of_rewards)
        self.rewards = self.state_handler.rewards

        self.state_handler._print_map()
    
    def _init_outer_walls(self):

        for i in range(self.grid_size):

            self.walls.append(Wall((0, i)))
            self.walls.append(Wall((i, 0)))

            self.walls.append(Wall((self.grid_size - 1, i)))
            self.walls.append(Wall((i, self.grid_size - 1)))
    
    def step(self, action: any):
        """
        """
        self.action = action
        self.snake.act(self.action)

        reward = self.state_handler.update_state(self.snake.body_parts)
        self.rewards = self.state_handler.rewards
        observation = self.state_handler.get_observation()

        # TODO! Change This!
        is_alive = True if reward != -10 else False

        return observation, reward, not is_alive, False, {}
    
    def close(self):

        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def render(self) -> None:
        return self._render_frame()

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
        for wall in self.walls:
            wall.render(canvas, pix_square_size)
        for food in self.rewards:
            food.render(canvas, pix_square_size)

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
        self.clock.tick(30)
