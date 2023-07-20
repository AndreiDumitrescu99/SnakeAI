import pygame
from typing import Tuple, List
from snake_ai.envs.game_components.renderable import Renderable
from snake_ai.envs.utils.custom_types import Position
from snake_ai.envs.utils.position_handler import PositionHandler

class Snake(Renderable):

    def __init__(self, size: int, initial_head_position: Position):

        self.size = size
        self.head_position = initial_head_position

        self.body_parts = [self.head_position]
        for _ in range(self.size - 1):
            self.body_parts.append(PositionHandler.get_right_position(self.body_parts[-1]))

        self._body_parts_colors = [(0, 255, 255), (127, 255, 212)]
    
    def render(self, window: pygame.Surface, pix_square_size) -> None:
        
        for i, body_part in enumerate(self.body_parts):

            pygame.draw.rect(
                window,
                self._body_parts_colors[i % 2],
                pygame.Rect(
                    (pix_square_size * body_part[0], pix_square_size * body_part[1]),
                    (pix_square_size, pix_square_size),
                ),
            )
    
