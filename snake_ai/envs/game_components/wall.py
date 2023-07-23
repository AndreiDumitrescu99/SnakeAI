import pygame
from typing import Tuple, List
from snake_ai.envs.game_components.renderable import Renderable
from snake_ai.envs.utils.custom_types import Position, Direction, Action
from snake_ai.envs.utils.position_handler import PositionHandler

class Wall(Renderable):
    
    def __init__(self, position: Position) -> None:
        
        self.position = position
        self._color = (77, 77, 77) if (position[0] + position[1]) % 2 == 0 else (115, 115, 115) #Cul1 :(64, 66, 88), Cul 2: (71, 78, 104)
    
    def render(self, window: pygame.Surface, pix_square_size: float) -> None:
        
        pygame.draw.rect(
            window,
            self._color,
            pygame.Rect(
                (pix_square_size * self.position[0], pix_square_size * self.position[1]),
                (pix_square_size, pix_square_size),
            ),
        )