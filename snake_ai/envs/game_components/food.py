import pygame
from snake_ai.envs.game_components.renderable import Renderable
from snake_ai.envs.utils.custom_types import Position

class Food(Renderable):

    def __init__(self, position: Position):

        self.position = position
        self.color = (87, 8, 97)
        self.value = 10.0
    
    def render(self, window: pygame.Surface, pix_square_size: float) -> None:
        
        pygame.draw.rect(
            window,
            self.color,
            pygame.Rect(
                (pix_square_size * self.position[0], pix_square_size * self.position[1]),
                (pix_square_size, pix_square_size),
            ),
        )
