import pygame
from abc import ABC, abstractmethod

class Renderable(ABC):

    @abstractmethod
    def render(self, window: pygame.Surface, pix_square_size: float) -> None:
        pass