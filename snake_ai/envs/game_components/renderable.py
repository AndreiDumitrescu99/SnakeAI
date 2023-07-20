import pygame
from abc import ABC, abstractmethod

class Renderable(ABC):

    @abstractmethod
    def render(self, window: pygame.Surface) -> None:
        pass