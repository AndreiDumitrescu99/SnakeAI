import pygame
from typing import Tuple, List
from snake_ai.envs.game_components.renderable import Renderable
from snake_ai.envs.utils.custom_types import Position, Direction, Action
from snake_ai.envs.utils.position_handler import PositionHandler

class Snake(Renderable):

    def __init__(self, size: int, initial_head_position: Position):

        self.size = size
        self.head_position = initial_head_position

        self.body_parts = [self.head_position]
        for _ in range(self.size - 1):
            self.body_parts.append(PositionHandler.get_right_position(self.body_parts[-1]))

        self._body_parts_colors = [(0, 255, 255), (127, 255, 212)]

        self._heading_direction: Direction = Direction.WEST

        self._map_action_to_heading_direction = {
            Action.MOVE_UP.value: Direction.NORTH,
            Action.MOVE_DOWN.value: Direction.SOUTH,
            Action.MOVE_LEFT.value: Direction.WEST,
            Action.MOVE_RIGHT.value: Direction.EAST
        }
    
    def render(self, window: pygame.Surface, pix_square_size: float) -> None:
        
        for i, body_part in enumerate(self.body_parts):

            pygame.draw.rect(
                window,
                self._body_parts_colors[i % 2],
                pygame.Rect(
                    (pix_square_size * body_part[0], pix_square_size * body_part[1]),
                    (pix_square_size, pix_square_size),
                ),
            )
    
    def act(self, action: Action):
        
        self._heading_direction = self._heading_direction if action == Action.NOOP.value else self._map_action_to_heading_direction[action]
        future_head_position = PositionHandler.move_to_direction(self.head_position, self._heading_direction)
        self.body_parts.pop()
        self.body_parts.insert(0, future_head_position)
        self.head_position = future_head_position
    
