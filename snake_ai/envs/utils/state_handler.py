from typing import List
from snake_ai.envs.utils.custom_types import Position, ComponentCode
import numpy as np
from copy import deepcopy

class StateHandler:

    def __init__(self, map_size: int, walls: List[Position], snake_position: List[Position]) -> None:
        
        self.map_size = map_size
        self.walls = walls
        self.snake_position = deepcopy(snake_position)

        self.map = np.zeros((self.map_size, self.map_size))

        self._init_map()
    
    def _init_map(self) -> None:

        for wall in self.walls:
            self.map[wall.position[1], wall.position[0]] = ComponentCode.WALL.value

    def _check_collision(self, position: Position) -> bool:

        for wall in self.walls:
            if position[0] == wall.position[0] and position[1] == wall.position[1]:
                return True
        
        return False
    
    def get_observation(self) -> np.ndarray:

        return self.map

    def update_state(self, snake_position: List[Position]) -> bool:

        for position in snake_position:
            if self._check_collision(position):
                return False
        
        for position in self.snake_position:
            self.map[position[1], position[0]] = ComponentCode.EMPTY_SPACE.value
        
        for position in snake_position:
            self.map[position[1], position[0]] = ComponentCode.SNAKE.value

        self.snake_position = deepcopy(snake_position)

        return True
    
    def _print_map(self) -> None:

        print(self.map)
