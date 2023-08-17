from typing import TypeAlias, Tuple
from enum import Enum

Position = Tuple[int, int]

class Direction(Enum):
    WEST = 'WEST'
    EAST = 'EAST'
    SOUTH = 'SOUTH'
    NORTH = 'NORTH'

class Action(Enum):
    NOOP = 0
    MOVE_UP = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    MOVE_LEFT = 4

class ComponentCode(Enum):
    EMPTY_SPACE = 1
    SNAKE = 2
    SNAKE_HEAD = 3
    WALL = 4
    FOOD = 5

class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

class Reward(Enum):
    DEATH = -10.0
    MOVE = -0.1