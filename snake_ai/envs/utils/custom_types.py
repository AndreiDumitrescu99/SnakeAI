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