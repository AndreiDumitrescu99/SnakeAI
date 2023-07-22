from snake_ai.envs.utils.custom_types import Position, Direction

class PositionHandler:

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def move_to_direction(position: Position, direction: Direction) -> Position:

        if direction == Direction.WEST:
            return PositionHandler.get_left_position(position)
        elif direction == Direction.EAST:
            return PositionHandler.get_right_position(position)
        elif direction == Direction.NORTH:
            return PositionHandler.get_up_position(position)
        elif direction == Direction.SOUTH:
            return PositionHandler.get_down_position(position)

    @staticmethod
    def get_left_position(position: Position) -> Position:

        return (position[0] - 1, position[1])
    
    @staticmethod
    def get_right_position(position: Position) -> Position:

        return (position[0] + 1, position[1])
    
    @staticmethod
    def get_up_position(position: Position) -> Position:

        return (position[0], position[1] - 1)
    
    @staticmethod
    def get_down_position(position: Position) -> Position:

        return (position[0], position[1] + 1)