from snake_ai.envs.utils.custom_types import Position

class PositionHandler:

    def __init__(self) -> None:
        pass
    
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