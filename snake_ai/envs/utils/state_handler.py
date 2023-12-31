from typing import List, Tuple
import torch as th
import numpy as np
from copy import deepcopy
from snake_ai.envs.utils.custom_types import Position, ComponentCode, Reward
from snake_ai.envs.game_components.food import Food
from snake_ai.envs.game_components.wall import Wall

class StateHandler:

    def __init__(self, map_size: int, walls: List[Wall], snake_position: List[Position], number_of_rewards: int) -> None:
        
        self.map_size = map_size
        self.walls = walls
        self.snake_position = deepcopy(snake_position)
        
        self.number_of_rewards = number_of_rewards
        self.rewards: List[Food] = []
        self.map = np.ones((self.map_size, self.map_size), dtype=np.float32)

        self._init_map()
    
    def _init_map(self) -> None:

        for wall in self.walls:
            self.map[wall.position[1], wall.position[0]] = ComponentCode.WALL.value

        for position in self.snake_position:
            self.map[position[1], position[0]] = ComponentCode.SNAKE.value

        self.map[self.snake_position[0][1], self.snake_position[0][0]] = ComponentCode.SNAKE_HEAD.value
        # Init rewards
        self.fill_rewards()

    def _get_random_reward(self) -> Position:

        while True:

            x = np.random.randint(0, self.map_size)
            y = np.random.randint(0, self.map_size)

            if self.map[y, x] == ComponentCode.EMPTY_SPACE.value:
                return x, y
            
    def fill_rewards(self) -> None:

        while len(self.rewards) < self.number_of_rewards:
            position = self._get_random_reward()

            self.map[position[1], position[0]] = ComponentCode.FOOD.value
            self.rewards.append(Food(position))

    @staticmethod
    def check_position_overlap(a: Position, b: Position) -> bool:

        return a[0] == b[0] and a[1] == b[1]

    def _check_wall_collision(self, position: Position) -> bool:

        for wall in self.walls:
            if StateHandler.check_position_overlap(position, wall.position):
                return True
        
        return False
    
    def _check_food_collision(self, position: Position) -> Food | None:

        for reward in self.rewards:
            if StateHandler.check_position_overlap(position, reward.position):
                return reward

        return None
    
    def get_observation(self) -> np.ndarray:

        return np.reshape(self.map / 5, [1, self.map_size, self.map_size])

    def update_state(self, snake_position: List[Position]) -> float:

        overall_score = Reward.MOVE.value
        for position in snake_position:
            if self._check_wall_collision(position):
                for position in self.snake_position:
                    self.map[position[1], position[0]] = ComponentCode.EMPTY_SPACE.value
        
                for position in snake_position:
                    self.map[position[1], position[0]] = ComponentCode.SNAKE.value
                
                self.map[snake_position[0][1], snake_position[0][0]] = ComponentCode.WALL.value
                return Reward.DEATH.value
        
        sem = 0
        for position in snake_position:
            reward = self._check_food_collision(position)
            if reward is not None:
                self.rewards.remove(reward)
                self.map[reward.position[0], reward.position[1]] = ComponentCode.EMPTY_SPACE.value
                overall_score += reward.value
                sem = 1

        if sem == 0:
            distance = self.map_size ** 2
            for reward in self.rewards:
                current_distance = np.abs(reward.position[0] - snake_position[0][0]) + np.abs(reward.position[1] - snake_position[0][1])
                distance = min(distance, current_distance)
            
            overall_score = overall_score - ((self.map_size - 2) / 2 + 1) * Reward.MOVE.value / distance          

        for position in self.snake_position:
            self.map[position[1], position[0]] = ComponentCode.EMPTY_SPACE.value
        
        for position in snake_position:
            self.map[position[1], position[0]] = ComponentCode.SNAKE.value

        self.map[snake_position[0][1], snake_position[0][0]] = ComponentCode.SNAKE_HEAD.value

        for i, position in enumerate(snake_position):

            if StateHandler.check_position_overlap(snake_position[0], snake_position[i]) and i != 0:
                return Reward.DEATH.value
  
        self.snake_position = deepcopy(snake_position)

        self.fill_rewards()
        return overall_score
    
    def _print_map(self) -> None:

        print(self.map)
