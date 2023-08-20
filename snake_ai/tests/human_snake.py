import gymnasium as gym
import numpy as np
import pygame

from snake_ai.envs.snake_env import SnakeEnv
from gymnasium.utils.play import play

from gymnasium.envs.registration import register
import time

__all__ = [SnakeEnv]

register(id="Snake-v0", entry_point="snake_ai.envs.snake_env:SnakeEnv")

def play():

    env = gym.make(
        "Snake-v0",
        render_mode="rgb_array",
        window_size=768,
        grid_size=5,
    )

    clock = pygame.time.Clock()

    score = 0
    obs = env.reset()

    while True:
        env.render()

        # Getting action:
        action = 0
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                action = 3
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                action = 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                action = 4
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                action = 2

        # Processing:
        obs, reward, done, _, info = env.step(action)

        score = score + reward
        clock.tick(15)

        if done:
            env.render()
            break

    env.close()
    print("Overall score: ", score)

if __name__ == "__main__":
    play()