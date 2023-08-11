import gymnasium as gym
import torch
from snake_ai.envs.snake_env import SnakeEnv

class TorchWrapper(gym.ObservationWrapper):
    """ Applies a couple of transformations depending on the mode.
        Receives numpy arrays and returns torch tensors.
    """

    def __init__(self, env: SnakeEnv, device: torch.device):
        super().__init__(env)
        self._device = device
    
    def observation(self, obs):
        return torch.from_numpy(obs).float().unsqueeze(0).to(self._device)