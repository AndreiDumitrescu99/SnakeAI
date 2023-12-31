import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.distributions import Categorical

from typing import cast

from snake_ai.models.a2c.policy import ActorCriticPolicy

class A2C:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        gamma: float,
        optimizer: Optimizer,
        nsteps: int = 5,
        device: torch.device = torch.device('cpu')
    ):

        self._policy = policy
        self._gamma = gamma
        self._optimizer = optimizer
        self._device = device
        self._nsteps = nsteps

        self._beta = 0.01       # beta term in entropy regularization
        self._values = []       # keeps episodic/nstep value estimates
        self._entropies = []    # keeps episodic/nstep policy entropies
        self._fp32_err = 2e-07  # used to avoid division by 0
        self._log_probs = []
        self._rewards = []
        self._step_cnt = 0

    def act(self, state: torch.Tensor) -> int:

        # Extract pi and value from the policy network.
        pi, value = self._policy(state)

        # Hint type of pi. TODO: Make sure to not disrupt GPU training.
        pi = cast(Categorical, pi)

        # Sample action.
        action = pi.sample()

        self._log_probs.append(pi.log_prob(action))
        self._values.append(value)
        self._entropies.append(pi.entropy())

        return action.item()

    def best_act(self, state: torch.Tensor) -> int:

        pi, _ = self._policy(state)
        pi = cast(Categorical, pi)

        return torch.argmax(pi.probs).item()
    
    def learn(self, reward: float, state_: torch.Tensor, done: bool):

        self._rewards.append(reward)

        if done or (self._step_cnt % (self._nsteps - 1) == 0 and self._step_cnt != 0):
            self._update_policy(done, state_)

        self._step_cnt = 0 if done else self._step_cnt + 1
    
    def _compute_returns(self, done: bool, state_: torch.Tensor, tau: float = 0.95) -> torch.Tensor:

        returns = []
        R = self._policy(state_)[1].detach() * (1 - done)
        for r in self._rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # if len(returns) > 1:
        #     returns = (returns - returns.mean()) / (returns.std() + self._fp32_err)

        return returns

    def _update_policy(self, done: bool, state_: torch.Tensor):

        returns = self._compute_returns(done, state_)

        values = torch.cat(self._values).squeeze(1)
        log_probs = torch.cat(self._log_probs)
        entropy = torch.cat(self._entropies)
        advantage = returns.to(values.device) - values

        policy_loss = (-log_probs * advantage.detach()).sum()
        critic_loss = F.smooth_l1_loss(values.to(self._device), returns.to(self._device))

        self._optimizer.zero_grad()
        (policy_loss + critic_loss - self._beta * entropy.mean()).backward() 
        self._optimizer.step()

        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._entropies.clear()