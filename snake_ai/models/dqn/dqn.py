import torch
from copy import deepcopy
from torch.optim import Optimizer
import itertools
from typing import List
from snake_ai.models.utils.replay_memory import ReplayMemory
from snake_ai.models.dqn.estimartor import Estimator

class DQN:
    def __init__(
        self,
        estimator: Estimator,
        buffer: ReplayMemory,
        optimizer: Optimizer,
        epsilon_schedule: itertools.chain,
        action_num: int = 5,
        gamma: float = 0.92,
        update_steps: int = 4,
        update_target_steps: int = 10,
        warmup_steps: int = 100,
    ):
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator)
        self._buffer = buffer
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = action_num
        self._gamma = gamma
        self._update_steps=update_steps
        self._update_target_steps=update_target_steps
        self._warmup_steps = warmup_steps
        self._step_cnt = 0
        assert warmup_steps > self._buffer._batch_size, (
            "You should have at least a batch in the ER.")
    
    def step(self, state: torch.Tensor):
        # implement an epsilon greedy policy using the
        # estimator and epsilon schedule attributes.

        # warning, you should make sure you are not including
        # this step into torch computation graph
        
        if self._step_cnt < self._warmup_steps:
            return torch.randint(self._action_num, (1,)).item()

        if next(self._epsilon) < torch.rand(1).item():
            with torch.no_grad():
                qvals = self._estimator(state)
                return qvals.argmax().item()
        else:
            return torch.randint(self._action_num, (1,)).item()
    
    def best_act(self, state: torch.Tensor):
        
        qvals = self._estimator(state)
        return qvals.argmax().item()


    def learn(self, state: torch.Tensor, action: int, reward: float, state_: torch.Tensor, done: bool):

        # add transition to the experience replay
        self._buffer.push((state, action, reward, state_, done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            # sample from experience replay and do an update
            batch = self._buffer.sample() 
            self._update(*batch)
        
        # update the target estimator
        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        states_: torch.Tensor,
        done: torch.Tensor
    ):
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)
        
        # compute Q(s, a) and max_a' Q(s', a')
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

        # compute target Q(s', a')
        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        # at this step you should check the target values
        # are looking about right :). You can use this code.
        # if rewards.squeeze().sum().item() > 0.0:
        #     print("R: ", rewards.squeeze())
        #     print("T: ", target_qsa.squeeze())
        #     print("D: ", done.squeeze())

        # compute the loss and average it over the entire batch
        loss = (qsa - target_qsa).pow(2).mean()

        # backprop and optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
