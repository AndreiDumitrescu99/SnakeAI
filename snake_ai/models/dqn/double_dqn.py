import torch
from snake_ai.models.dqn.dqn import DQN
from snake_ai.models.utils.replay_memory import ReplayMemory

class DoubleDQN(DQN):
    def _update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, states_: torch.Tensor, done: torch.Tensor):
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_)
        q_values = self._estimator(states)
        
        # compute Q(s, a) and TODO:
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)

        # compute target Q(s', a')
        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        # compute the loss and average it over the entire batch
        loss = (qsa - target_qsa).pow(2).mean()

        # backprop and optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()