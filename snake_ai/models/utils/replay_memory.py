import torch
from collections import deque
import random

class ReplayMemory:
    def __init__(
        self,
        size: int = 1000,
        batch_size: int = 32,
        device: torch.device = torch.device('cpu')
    ):
        self._buffer = deque(maxlen=size)
        self._batch_size = batch_size
        self._device = device
    
    def push(self, transition):
        self._buffer.append(transition)
    
    def sample(self):
        """ Sample from self._buffer

            Should return a tuple of tensors of size: 
            (
                states:     N * (C*K) * H * W,  (torch.uint8)
                actions:    N * 1, (torch.int64)
                rewards:    N * 1, (torch.float32)
                states_:    N * (C*K) * H * W,  (torch.uint8)
                done:       N * 1, (torch.uint8)
            )

            where N is the batch_size, C is the number of channels = 3 and
            K is the number of stacked states.
        """
        # sample
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        # reshape, convert if needed, put on device (use torch.to(DEVICE))
        return (
            torch.cat(s, 0).to(self._device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self._device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self._device),
            torch.cat(s_, 0).to(self._device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self._device)
        )
    
    def __len__(self):
        return len(self._buffer)