from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        map_size: int = 64,
        num_of_layers: int = 4,
        channels: List[int] = [8, 16, 32, 64],
        action_num: int = 5,
        hidden_embedding_size: int = 128,
        apply_pooling: bool = True,
        device: torch.device = torch.device("cpu")
    ):
        super(ActorCriticPolicy, self).__init__()

        assert num_of_layers == len(channels)

        self.action_num = action_num
        self.in_channels = in_channels
        self.num_of_layers = num_of_layers
        self.channels = channels
        self.hidden_embedding_size = hidden_embedding_size
        self.apply_pooling = apply_pooling

        self.convs = [
            nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = channels[0],
                kernel_size = 3,
                stride = 1,
                padding = "same",
                device=device
            )
        ]

        for i in range(1, len(channels)):
            self.convs.append(
                nn.Conv2d(
                    in_channels = channels[i - 1],
                    out_channels = channels[i],
                    kernel_size = 3,
                    stride = 1,
                    padding = "same",
                    device=device
                )
            )

        self.convs = nn.ModuleList(self.convs)
        self.pool = nn.MaxPool2d(2, 2)

        self.embed_size = self.compute_conv_encoder_output_shape(
            map_size = map_size
        )
        self.affine = nn.Linear(channels[-1] * self.embed_size * self.embed_size, self.hidden_embedding_size, device=device)
        self.policy = nn.Linear(self.hidden_embedding_size, self.action_num, device=device)
        self.value = nn.Linear(self.hidden_embedding_size, 1, device=device)
    
    def compute_conv_encoder_output_shape(
        self,
        map_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = 'same',
        pool_kernel_size: int = 2,
        pool_stride: int = 2
    ) -> int:

        shape = map_size

        for _ in range(self.num_of_layers):
            
            # If padding is set to "same" the output shape from a convolution is equal to the input shape.
            # Compute output shape from a convolution.
            if padding != 'same':
                shape = (shape + 2 * padding - 1 * (kernel_size - 1) - 1) // stride + 1
            
            # Compute output shape of a max pool layer.
            if self.apply_pooling is True:
                shape = (shape - 1 * (pool_kernel_size - 1) - 1) // pool_stride + 1
        
        return shape

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:

        internal_embedding = x

        # Embed the input through the Convolution Encoder.
        for i in range(self.num_of_layers):
            internal_embedding = self.convs[i](internal_embedding)
            internal_embedding = self.pool(internal_embedding) if self.apply_pooling else internal_embedding
            internal_embedding = F.relu(internal_embedding)

        # Linearize the embedding.
        internal_embedding = torch.reshape(internal_embedding, (-1, self.channels[-1] * self.embed_size * self.embed_size))
        # Pass the linearized embedding through an affine transformation (reduce the dimenstionality).
        internal_embedding = F.relu(self.affine(internal_embedding))

        # Compute outputs.
        pi = Categorical(F.softmax(self.policy(internal_embedding), dim=-1))
        value = self.value(internal_embedding)

        return pi, value