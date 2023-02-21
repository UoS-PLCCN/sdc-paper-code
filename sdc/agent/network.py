"""
network.py - This module holds the Agent's DQN.
"""
from torch import nn
from torch.nn import functional as F

from typing import List
from numbers import Number


class DQN(nn.Module):
    """Neural Network used as the DQN."""
    def __init__(self, input_size: int, output_size: int, height: int):
        """Neural network to approximate Q values.

        Args:
            input size (int): number of nodes of the input layer. The size of the PBN.
            output_size (int): number of nodes in the output layer. The number of actions. Size + 1
            height (int): number of nodes in the hidden layer of the NN. Arbitrary.
        """
        super().__init__()
        self.input = input_size
        self.fc1 = nn.Linear(input_size, height,  bias=True)
        self.fc2 = nn.Linear(height, height, bias=True)
        self.fc3 = nn.Linear(height, output_size, bias=True)

    def forward(self, x: List[Number]) -> List[float]:
        """A forward-pass of the neural network.

        Args:
            x (List[Number]): Network input. The PBN state in this case.

        Returns:
            List[float]: The network output. Value at index A is the expected cumulative reward if action A is taken.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
