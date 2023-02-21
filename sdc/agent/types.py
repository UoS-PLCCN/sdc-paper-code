from typing import List, Tuple

import torch

PERMinibatch = Tuple[
    # States, Actions, Intervals, Rewards, Next States, Dones
    torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
    # Indices, Weights
    List[int], torch.FloatTensor
]

Minibatch = Tuple[
    # States, Actions, Intervals, Rewards, Next States, Dones
    torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
]
