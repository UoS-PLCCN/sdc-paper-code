from .pbn import PBN
import numpy as np
from gym_PBN.types import PBN_DATA, LOGIC_FUNC_DATA

class BN(PBN):

    def __init__(
        self, PBN_data: PBN_DATA = [], logic_func_data: LOGIC_FUNC_DATA = None
    ):
        
        super().__init__(PBN_data, logic_func_data)

    def step(self):
        """Perform a step of natural evolution."""
        self.state = np.array(
            [node.get_next_value_prob(self.state) for node in self.nodes], dtype=bool
        )