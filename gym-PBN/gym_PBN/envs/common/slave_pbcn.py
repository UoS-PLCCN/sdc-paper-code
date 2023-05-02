from .pbcn import PBCN
import numpy as np
import itertools
from gym_PBN.types import PBN_DATA, LOGIC_FUNC_DATA
from gym_PBN.utils.converters import slave_logic_funcs_to_PBN_data
from .node import Node



class SlavePBCN(PBCN):

    def __init__(
        self, PBN_data: PBN_DATA = [], logic_func_data: LOGIC_FUNC_DATA = None
    ):
        
        super().__init__(PBN_data, logic_func_data)

    def _logic_funcs_to_pbn_data(self, logic_func_data: LOGIC_FUNC_DATA):
        return slave_logic_funcs_to_PBN_data(*logic_func_data)
    
    def step(self, master_bn_slave_pbn_state):
        """Perform a step of natural evolution."""
        self.state = np.array(
            [node.compute_next_value(master_bn_slave_pbn_state) for node in self.nodes], dtype=bool
        )
