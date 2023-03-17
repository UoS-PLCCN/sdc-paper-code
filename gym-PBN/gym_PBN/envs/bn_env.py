from .pbn_env import PBNEnv
from .common.bn import BN
import numpy as np
from typing import List, Set, Tuple, Union
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED

class BNEnv(PBNEnv):

    def __init__(
        self,
        PBN_data=[],
        logic_func_data=None,
        name: str = None,
        goal_config: dict = None,
        reward_config: dict = None,
    ):
        super().__init__(PBN_data, logic_func_data, name, goal_config, reward_config)
        
    
    def setPBN(self,PBN_data, logic_func_data):
        self.PBN = BN(PBN_data, logic_func_data)

    def stepMaster(self):
        self.PBN.step()
        observation = self.PBN.state
        return observation

