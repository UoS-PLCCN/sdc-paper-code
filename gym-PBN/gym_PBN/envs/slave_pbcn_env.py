from .pbcn_env import PBCNEnv
from gym_PBN.types import REWARD, STATE, TERMINATED, GYM_STEP_RETURN
from typing import Tuple
import numpy as np
import itertools
import gym
from .common.slave_pbcn import SlavePBCN
from gym_PBN.utils import booleanize





class SlavePBCNEnv(PBCNEnv):

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
        self.PBN = SlavePBCN(PBN_data, logic_func_data)

    def _get_reward(self, master_BN_state) -> Tuple[REWARD, TERMINATED]:

        reward, done = 0, False

        if (self.PBN.state == master_BN_state).all():
            reward += 3
            done = True
        else:
            reward -= self.action_cost

        return reward, done
    
    def slave_step(self, masterBNPreviousState, master_BN_state, action: int = 0) -> GYM_STEP_RETURN:
        
        """Transition the environment by 1 step. Optionally perform an action.

        Args:
            action (int, optional): The action to perform (1-indexed node to flip). Defaults to 0, meaning no action.

        Raises:
            Exception: When the action is outside the action space.

        Returns:
            GYM_STEP_RETURN: The typical Gym environment 4-item Tuple.\
                 Consists of the resulting environment state, the associated reward, the termination status and additional info.
        """
        #if not self.action_space.contains(action):
            #raise Exception(f"Invalid action {action}, not in action space.")

        #if action != 0:  # Action 0 is taking no action.
            #action -= 1
            #self.PBN.flip(action)

        if type(action) is int:
            action = booleanize(action, self.action_space.n)

        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        self.PBN.apply_control(action)

        master_bn_slave_pbn_state = np.concatenate([masterBNPreviousState, self.PBN.control_state, self.PBN.state])
        self.PBN.step(master_bn_slave_pbn_state)

        observation = self.PBN.state
        reward, done = self._get_reward(master_BN_state)
        info = {"observation_idx": self._state_to_idx(observation)}

        return observation, reward, done, info
    
        
        
        
    def slave_step_test(self, masterBNPreviousState, master_BN_state, actions_len) -> GYM_STEP_RETURN:
        
        optimal_actions = []
        not_optimal_actions = []
        for action_test in range(actions_len):
            
            if type(action_test) is int:
                action = booleanize(action_test, self.action_space.n)

            if not self.action_space.contains(action):
                raise Exception(f"Invalid action {action}, not in action space.")

            self.PBN.apply_control(action)

            master_bn_slave_pbn_state = np.concatenate([masterBNPreviousState, self.PBN.control_state, self.PBN.state])
            state_test = self.PBN.step_test(master_bn_slave_pbn_state)

            if (state_test == master_BN_state).all():
                optimal_actions.append(action_test)
            else:
                not_optimal_actions.append(action_test)

        return optimal_actions, not_optimal_actions 
