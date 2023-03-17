import gym
from .bn_env import BNEnv
from .pbn_env import PBNEnv
#from config import gamma, horizon, min_epsilon, n_episodes, n_epochs


class MasterBNSlavePBNEnv(gym.Env):

    def __init__(
            self,
            master_BN_data=[],
            master_logic_func_data=None,
            slave_PBN_data=[],
            slave_logic_func_data=None

    ):
        
        self.masterBNEnv = BNEnv(master_BN_data, master_logic_func_data)
        self.slavePBNEnv = PBNEnv(slave_PBN_data, slave_logic_func_data)

        #self.agent = PERAgent({
         #   "seed": 1234,
          #  "height": 50,
           # "gamma": gamma,
            
        #})

        
    def step(self):
        #Master BN does a step
        self.masterBNEnv.PBN.step()

        #Slave PBN does a step
        self.slavePBNEnv.PBN.step()