from  master_bn_slave_pbn import MasterBNSlavePBN
import envs
from config import min_epsilon, n_episodes, n_epochs
import pickle


master_BN_env = envs.MASTER_BN
slave_PBN_env = envs.SLAVE_PBCN

master_BN_slave_PBN_env = MasterBNSlavePBN(master_BN_env, slave_PBN_env)

with open("runs/DRL/masterSlaveTestNetworks/masterslave", "rb") as r:
    load_weights = pickle.load(r)

master_BN_slave_PBN_env.slaveAgent.controller = load_weights

master_BN_slave_PBN_env.test()