from  master_bn_slave_pbn import MasterBNSlavePBN
import envs
from config import min_epsilon, n_episodes, n_epochs


master_BN_env = envs.MASTER_BN
slave_PBN_env = envs.SLAVE_PBCN

master_BN_slave_PBN_env = MasterBNSlavePBN(master_BN_env, slave_PBN_env)

master_BN_slave_PBN_env.train_Only_Slave_RL_Agent(
    {
        "train_episodes": n_episodes,
        "train_epoch": n_epochs,
        "batch_size": 128,
        "memory_size": 5120,
        "min_epsilon": min_epsilon,
        "decay_rate": 0.05,
        "target_update": 1000,
        "mode": "normal",
    }
)

#master_BN_slave_PBN_env.test()

