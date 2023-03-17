import envs
from agent import PERAgent
from config import gamma, horizon, min_epsilon, n_episodes, n_epochs
from q_learning import QLearningAgent

SDC_ENVS = [
    # envs.PBCN_4_1,
    # envs.PBCN_9_2,
    # envs.PBCN_16_3,
    # envs.PBCN_28_3,
    # envs.PBCN_7_1_HIGH_PROB,
    #envs.PBCN_9_4_HIGH_PROB,
    # envs.PBCN_7_1_HIGH_PROB_5,
    # envs.PBCN_7_1_HIGH_PROB_15,
    # envs.PBCN_28_3,
    # envs.PBCN_28_3_15,
]

STC_ENVS = [
    #envs.MASTER_BN
    # envs.PBCN_4_1_STC,
    # envs.PBCN_7_1_HIGH_PROB_STC,
    # envs.PBCN_28_3_STC,
    envs.PBCN_9_4_HIGH_PROB_STC
]

SHORTEST_PATH_ENVS = [
    # envs.PBCN_4_1_NORMAL,
    # envs.PBCN_7_1_HIGH_PROB_NORMAL,
    # envs.PBCN_28_3_NORMAL,
    # envs.PBCN_9_4_HIGH_PROB_NORMAL,
]


def train_q_learning():
    for env in SDC_ENVS:
        agent = QLearningAgent(
            2 ** env.observation_space.n, env.discrete_action_space.n
        )
        print(f"Training Q Learning on {env.name}")
        print(
            f"Input dim: {2 ** env.observation_space.n}, Output dim: {env.discrete_action_space.n}"
        )
        agent.train(
            env,
            {
                "train_episodes": n_episodes,
                "train_epoch": n_epochs,
                "min_epsilon": min_epsilon,
            },
        )


def train_drl_agent():
    for env in STC_ENVS:
        agent = PERAgent(
            {
                "seed": 1234,
                "height": 50,
                "gamma": gamma,
                "train_time_horizon": horizon,
                "input_size": env.observation_space.n,
                "output_size": env.discrete_action_space.n,
            }
        )
        print(f"Training a DDQN+PER Agent on {env.name}")
        #print(
            #f"Input dim: {env.observation_space.n}, Output dim: {env.discrete_action_space.n}"
        #)
        agent.train(
            env,
            {
                "train_episodes": n_episodes,
                "train_epoch": n_epochs,
                "batch_size": 128,
                "memory_size": 5120,
                "min_epsilon": min_epsilon,
                "decay_rate": 0.05,
                "target_update": 1000,
                "mode": "a",
            },
        )


def train_shortest_path_agent():
    for env in SHORTEST_PATH_ENVS:
        agent = PERAgent(
            {
                "seed": 1234,
                "height": 50,
                "gamma": gamma,
                "horizon": horizon,
                "input_size": env.observation_space.n,
                "output_size": env.discrete_action_space.n,
            }
        )
        print(f"Training a DDQN+PER Agent on {env.name}")
        print(
            f"Input dim: {env.observation_space.n}, Output dim: {env.discrete_action_space.n}"
        )
        agent.train(
            env,
            {
                "train_episodes": n_episodes,
                "train_epoch": n_epochs,
                "batch_size": 128,
                "memory_size": 5120,
                "min_epsilon": min_epsilon,
                "decay_rate": 0.05,
                "target_update": 1000,
                "mode": "normal",
            },
        )


if __name__ == "__main__":
    train_drl_agent()
