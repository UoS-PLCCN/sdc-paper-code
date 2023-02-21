from agent.main import PERAgent
from config import *

# Deep Q Learning
state_space = 2 ** env.observation_space.n

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
