import itertools

import numpy as np
import pandas as pd

import envs
from agent import PERAgent
from config import horizon
from utils import convert_state


def _state_to_idx(state):
    return int("".join([str(x) for x in np.array(state, dtype=np.int8).tolist()]), 2)


N_iterations = 1_000
MAX_TIME_STEPS = 50

env = envs.PBCN_9_4_HIGH_PROB

agent = PERAgent.load(env)

start_states = map(tuple, itertools.product([0, 1], repeat=env.observation_space.n))

node_value_time_series = np.zeros(
    (2 ** env.observation_space.n, MAX_TIME_STEPS, env.observation_space.n)
)
done_steps_time_series = np.zeros((2 ** env.observation_space.n))
win_rate = 0

for state in start_states:
    if state in env.target:
        continue

    iteration_values = np.zeros((N_iterations, MAX_TIME_STEPS, env.observation_space.n))
    done_steps = np.zeros(N_iterations)

    for iteration in range(N_iterations):
        env.reset()
        env.set_state(state)
        ep_state = env.render(mode="float")
        iteration_values[iteration, 0, :] = ep_state

        t_k = 1

        for t in range(horizon):
            encoded_action = agent.get_action(ep_state)
            _, _, done, info = env.step(encoded_action)
            ep_state = env.render(mode="float")

            trajectory = info["states"]
            timesteps_elapsed = info["interval"]

            iteration_values[iteration, t_k : (t_k + timesteps_elapsed), :] = np.array(
                [np.array(convert_state(_state)) for _state in trajectory]
            )

            t_k += timesteps_elapsed

            if done:
                done_steps[iteration] = t_k
                win_rate += 1
                break

    node_value_time_series[_state_to_idx(state), :, :] = iteration_values.mean(axis=0)
    done_steps_time_series[_state_to_idx(state)] = done_steps.mean()

df = pd.DataFrame(
    node_value_time_series.mean(axis=0),
    columns=[f"Node {x + 1}" for x in range(env.observation_space.n)],
)
df.to_csv("results/eval4.csv")

print(f"Done step average: {done_steps_time_series.mean()}")
print(f"Winrate: {win_rate/(N_iterations * (2 ** env.observation_space.n))}")
