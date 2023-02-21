import itertools
import random
import sys

import numpy as np
from gym_PBN.utils import booleanize

import envs
from agent import PERAgent
from config import *
from q_learning import QLearningAgent
from utils import convert_state


class RandomAgent:
    def __init__(self) -> None:
        self.action_space = None
        self.options = False

    def get_action(self, state):
        action = self.action_space.sample()
        return (action, 1) if self.options else action

    @classmethod
    def load(cls, env):
        ret = cls()
        try:
            ret.action_space = env.primitive_action_space
            ret.options = True
        except AttributeError:
            ret.action_space = env.action_space
            ret.options = False
        return ret


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] not in ["drl", "ql", "random"]:
        raise ValueError(
            'Need to specify which agent to evaluate. Either "drl" or "ql" or "random"'
        )

    SDC_ENVS = [
        # envs.PBCN_4_1,
        # envs.PBCN_9_2,
        # envs.PBCN_16_3,
        # envs.PBCN_7_1_HIGH_PROB,
        # envs.PBCN_28_3,
        envs.PBCN_9_4_HIGH_PROB,
        # envs.PBCN_7_1_HIGH_PROB_15,
        # envs.PBCN_28_3_5,
        # envs.PBCN_28_3_15,
        # envs.PBCN_7_1_HIGH_PROB_STC_10,
    ]

    STC_ENVS = [
        # envs.PBCN_4_1_STC,
        # envs.PBCN_7_1_HIGH_PROB_STC,
        # envs.PBCN_28_3_STC,
        envs.PBCN_9_4_HIGH_PROB,
        envs.PBCN_9_4_HIGH_PROB_STC,
    ]

    SHORTEST_PATH_ENVS = [
        # envs.PBCN_4_1_NORMAL,
        # envs.PBCN_7_1_HIGH_PROB_NORMAL,
        # envs.PBCN_28_3_NORMAL,
        envs.PBCN_9_4_HIGH_PROB_NORMAL,
    ]

    def _eval(env, uses_options=True):
        print(f"Evaluating a {sys.argv[1].upper()} Agent on {env.name}.")
        agent = {"drl": PERAgent, "ql": QLearningAgent, "random": RandomAgent}[
            sys.argv[1]
        ].load(env)

        start_states = map(
            tuple, itertools.product([0, 1], repeat=env.observation_space.n)
        )

        i = 0
        actions, wins, steps = 0, 0, 0
        for state in start_states:
            if state not in env.target:
                i += 1
                env.reset()
                ep_state = env.set_state(state)

                for _ in range(horizon):
                    ep_state_float = convert_state(ep_state)
                    encoded_action = agent.get_action(ep_state_float)
                    actions += 1
                    ep_state, _, done, info = env.step(encoded_action)
                    if uses_options:
                        steps += info["interval"]
                    else:
                        steps += 1

                    if done:
                        wins += 1
                        break

        print(f"Winrate: {(wins / i) * 100}%")
        print(f"Average amount of interractions: {actions / i}")
        print(f"Average number of time steps until victory: {steps / i}")

    def _eval_sampled(env, uses_options=True):
        print(f"Evaluating a {sys.argv[1].upper()} Agent on {env.name}.")
        agent = {"drl": PERAgent, "ql": QLearningAgent, "random": RandomAgent}[
            sys.argv[1]
        ].load(env)

        N_SAMPLES = 200_000
        used_states = set()

        actions, wins, steps = 0, 0, 0
        for state in range(N_SAMPLES):
            print(f"{state + 1}/{N_SAMPLES}", end="\r")
            state_int = random.randint(0, 2 ** env.observation_space.n - 1)
            while (
                state_int in used_states
                or tuple(booleanize(state_int, env.observation_space.n)) in env.target
            ):  # Re-roll
                state_int = random.randint(0, 2 ** env.observation_space.n - 1)
            used_states.add(state_int)

            env.reset()
            ep_state = env.set_state(booleanize(state_int, env.observation_space.n))

            for _ in range(horizon):
                ep_state_float = convert_state(ep_state)
                encoded_action = agent.get_action(ep_state_float)
                actions += 1
                ep_state, _, done, info = env.step(encoded_action)
                if uses_options:
                    steps += info["interval"]
                else:
                    steps += 1

                if done:
                    wins += 1
                    break
        print(end="\n")

        print(f"Winrate: {(wins / N_SAMPLES) * 100}%")
        print(f"Average amount of interractions per time step: {actions / N_SAMPLES}")
        print(f"Average number of time steps until victory: {steps / N_SAMPLES}")

    uses_options = True
    for env in STC_ENVS:
        if env.observation_space.n > 16:
            _eval_sampled(env, uses_options)
        else:
            _eval(env, uses_options)
        print(end="\n")
