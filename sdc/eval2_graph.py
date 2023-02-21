import itertools

from numpy import e

from agent import PERAgent
from config import horizon
from envs import PBCN_4_1, PBCN_4_1_NORMAL

env = PBCN_4_1_NORMAL


agent = PERAgent.load(env)

start_states = map(tuple, itertools.product([0, 1], repeat=env.observation_space.n))

action_monitoring = []


def state_str(state) -> str:
    return "".join([str(int(i)) for i in state])


# state_map = []
# for state in start_states:
#     if state not in env.target:
#         env.reset()
#         end_states = []

#         for action in range(env.discrete_action_space.n):
#             env.reset()
#             env.set_state(state)
#             _state, _, _, _ = env.step(action)
#             end_states.append(_state)

#         state_map.append((state, end_states))

# for state, end_states in state_map:
#     print(f"{state_str(state)} - {[state_str(_s) for _s in end_states]}")


i = 0
n_actions, wins, steps = 0, 0, 0
for state in start_states:
    if state not in env.target:
        i += 1
        env.reset()
        env.set_state(state)
        actions = []
        states = []

        for _ in range(horizon):
            ep_state_float = env.render(mode="float")
            encoded_action = agent.get_action(ep_state_float)
            n_actions += 1
            ep_state, _, done, info = env.step(encoded_action)
            try:
                steps += info["interval"]
            except:
                steps += 1

            try:
                actions.append((info["control_action"], info["interval"]))
            except:
                actions.append((info["control_action"], 1))

            try:
                states.append(info["states"])
            except:
                states.append(ep_state)

            if done:
                wins += 1
                break

        action_monitoring.append((state, actions, states))

for state, actions, states in action_monitoring:
    print(
        f"{state_str(state)} - {', '.join([f'{state_str(action)} for {interval} [{state_str(_state)}]' for (action, interval), _state in zip(actions, states)])}"
    )
