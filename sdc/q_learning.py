import pickle
import random
import time
from collections.abc import Iterable

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.memory import Transition
from config import *


class QLearningAgent:
    def __init__(self, state_space, action_space):
        self.Q_table = np.zeros((state_space, action_space))
        self.EPSILON = None
        self.OMEGA = 0.85
        self._episode = 0
        self.training = False
        self.action_space = action_space

    def toggle_train(self, conf):
        self.training = True
        self.train_steps = conf["train_episodes"] * conf["train_epoch"]
        self.train_count = 0

        # Explore-exploit
        self.EPSILON = 1
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = conf["min_epsilon"]
        self.EPSILON_DECREMENT = (
            self.MAX_EPSILON - self.MIN_EPSILON
        ) / self.train_steps

    @property
    def a_ep(self):
        return 1 / ((self._episode + 1) ** self.OMEGA)

    def _state_to_idx(self, state):
        return int(
            "".join([str(x) for x in np.array(state, dtype=np.int8).tolist()]), 2
        )

    def get_action(self, state_idx):  # e-greedy
        if isinstance(state_idx, Iterable):  # Convert state vectors to indices
            state_idx = self._state_to_idx(state_idx)
        if self.training and self.EPSILON and random.uniform(0, 1) <= self.EPSILON:
            action = np.random.randint(self.action_space)
        else:
            action = self.Q_table[state_idx, :].argmax()
        return int(action)

    def feedback(self, transition):
        state_idx, encoded_action, interval, reward, next_state_idx, done = transition
        target_Q = reward + gamma * self.Q_table[next_state_idx, :].max()
        current_Q = self.Q_table[state_idx, encoded_action]
        loss = target_Q - current_Q
        self.Q_table[state_idx, encoded_action] = current_Q + self.a_ep * loss

    def update_params(self):
        self._episode += 1
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

    @property
    def extra_params(self):
        return {"E": self.EPSILON, "lr": self.a_ep}

    def save(self, env):
        with open(f"results/models/Q_learning/{env.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, env):
        with open(f"results/models/Q_learning/{env.name}.pkl", "rb") as f:
            agent = pickle.load(f)
        agent.training = False
        return agent

    def train(self, env, conf):
        self.toggle_train(conf)
        rewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)
        writer = SummaryWriter(f"runs/Q-Learning/{env.name}")

        for epoch in range(conf["train_epoch"]):
            start_time = time.time()
            actions_chosen = set()
            steps = 0

            for episode in range(conf["train_episodes"]):
                print(
                    f"Epoch {epoch + 1}/{n_epochs} "
                    + f"- Episode {episode + 1}/{n_episodes} "
                    + f"- Reward: {np.mean(rewards, axis=1)[epoch]:.4f}, "
                    + ", ".join(
                        [
                            f"- {name}: {value:.8f}"
                            for name, value in self.extra_params.items()
                        ]
                    )
                    + ", "
                    + f"Time elapsed: {int(time.time() - start_time)}s",
                    end="\r",
                )

                env.reset()
                state = env.render(mode="idx")
                episode_reward = 0

                for _ in range(horizon):
                    action = self.get_action(state)
                    actions_chosen.add(action)
                    steps += 1

                    next_state, reward, done, info = env.step(action)
                    next_state = info["observation_idx"]

                    self.feedback(
                        Transition(
                            state,
                            action,
                            info["interval"],
                            reward,
                            next_state,
                            done,
                        )
                    )

                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                self.update_params()
                rewards[epoch, episode] = episode_reward

            self.save(env)
            writer.add_scalar("epoch_reward", np.mean(rewards, axis=1)[epoch], epoch)
            writer.add_scalars(
                "action_stats",
                {
                    "actions_chosen": len(actions_chosen),
                    "n_interractions": steps / conf["train_episodes"],
                },
                epoch,
            )
            print(end="\n")
