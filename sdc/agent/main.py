"""
main.py - This module holds the actual Agent.
"""
import pickle
import random
import time
from typing import List

import numpy as np
import torch
from config import gamma, horizon
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import convert_state

from .consts import DEVICE
from .memory import ExperienceReplay, PrioritisedER, Transition
from .network import DQN
from .types import Minibatch, PERMinibatch

State = List[float]


class Agent:
    """The agent of the RL algorithm. Houses the DQN, ER, etc."""

    def __init__(self, config):
        """The agent of the RL algorithm. Houses the DQN, ER, etc.

        Attributes:
            input_size (int): DQN input size. Number of genes in the PBN for this case.
            device (torch.device): The devices that would be used in the DDQN.
            gamma (float): discount factor.
            train (bool): True if training, otherwise False.
            actions (list): The list containing all possible actions.
            controller (DQN): The primary DQN. The Controll network in this case.
            target (DQN): The Target DQN.

        Args:
            config (AgentConfig): the config lol
        """
        torch.manual_seed(config["seed"])
        self.device = DEVICE

        # The size of the PBN
        self.input_size = config["input_size"]
        self.actions = list(range(config["output_size"]))

        # Networks
        self.controller = DQN(
            config["input_size"], config["output_size"], config["height"]
        ).to(self.device)
        self.target = DQN(
            config["input_size"], config["output_size"], config["height"]
        ).to(self.device)
        self.target.load_state_dict(self.controller.state_dict())

        # Reinforcement learning parameters
        self.gamma = config["gamma"]

        # State
        self.training = False

    def load_model(self, model: DQN):
        """Load a saved model.

        Args:
            model (DQN): the saved model to load.
        """
        self.controller = model.to(self.device)
        self.target = model.to(self.device)
        self.target.load_state_dict(self.controller.state_dict())

    def _get_learned_action(self, state: State) -> int:
        with torch.no_grad():
            q_vals = self.controller(
                torch.tensor(state, device=self.device, dtype=torch.float)
            )
            action = q_vals.max(0)[1].view(1, 1).item()
        return action

    def get_action(self, state: State) -> int:
        """Receive current state. Run it through DQN. If training, use epsilon greedy policy.
        Else, return the action which yields highest Q value.

        Args:
            state (State): State :)

        Returns:
            int: Integer representing the action which grants highest cumulative reward.
        """
        if self.training and random.uniform(0, 1) <= self.EPSILON:
            return random.choice(self.actions)
        else:
            return self._get_learned_action(state)

    def get_controller(self):
        # TODO Is this *really* necessary?
        return self.controller

    def _fetch_minibatch(self) -> Minibatch:
        """Fetch a minibatch from the replay memory and load it into the chosen device.

        Returns:
            Minibatch: a minibatch.
        """
        # Fetch data
        experiences = self.replay_memory.sample(self.batch_size)
        (
            state_batch,
            action_batch,
            interval_batch,
            reward_batch,
            next_state_batch,
            dones,
        ) = zip(*experiences)

        # Load to device
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        action_batch = torch.tensor(
            action_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        interval_batch = torch.tensor(
            interval_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)

        return (
            state_batch,
            action_batch,
            interval_batch,
            reward_batch,
            next_state_batch,
            dones,
        )

    def _get_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        intervals: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Get huber loss based on a batch of experiences.

        Args:
            states (torch.Tensor): the batch of states.
            actions (torch.Tensor): the batch of agent actions.
            intervals (torch.Tensor): the batch of intervals elapsed as a result of the action.
            rewards (torch.Tensor): the batch of rewards received.
            next_states (torch.Tensor): the batch of the resulting states.
            dones (torch.Tensor): the batch of done flags.
            reduction (str, optional): the reduction to use on the loss.

        Returns:
            torch.Tensor: the huber loss as a tensor.
        """
        # Calculate predicted actions
        with torch.no_grad():
            vals = self.controller(next_states)  # TODO Wait shouldn't this be target?
            action_prime = vals.max(1)[1].unsqueeze(1)

        controller_Q = self.controller(states).gather(1, actions)  # VVVVV HACK
        target_Q = rewards + (1 - dones) * torch.pow(
            self.gamma, intervals
        ) * self.target(next_states).gather(1, action_prime)
        return F.smooth_l1_loss(controller_Q, target_Q, reduction=reduction)

    def _back_propagate(self, loss: torch.Tensor):
        """Do a step of back propagation based on a loss vector.

        Args:
            loss (torch.Tensor): the loss vector as a tensor.
        """
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def learn(self):
        """Sample a minibatch of experiences, do a step of back propagation."""
        if len(self.replay_memory) >= self.batch_size:
            loss = self._get_loss(*self._fetch_minibatch())
            self._back_propagate(loss)
        self.train_count += 1

        # Oh yeah after every TARGET_UPDATE steps the target network gets updated.
        if self.train_count % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.controller.state_dict())

    def feedback(self, transition: Transition, learn: bool = True):
        """Save an experience tuple, do a step of back propagation.

        Args:
            transition (Transition): Transition :)
            learn (bool, optional): Whether or not to do a step of back propagation.
        """
        self.replay_memory.store(transition)
        if learn:
            self.learn()

    def update_params(self):
        """Update internal parameters such as exploration rate."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

    @property
    def extra_params(self):
        return {"E": self.EPSILON}

    def toggle_train(self, conf):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        self.training = True
        self.train_steps = conf["train_episodes"] * conf["train_epoch"]
        self.train_count = 0

        self.optimizer = optim.RMSprop(self.controller.parameters())

        # Memory
        self.batch_size = conf["batch_size"]
        self.replay_memory = ExperienceReplay(conf["memory_size"])

        # Explore-exploit
        self.EPSILON = 1
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = conf["min_epsilon"]
        self.EPSILON_DECREMENT = (
            self.MAX_EPSILON - self.MIN_EPSILON
        ) / self.train_steps

        self.TARGET_UPDATE = conf["target_update"]

    def save(self, env):
        with open(f"results/models/DRL/{env.name}.pkl", "wb") as f:
            pickle.dump(self.controller, f)

    @classmethod
    def load(cls, env):
        agent = cls(
            {
                "seed": 1234,
                "height": 50,
                "gamma": gamma,
                "train_time_horizon": horizon,
                "input_size": env.observation_space.n,
                "output_size": env.discrete_action_space.n,
            }
        )

        with open(f"results/models/DRL/{env.name}.pkl", "rb") as f:
            agent.load_model(pickle.load(f))
            agent.training = False

        return agent

    def train(self, env, conf):
        print(f"Training using {DEVICE}")
        self.toggle_train(conf)
        writer = SummaryWriter(f"runs/DRL/{env.name}")
        rewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)

        for epoch in range(conf["train_epoch"]):
            start_time = time.time()
            actions_chosen = set()
            steps = 0

            for episode in range(conf["train_episodes"]):
                print(
                    f"Epoch {epoch + 1}/{conf['train_epoch']} "
                    + f"- Episode {episode + 1}/{conf['train_episodes']} "
                    + f"- Reward: {np.mean(rewards, axis=1)[epoch]:.4f}, "
                    + ", ".join(
                        [
                            f"- {name}: {value:.4f}"
                            for name, value in self.extra_params.items()
                        ]
                    )
                    + ", "
                    + f"Time elapsed: {int(time.time() - start_time)}s",
                    end="\r",
                )

                env.reset()
                state = env.render(mode="float")
                episode_reward = 0

                for _ in range(horizon):
                    action = self.get_action(state)
                    actions_chosen.add(action)
                    steps += 1

                    next_state, reward, done, info = env.step(action)
                    next_state = convert_state(next_state)

                    if (
                        conf["mode"] == "sampled-data"
                        or conf["mode"] == "self-triggering"
                    ):
                        interval = info["interval"]
                    else:
                        interval = 1

                    self.feedback(
                        Transition(
                            state,
                            action,
                            interval,
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
            for name, param in self.controller.named_parameters():
                writer.add_histogram(name, param, epoch)
            print(end="\n")


class PERAgent(Agent):
    """Agent using Prioritized Experience Replay."""

    def _fetch_minibatch(self) -> PERMinibatch:
        """Fetch a minibatch from the replay memory and load it into the chosen device.

        Returns:
            PERMinibatch: a minibatch.
        """
        # Fetch data
        experiences, indices, weights = self.replay_memory.sample(
            self.batch_size, self.BETA
        )
        (
            state_batch,
            action_batch,
            interval_batch,
            reward_batch,
            next_state_batch,
            dones,
        ) = zip(*experiences)

        # Load to device
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        action_batch = torch.tensor(
            action_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        interval_batch = torch.tensor(
            interval_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)
        weights = (
            torch.tensor(weights, device=self.device, dtype=torch.float)
            .squeeze()
            .unsqueeze(1)
        )

        return (
            state_batch,
            action_batch,
            interval_batch,
            reward_batch,
            next_state_batch,
            dones,
            indices,
            weights,
        )

    def learn(self):
        """Sample a minibatch of experiences, do a step of back propagation."""
        if len(self.replay_memory) >= self.batch_size:
            (
                state_batch,
                action_batch,
                interval_batch,
                reward_batch,
                next_state_batch,
                dones,
                indices,
                weights,
            ) = self._fetch_minibatch()
            loss = self._get_loss(
                state_batch,
                action_batch,
                interval_batch,
                reward_batch,
                next_state_batch,
                dones,
                reduction="none",
            )
            loss *= weights

            # Update priorities in the PER buffer
            priorities = loss + self.REPLAY_CONSTANT
            # TODO Wait don't these have to be positive? Maybe need to abs()?
            self.replay_memory.update_priorities(
                indices, priorities.data.detach().squeeze().cpu().numpy().tolist()
            )

            # Back propagation
            loss = loss.mean()
            self._back_propagate(loss)
        self.train_count += 1

        # Oh yeah after every TARGET_UPDATE steps the target network gets updated.
        if self.train_count % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.controller.state_dict())

    def update_params(self):
        """Update internal parameters such as exploration rate, or the beta exponent."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)
        self.BETA = min(self.BETA + self.BETA_INCREMENT_CONSTANT, 1)

    @property
    def extra_params(self):
        return {"E": self.EPSILON, "B": self.BETA}

    def toggle_train(self, conf):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        super().toggle_train(conf)

        # PER
        self.REPLAY_CONSTANT = 1e-5
        self.BETA = 0.4
        self.BETA_INCREMENT_CONSTANT = self.BETA / (0.75 * self.train_steps)
        self.replay_memory = PrioritisedER(conf["memory_size"])
