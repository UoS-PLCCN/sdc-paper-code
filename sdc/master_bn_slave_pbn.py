import envs
import sys
from config import gamma, horizon, min_epsilon, n_episodes, n_epochs
from agent import PERAgent
from agent.memory import Transition
import numpy as np
import time
from utils import convert_state
from agent.consts import DEVICE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import itertools
import pickle



class MasterBNSlavePBN:

    def __init__(self, masterBN, slavePBN):
        
        self.masterBN = masterBN
        self.slavePBN = slavePBN

        self.masterAgent = PERAgent(
            {
                "seed": 1234,
                "height": 50,
                "gamma": gamma,
                "train_time_horizon": horizon,
                "input_size": masterBN.observation_space.n,
                "output_size": masterBN.action_space.n
            }
        )

        input_s = 2 * masterBN.observation_space.n

        self.slaveAgent = PERAgent(
            {
                "seed": 1234,
                "height": 50,
                "gamma": gamma,
                "train_time_horizon": horizon,
                "input_size": input_s,
                "output_size": slavePBN.discrete_action_space.n
                #"output_size": slavePBN.action_space.n
            }
        )

    """
    def trainMasterBN(self, conf):
        print(f"Training using {DEVICE}")
        self.masterAgent.toggle_train(conf)
        writer = SummaryWriter(f"runs/DRL/masterBN")
        masterBNRewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)
        for epoch in tqdm(range(conf["train_epoch"])):
            start_time = time.time()
            master_BN_actions_chosen = set()
            steps = 0

            for episode in tqdm(range(conf["train_episodes"])):
                print(
                    f"Epoch {epoch + 1}/{conf['train_epoch']} "
                    + f"- Episode {episode + 1}/{conf['train_episodes']} "
                    + f"- MasterBNReward: {np.mean(masterBNRewards, axis=1)[epoch]:.4f}, "
                    + ", ".join(
                        [
                            f"-MasterBn {name}: {value:.4f}"
                            for name, value in self.masterAgent.extra_params.items()
                        ]
                    )
                    + ", "
                    + f"Time elapsed: {int(time.time() - start_time)}s",
                    end="\r",
                )

                self.masterBN.reset()
                masterBNstate = self.masterBN.render(mode="float")
                master_BN_episode_reward = 0

                for _ in tqdm(range(horizon)):
                    steps += 1
                    interval = 1

                    master_BN_action = self.masterAgent.get_action(masterBNstate)
                    master_BN_actions_chosen.add(master_BN_action)
                    master_BN_next_state_bool, master_BN_reward, master_BN_done, master_BN_info = self.masterBN.step(master_BN_action)
                    master_BN_next_state = convert_state(master_BN_next_state_bool)
                    self.masterAgent.feedback(
                        Transition(
                            masterBNstate,
                            master_BN_action,
                            interval,
                            master_BN_reward,
                            master_BN_next_state,
                            master_BN_done,
                        )
                    )
                    masterBNstate = master_BN_next_state
                    master_BN_episode_reward  += master_BN_reward

                    if master_BN_done:
                        break

                self.masterAgent.update_params()

                masterBNRewards[epoch, episode] = master_BN_episode_reward

            #self.masterAgent.save(self.masterBN)
            writer.add_scalar("master_BN_epoch_reward", np.mean(masterBNRewards, axis=1)[epoch], epoch)
            writer.add_scalars(
                "master_BN_action_stats",
                {
                    "master_BN_actions_chosen": len(master_BN_actions_chosen),
                    "master_BN_n_interractions": steps / conf["train_episodes"],
                },
                epoch,
            )
    """


    """   
    def train(self, conf):
        self.trainMasterBN(conf)
        print(f"Training using {DEVICE}")
        self.masterAgent.training = False
        self.slaveAgent.toggle_train(conf)
        
        writer = SummaryWriter(f"runs/DRL/slavePBN")

        masterBNRewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)
        slavePBNRewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)

        for epoch in range(conf["train_epoch"]):
            start_time = time.time()
            master_BN_actions_chosen = set()
            slave_PBN_actions_chosen = set()
            steps = 0

            for episode in tqdm(range(conf["train_episodes"])):
                print(
                    f"Epoch {epoch + 1}/{conf['train_epoch']} "
                    + f"- Episode {episode + 1}/{conf['train_episodes']} "
                    + f"- MasterBNReward: {np.mean(masterBNRewards, axis=1)[epoch]:.4f}, "
                    + f"- SlavePBNReward: {np.mean(slavePBNRewards, axis=1)[epoch]:.4f}, "
                    + ", ".join(
                        [
                            f"-SlavePBN {name}: {value:.4f}"
                            for name, value in self.slaveAgent.extra_params.items()
                        ]
                    )
                    + ", "
                    + f"Time elapsed: {int(time.time() - start_time)}s",
                    end="\r",
                )

                self.masterBN.reset()
                masterBNstate = self.masterBN.render(mode="float")
                masterBNPreviousState = self.masterBN.PBN.state
                master_BN_episode_reward = 0


                self.slavePBN.reset()
                slavePBNstate = self.slavePBN.render(mode="float")
                slave_PBN_episode_reward = 0

                for _ in range(horizon):
                    steps += 1
                    interval = 1

                    master_BN_action = self.masterAgent.get_action(masterBNstate)
                    master_BN_actions_chosen.add(master_BN_action)
                    master_BN_next_state_bool, master_BN_reward, master_BN_done, master_BN_info = self.masterBN.step(master_BN_action)
                    master_BN_next_state = convert_state(master_BN_next_state_bool)

                    masterBNstate = master_BN_next_state
                    master_BN_episode_reward  += master_BN_reward

                    slave_PBN_action = self.slaveAgent.get_action(slavePBNstate)
                    slave_PBN_actions_chosen.add(slave_PBN_action)
                    slave_PBN_next_state, slave_PBN_reward, slave_PBN_done, slave_PBN_info = self.slavePBN.slave_step(masterBNPreviousState, master_BN_next_state_bool, slave_PBN_action)
                    slave_PBN_next_state = convert_state(slave_PBN_next_state)
                    self.slaveAgent.feedback(
                        Transition(
                            slavePBNstate,
                            slave_PBN_action,
                            interval,
                            slave_PBN_reward,
                            slave_PBN_next_state,
                            slave_PBN_done,
                        )
                    )
                    slavePBNstate = slave_PBN_next_state
                    slave_PBN_episode_reward += slave_PBN_reward

                    if master_BN_done:
                        break

                self.slaveAgent.update_params()

                masterBNRewards[epoch, episode] = master_BN_episode_reward
                slavePBNRewards[epoch, episode] = slave_PBN_episode_reward

            #self.masterAgent.save(self.masterBN)
            #self.slaveAgent.save(self.slavePBN)
            writer.add_scalar("slave_PBN_epoch_reward", np.mean(slavePBNRewards, axis=1)[epoch], epoch)
            writer.add_scalars(
                "slave_PBN_action_stats",
                {
                    "slave_PBN_actions_chosen": len(slave_PBN_actions_chosen),
                    "slave_PBN_n_interractions": steps / conf["train_episodes"],
                },
                epoch,
            )
    """

    def test(self):
        self.slaveAgent.training = False
        logging.basicConfig(filename='myapp.log', level=logging.DEBUG)

        correctAllEpisodes = 0
        slaveFollowedMasterAllEpisodes = []
        slaveFollowedMasterAllEpisodesIgnoreFirstSteps = []
        for episode in tqdm(range(50)):

            #master_BN_state_history = []
            #slave_PBN_state_history = []

            correctEpisode = 0
            self.masterBN.reset()
            masterBNPreviousState = self.masterBN.PBN.state
            masterBNPreviousStateFloat = self.masterBN.render(mode="float")


            self.slavePBN.reset()
            slavePBNstate = self.slavePBN.render(mode="float")
            masterSlaveStateFloat = masterBNPreviousStateFloat + slavePBNstate

            #graphs
            #master_BN_state_int = self.convert_state_int(masterBNPreviousStateFloat)
            #slave_PBN_state_int = self.convert_state_int(slavePBNstate)

            #master_BN_binary_state = self.convert_binary_to_decimal(master_BN_state_int)
            #slave_PBN_binary_state = self.convert_binary_to_decimal(slave_PBN_state_int)

            #master_BN_state_history.append(master_BN_state_int)
            #slave_PBN_state_history.append(slave_PBN_state_int)

            logging.debug(" ")
            logging.debug(f"Start of episode {episode + 1}" + f"masterBNOriginalState {masterBNPreviousState}" + f"slavePBNOriginalstate {slavePBNstate}")
            logging.debug(f"Episode {episode + 1}" + f"Before Step Master state {masterBNPreviousState}" + f"Before step Slave state {slavePBNstate}")

            firstMatch = False
            numberOfStepsTakenForMatch = 0

            for h in range(150):

                master_BN_next_state_bool = self.masterBN.stepMaster()

                slave_PBN_action = self.slaveAgent.get_action(masterSlaveStateFloat)
                logging.debug(f"Episode {episode + 1}" + f"Action chosen {slave_PBN_action}")
                slave_PBN_next_state, slave_PBN_reward, slave_PBN_done, slave_PBN_info = self.slavePBN.slave_step(masterBNPreviousState, master_BN_next_state_bool, slave_PBN_action)
                slavePBNstate = convert_state(slave_PBN_next_state)
                logging.debug(f"Episode {episode + 1}" + f"After Step Master state {master_BN_next_state_bool}" + f"After step Slave state {slave_PBN_next_state}")
                masterBNPreviousState = master_BN_next_state_bool
                master_BN_next_state_float = convert_state(master_BN_next_state_bool)

                #graphs
                #master_BN_next_state_int = self.convert_state_int(master_BN_next_state_bool)
                #slave_PBN_next_state_int = self.convert_state_int(slave_PBN_next_state)

                #master_BN_binary_next_state = self.convert_binary_to_decimal(master_BN_next_state_int)
                #slave_PBN_binary_next_state = self.convert_binary_to_decimal(slave_PBN_next_state_int)

                #master_BN_state_history.append(master_BN_next_state_int)
                #slave_PBN_state_history.append(slave_PBN_next_state_int)

                masterSlaveStateFloat = master_BN_next_state_float + slavePBNstate
                if (np.array_equal(master_BN_next_state_bool, slave_PBN_next_state)):
                    if (firstMatch == False):
                        firstMatch = True
                        numberOfStepsTakenForMatch = h + 1
                    correctAllEpisodes = correctAllEpisodes +1
                    correctEpisode = correctEpisode + 1

            slaveFollowedMasterEpisode = correctEpisode * (10/15)
            #if (episode<10):
                #self.plotGraphs(master_BN_state_history, slave_PBN_state_history, episode, slaveFollowedMasterEpisode, correctEpisode)
            slaveFollowedMasterEpisodeIgnoreFirstSteps = (correctEpisode - numberOfStepsTakenForMatch) * (100/(150 - numberOfStepsTakenForMatch))
            logging.debug(f"Episode {episode + 1}" + f"Slave followed master {slaveFollowedMasterEpisode} percent in this episode" + f"Slave followed master {correctEpisode} steps out of 150 steps")
            logging.debug(f"Episode {episode + 1}" + f"Slave followed master {slaveFollowedMasterEpisodeIgnoreFirstSteps} percent in this episode if we ignore the first steps until to reach the attractor" + f"Slave followed master {correctEpisode} steps out of (150 - {numberOfStepsTakenForMatch}) steps")
            slaveFollowedMasterAllEpisodes.append(slaveFollowedMasterEpisode)
            slaveFollowedMasterAllEpisodesIgnoreFirstSteps.append(slaveFollowedMasterEpisodeIgnoreFirstSteps)

        slaveFollowedMasterAll = correctAllEpisodes / 75
        logging.debug(f"Slave followed master {slaveFollowedMasterAll} percent in all episodes" + f"Slave followed master {correctAllEpisodes} steps out of 7500 steps")
        sumAll = 0.0
        sumWithout = 0.0
        for i in range(50):
            sumAll = sumAll + slaveFollowedMasterAllEpisodes[i]
            sumWithout = sumWithout + slaveFollowedMasterAllEpisodesIgnoreFirstSteps[i]
        avgAll = sumAll/50
        avgWithout = sumWithout/50
        logging.debug(f"Slave followed master {avgAll} percent in all episodes")
        logging.debug(f"Slave followed master {avgWithout} percent in all episodes if we ignore the first steps until slave's state equals master's state for first time")



    def save(self):
        with open("runs/DRL/masterBNslavePBNynodes25horizonChanged/masterslave", "wb") as f:
            pickle.dump(self.slaveAgent.controller, f)
        


    def train_Only_Slave_RL_Agent(self, conf):
        print(f"Training using {DEVICE}")
        self.slaveAgent.toggle_train(conf)
        
        writer = SummaryWriter("runs/DRL/masterBNslavePBNynodes25horizonChanged")

        slavePBNRewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)

        for epoch in tqdm(range(conf["train_epoch"])):
            start_time = time.time()
            slave_PBN_actions_chosen = set()
            steps_slave_followed_master = 0
            steps_slave_followed_master_first_time_episode = []
            #steps = 0

            for episode in tqdm(range(conf["train_episodes"])):
                print(
                    f"Epoch {epoch + 1}/{conf['train_epoch']} "
                    + f"- Episode {episode + 1}/{conf['train_episodes']} "
                    + f"- SlavePBNReward: {np.mean(slavePBNRewards, axis=1)[epoch]:.4f}, "
                    + ", ".join(
                        [
                            f"-SlavePBN {name}: {value:.4f}"
                            for name, value in self.slaveAgent.extra_params.items()
                        ]
                    )
                    + ", "
                    + f"Time elapsed: {int(time.time() - start_time)}s",
                    end="\r",
                )

                steps_slave_followed_master_first_time = horizon + 1
                slave_followed_master_first_time = False

                self.masterBN.reset()
                masterBNPreviousState = self.masterBN.PBN.state
                masterBNPreviousStateFloat = self.masterBN.render(mode="float")


                self.slavePBN.reset()
                slavePBNstate = self.slavePBN.render(mode="float")
                slave_PBN_episode_reward = 0

                masterSlaveStateFloat =  masterBNPreviousStateFloat + slavePBNstate

                for _ in range(horizon):
                    #steps += 1
                    interval = 1

                    master_BN_next_state_bool = self.masterBN.stepMaster()


                    slave_PBN_action = self.slaveAgent.get_action(masterSlaveStateFloat)
                    slave_PBN_actions_chosen.add(slave_PBN_action)
                    slave_PBN_next_state, slave_PBN_reward, slave_PBN_done, slave_PBN_info = self.slavePBN.slave_step(masterBNPreviousState, master_BN_next_state_bool, slave_PBN_action)
                    slave_PBN_next_state = convert_state(slave_PBN_next_state)

                    master_BN_next_state = convert_state(master_BN_next_state_bool)
                    masterSlaveNextStateFloat = master_BN_next_state + slave_PBN_next_state

                    self.slaveAgent.feedback(
                        Transition(
                            masterSlaveStateFloat,
                            slave_PBN_action,
                            interval,
                            slave_PBN_reward,
                            masterSlaveNextStateFloat,
                            slave_PBN_done,
                        )
                    )
                    slavePBNstate = slave_PBN_next_state
                    slave_PBN_episode_reward += slave_PBN_reward

                    masterBNPreviousState = master_BN_next_state_bool

                    if slave_PBN_done:
                        #steps_slave_followed_master = steps_slave_followed_master + 1
                        steps_slave_followed_master_first_time = _ + 1
                        break

                    #if (slave_PBN_done and (not slave_followed_master_first_time)):
                        #steps_slave_followed_master_first_time = _ + 1
                        #slave_followed_master_first_time = True

                steps_slave_followed_master_first_time_episode.append(steps_slave_followed_master_first_time)

                self.slaveAgent.update_params()


                slavePBNRewards[epoch, episode] = slave_PBN_episode_reward

            writer.add_scalar("slave_PBN_epoch_reward", np.mean(slavePBNRewards, axis=1)[epoch], epoch)
            writer.add_scalars(
                "slave_PBN_action_stats",
                {
                    #"slave_PBN_actions_chosen": len(slave_PBN_actions_chosen),
                    #"slave_PBN_n_interractions": steps / conf["train_episodes"],
                    #"steps_slave_PBN_followed_master_BN_percentage": (steps_slave_followed_master * 100) / ((conf["train_episodes"]) * horizon),
                    "steps_slave_PBN_followed_master_BN_first_time": np.mean(steps_slave_followed_master_first_time_episode)
                    #"average_number_of_steps_taken_slave_followed_master_first_time": 
                },
                epoch,
            )
        self.save()


    def plotGraphs(self, master_BN_state_history, slave_PBN_state_history, episode, percentage, correct):
        all_states = self.get_states()
        master_BN_state_history_converted = self.get_numbers_from_states(all_states, master_BN_state_history)
        slave_PBN_state_history_converted = self.get_numbers_from_states(all_states, slave_PBN_state_history)
        y_axis_before = list(np.arange((2 ** self.masterBN.PBN.N)))
        plt.figure()
        for i in range(len(master_BN_state_history_converted)):
            if (master_BN_state_history_converted[i] == slave_PBN_state_history_converted[i]):
                plt.plot(i, master_BN_state_history_converted[i], 'bo', markersize=5)
            else:
                plt.plot(i, master_BN_state_history_converted[i], '+g', markersize=5)
                plt.plot(i, slave_PBN_state_history_converted[i], 'xr', markersize=5)
        #plt.plot(master_BN_state_history_converted, '+g',label='state of Master BN', markersize=7)
        #plt.plot(slave_PBN_state_history_converted, 'xr', label='state of Slave PBN', markersize=7)
        plt.plot([], [], "bo", label='state of Slave and state of Master are the same. Slave follows Master')
        plt.plot([], [], '+g', label='state of Master BN')
        plt.plot([], [], 'xr', label='state of Slave PBN')

        plt.title(f"Test episode 151 horizon: Slave PBN followed Master BN " + f"{round(percentage, 2)} percent in this expirement. " + f"Slave followed Master {correct} steps out of 151 steps")
        plt.xlabel('horizon')
        plt.ylabel('State')
        plt.legend()
        plt.yticks(y_axis_before, all_states)
        plt.show()
    
    def convert_binary_to_decimal(self, state):
        binary_sum = 0
        number_of_nodes = len(state)
        for i, value in enumerate(state):
            binary_sum = binary_sum + value * (2 ** (number_of_nodes - 1 - i))
        return binary_sum
    
    def get_states(self):
        all_states = itertools.product([0, 1], repeat=self.masterBN.PBN.N)
        states = []
        for state in all_states:
            states.append(list(state))
        return states
    
    def get_numbers_from_states(self, all_states, network_state_history):
        converted_states = []
        for state in network_state_history:
            converted_states.append(all_states.index(state))
        return converted_states
    
    def convert_state_int(self, state):
        return [int(i) for i in state]