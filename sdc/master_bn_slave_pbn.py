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

        self.slaveAgent = PERAgent(
            {
                "seed": 1234,
                "height": 50,
                "gamma": gamma,
                "train_time_horizon": horizon,
                "input_size": slavePBN.observation_space.n,
                "output_size": slavePBN.action_space.n
            }
        )

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


    def test(self):
        logging.basicConfig(filename='myapp.log', level=logging.DEBUG)

        correctAllEpisodes = 0
        for episode in tqdm(range(50)):

            correctEpisode = 0
            self.masterBN.reset()
            masterBNPreviousState = self.masterBN.PBN.state


            self.slavePBN.reset()
            slavePBNstate = self.slavePBN.render(mode="float")
            logging.debug(" ")
            logging.debug(f"Start of episode {episode + 1}" + f"masterBNOriginalState {masterBNPreviousState}" + f"slavePBNOriginalstate {slavePBNstate}")

            for h in range(150):

                logging.debug(f"Episode {episode + 1}" + f"Episode {episode + 1}" + f"Before Step Master state {masterBNPreviousState}" + f"Before step Slave state {slavePBNstate}")

                master_BN_next_state_bool = self.masterBN.stepMaster()

                slave_PBN_action = self.slaveAgent.get_action(slavePBNstate)
                slave_PBN_next_state, slave_PBN_reward, slave_PBN_done, slave_PBN_info = self.slavePBN.slave_step(masterBNPreviousState, master_BN_next_state_bool, slave_PBN_action)
                logging.debug(f"Episode {episode + 1}" + f"Episode {episode + 1}" + f"After Step Master state {master_BN_next_state_bool}" + f"After step Slave state {slave_PBN_next_state}")
                if (np.array_equal(master_BN_next_state_bool, slave_PBN_next_state)):
                    correctAllEpisodes = correctAllEpisodes +1
                    correctEpisode = correctEpisode + 1

            slaveFollowedMasterEpisode = correctEpisode * (10/15)
            logging.debug(f"Episode {episode + 1}" + f"Slave followed master {slaveFollowedMasterEpisode} percent in this episode" + f"Slave followed master {correctEpisode} steps out of 150 steps")

        slaveFollowedMasterAll = correctAllEpisodes / 75
        logging.debug(f"Slave followed master {slaveFollowedMasterAll} percent in all episodes" + f"Slave followed master {correctAllEpisodes} steps out of 7500 steps")



        


    def train_Only_Slave_RL_Agent(self, conf):
        print(f"Training using {DEVICE}")
        self.slaveAgent.toggle_train(conf)
        
        writer = SummaryWriter(f"runs/DRL/slavePBN")

        slavePBNRewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)

        for epoch in tqdm(range(conf["train_epoch"])):
            start_time = time.time()
            slave_PBN_actions_chosen = set()
            steps = 0

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

                self.masterBN.reset()
                masterBNstate = self.masterBN.render(mode="float")
                masterBNPreviousState = self.masterBN.PBN.state


                self.slavePBN.reset()
                slavePBNstate = self.slavePBN.render(mode="float")
                slave_PBN_episode_reward = 0

                for _ in tqdm(range(horizon)):
                    steps += 1
                    interval = 1

                    master_BN_next_state_bool = self.masterBN.stepMaster()
                    master_BN_next_state = convert_state(master_BN_next_state_bool)

                    masterBNstate = master_BN_next_state

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


                self.slaveAgent.update_params()

                slavePBNRewards[epoch, episode] = slave_PBN_episode_reward

            writer.add_scalar("slave_PBN_epoch_reward", np.mean(slavePBNRewards, axis=1)[epoch], epoch)
            writer.add_scalars(
                "slave_PBN_action_stats",
                {
                    "slave_PBN_actions_chosen": len(slave_PBN_actions_chosen),
                    "slave_PBN_n_interractions": steps / conf["train_episodes"],
                },
                epoch,
            )

