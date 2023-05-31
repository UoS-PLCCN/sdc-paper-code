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
import csv



class MasterBNSlavePBN:

    def __init__(self, masterBN, slavePBN):
        
        self.masterBN = masterBN
        self.slavePBN = slavePBN


        input_s = 2 * masterBN.observation_space.n

        self.slaveAgent = PERAgent(
            {
                "seed": 1234,
                "height": 75,
                "gamma": gamma,
                "train_time_horizon": horizon,
                "input_size": input_s,
                "output_size": slavePBN.discrete_action_space.n
                #"output_size": slavePBN.action_space.n
            }
        )

    def test(self):
        self.slaveAgent.training = False
        used_states = set()



        correctAllEpisodes = 0
        slaveFollowedMasterAllEpisodes = []
        slaveFollowedMasterAllEpisodesIgnoreFirstSteps = []
        stepsUntilSlaveFollowedMasterFistTimeAllEpisodes = []
        num_episodes = 10000
        num_horizon = 25
        
        average_node_changes_per_step = dict()
        for j in range(num_horizon):
            average_node_changes_per_step[j] = 0


        for episode in tqdm(range(num_episodes)):



            correctEpisode = 0
            self.masterBN.reset()
            masterBNPreviousState = self.masterBN.PBN.state
            masterBNPreviousStateFloat = self.masterBN.render(mode="float")


            self.slavePBN.reset()
            slavePBNstate = self.slavePBN.render(mode="float")
                
            masterSlaveStateFloat = masterBNPreviousStateFloat + slavePBNstate
            masterSlaveStateFloat = tuple(masterBNPreviousStateFloat + slavePBNstate)

            while (masterSlaveStateFloat in used_states):
            
                self.masterBN.reset()
                masterBNPreviousState = self.masterBN.PBN.state
                masterBNPreviousStateFloat = self.masterBN.render(mode="float")

                self.slavePBN.reset()
                slavePBNstate = self.slavePBN.render(mode="float")
                masterSlaveStateFloat = masterBNPreviousStateFloat + slavePBNstate

            used_states.add(masterSlaveStateFloat)


            firstMatch = False
            numberOfStepsTakenForMatch = 0

            for h in range(num_horizon):
                masterBN_previous_state_bool = self.masterBN.PBN.state
                master_BN_next_state_bool = self.masterBN.stepMaster()
                average_node_changes_per_step[h] += np.count_nonzero(np.not_equal(masterBN_previous_state_bool, master_BN_next_state_bool))

                

                slave_PBN_action = self.slaveAgent.get_action(masterSlaveStateFloat)


                slave_PBN_next_state, slave_PBN_reward, slave_PBN_done, slave_PBN_info = self.slavePBN.slave_step(masterBNPreviousState, master_BN_next_state_bool, slave_PBN_action)
                slavePBNstate = convert_state(slave_PBN_next_state)
                masterBNPreviousState = master_BN_next_state_bool
                master_BN_next_state_float = convert_state(master_BN_next_state_bool)



                masterSlaveStateFloat = master_BN_next_state_float + slavePBNstate
                if (np.array_equal(master_BN_next_state_bool, slave_PBN_next_state)):
                    if (firstMatch == False):
                        firstMatch = True
                        numberOfStepsTakenForMatch = h + 1
                    correctAllEpisodes = correctAllEpisodes +1
                    correctEpisode = correctEpisode + 1

            slaveFollowedMasterEpisode = correctEpisode * (100/num_horizon)
            slaveFollowedMasterEpisodeIgnoreFirstSteps = correctEpisode * (100/(num_horizon - numberOfStepsTakenForMatch + 1))
            slaveFollowedMasterAllEpisodes.append(slaveFollowedMasterEpisode)
            slaveFollowedMasterAllEpisodesIgnoreFirstSteps.append(slaveFollowedMasterEpisodeIgnoreFirstSteps)
            stepsUntilSlaveFollowedMasterFistTimeAllEpisodes.append(numberOfStepsTakenForMatch)

        slaveFollowedMasterAll = correctAllEpisodes * (100/(num_episodes*num_horizon))
        sumAll = 0.0
        sumWithout = 0.0
        sumStepsSlaveFollowedMaster = 0
        for i in range(num_episodes):
            sumAll = sumAll + slaveFollowedMasterAllEpisodes[i]
            sumWithout = sumWithout + slaveFollowedMasterAllEpisodesIgnoreFirstSteps[i]
            sumStepsSlaveFollowedMaster += stepsUntilSlaveFollowedMasterFistTimeAllEpisodes[i]
        avgAll = sumAll/num_episodes
        avgWithout = sumWithout/num_episodes
        meanStepsMasterFOllowedSlave = sumStepsSlaveFollowedMaster/num_episodes
        plot_average_steps = []
        for key in average_node_changes_per_step:
            average_node_changes_per_step[key] /= num_episodes
            plot_average_steps.append(average_node_changes_per_step[key])

        graph_x_axis = []
        for j in range(horizon):
            graph_x_axis.append(j+1)
        
        #graph for training and validation loss
        plt.rcParams.update({'font.size': 15})
        plt.plot(graph_x_axis, plot_average_steps, 'r')
        plt.title('Number of master Boolean Network nodes that changed their values in each deterinistic step of the Boolean Network')
        plt.xlabel('Deterministic step of the Boolean Network', fontsize=15)
        plt.ylabel('Number of master Boolean Network nodes that changed their values', fontsize=15)
        plt.legend()
        plt.show()

        with open('testingNetworks.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Percentage slave Followed master in all episodes'])
            writer.writerow([round(avgAll, 2)])
            writer.writerow([' '])
            writer.writerow(['Percentage slave followed master in all episodes if we ignore steps until slave state equals master stage first time'])
            writer.writerow([round(avgWithout, 2)])
            writer.writerow([' '])
            writer.writerow(['Mean of steps until followed first time'])
            writer.writerow([round(meanStepsMasterFOllowedSlave, 2)])
            for z in range(num_horizon):
                writer.writerow([' '])
                writer.writerow([f"Average number of the master Boolean Network nodes changed in step {z+1}"])
                writer.writerow([round(average_node_changes_per_step[z], 2)])






    def save(self):
        with open("runs/DRL/masterSlaveTestNetworks/masterslave", "wb") as f:
            pickle.dump(self.slaveAgent.controller, f)
        


    def train_Only_Slave_RL_Agent(self, conf):
        print(f"Training using {DEVICE}")
        self.slaveAgent.toggle_train(conf)
        
        writer = SummaryWriter("runs/DRL/masterSlaveTestNetworks")

        slavePBNRewards = np.zeros((conf["train_epoch"], conf["train_episodes"]), dtype=float)
        average_steps_slave_followed_master_first_time_epochs = []
        slave_followed_master_epochs = []

        for epoch in tqdm(range(conf["train_epoch"])):
            start_time = time.time()
            slave_PBN_actions_chosen = set()
            #steps = 0

            chosen_a = dict()
            for z in range (8):
                chosen_a[z] = 0

            slave_followed_master_epoch = 0
            steps_slave_followed_master_first_time_epoch = 0

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

                steps_slave_followed_master_first_time = -1

                self.masterBN.reset()
                masterBNPreviousState = self.masterBN.PBN.state
                masterBNPreviousStateFloat = self.masterBN.render(mode="float")


                self.slavePBN.reset()
                slavePBNstate = self.slavePBN.render(mode="float")
                slave_PBN_episode_reward = 0

                masterSlaveStateFloat =  masterBNPreviousStateFloat + slavePBNstate

                for _ in range(horizon):
                    interval = 1

                    master_BN_next_state_bool = self.masterBN.stepMaster()


                    slave_PBN_action = self.slaveAgent.get_action(masterSlaveStateFloat)
                    slave_PBN_actions_chosen.add(slave_PBN_action)
                    slave_PBN_previous_state = self.slavePBN.PBN.state
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


                    if slave_PBN_done:
                        chosen_a[slave_PBN_action] += 1
                        steps_slave_followed_master_first_time = _ + 1
                        slave_followed_master_epoch += 1
                        break


                    
                    masterBNPreviousState = master_BN_next_state_bool

                if (steps_slave_followed_master_first_time != -1):
                    steps_slave_followed_master_first_time_epoch += steps_slave_followed_master_first_time

                self.slaveAgent.update_params()


                slavePBNRewards[epoch, episode] = slave_PBN_episode_reward

            if (slave_followed_master_epoch != 0):
                average_steps_slave_followed_master_first_time_epoch = steps_slave_followed_master_first_time_epoch/slave_followed_master_epoch
            else:
                average_steps_slave_followed_master_first_time_epoch = -1
            average_steps_slave_followed_master_first_time_epochs.append(average_steps_slave_followed_master_first_time_epoch)
            slave_followed_master_epochs.append(slave_followed_master_epoch)


        with open('train_run_until_followed_first.csv', 'w', newline='') as fi:
            wr = csv.writer(fi)
            for j in range(len(average_steps_slave_followed_master_first_time_epochs)):
                wr.writerow([f'Epoch {j+1}'])
                wr.writerow(['Average steps of epoch the slave took until followed master first time', 'Number of times slave followed master in epoch'])
                wr.writerow([average_steps_slave_followed_master_first_time_epochs[j], slave_followed_master_epochs[j]])
                wr.writerow([' '])
        self.save()
