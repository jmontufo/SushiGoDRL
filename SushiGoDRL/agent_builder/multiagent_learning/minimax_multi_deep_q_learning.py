# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import random
import pickle

from agent_builder.utils import *
from agents_sushi_go.random_agent import RandomAgent
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase1

import torch
import torch.nn as nn

from collections import deque

import matplotlib.pyplot as plt

import gym

class StateTransformationData(object):
    
    def __init__(self, state_type):
        
        self.state_type = state_type
        self.fitted_lambdas = list()
        self.means = list()
        self.stDevs = list()
        
    def save(self, filename):
        
        output = open(filename + ".pkl", "wb")
        pickle.dump(self, output)
        output.close()    

class Memory(object):
    
    def __init__(self, batch_size, max_size):
        
        self.__batch_size = batch_size
        self.__max_size = max_size
        self.__buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        
        self.__buffer.append(experience)
    
    def sample(self):
        
        buffer_size = len(self.__buffer)
        
        index = np.random.choice(np.arange(buffer_size),
                                 size = self.__batch_size,
                                 replace = False)
        
        return [self.__buffer[i] for i in index]
    
    def get_buffer(self):
        return [self.__buffer[i] for i in range(len(self.__buffer))]
    
    def has_enough_experiments(self):
        
        return len(self.__buffer) >= self.__batch_size
    
    def is_full(self):
        
        return len(self.__buffer) == self.__max_size

    def num_experiments(self):
        
        return len(self.__buffer)

class DQNetwork(torch.nn.Module):
    
    def __init__(self, learning_rate, state_type, action_size, batch_size, discount):
        
        super(DQNetwork, self).__init__()
        
        self.__learning_rate = learning_rate
        self.__discount = discount
        self.__state_type = state_type        
        self.__batch_size = batch_size

        self.state_losses = []
        
        self.__state_size = state_type.get_number_of_observations()
        self.__action_size = action_size
        
        self.__q_network = self.__define_model()        
        self.__target_network = self.__define_model()
        
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        

    def __define_model(self):
               
        net = nn.Sequential(
            torch.nn.Linear(self.__state_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.__action_size * self.__action_size))
        
        return net
       
    
    def get_Q(self, state):
            
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
        
        return self.__q_network(state_t)
    
    
    def __get_target_Q(self, state):
            
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
        
        return self.__target_network(state_t)
        
    
    def get_action_state_value(self, state, actions_pair):

        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
                    
        actions_values = self.__q_network(state_t)
                
        return torch.gather(actions_values, 1, actions_pair)
    
    def get_next_action(self, state, legal_actions, rival_legal_actions):
        q_values = self.get_Q(state).detach()
                
        legal_actions_values = []
        
        for action in legal_actions:
            action_values = []
            
            for rival_action in rival_legal_actions:
                actions_pair = action + rival_action * self.__action_size
                action_values.append(q_values[actions_pair])
            
            legal_actions_values.append(np.amin(action_values))
            
        max_value = np.amax(legal_actions_values)           
        best_actions = np.argwhere(legal_actions_values == max_value).flatten().tolist()        
        chosen_action = legal_actions[random.choice(best_actions)]
        
        return chosen_action
    
    
    def update(self, states_batch, actions_batch, rewards_batch, 
                     new_states_batch, new_legal_actions_batch, 
                     rival_new_legal_actions_batch, dones_batch):
           
       self.__optimizer.zero_grad() 
                                     
       state_t = torch.FloatTensor(states_batch)
       reward_t = torch.FloatTensor(rewards_batch) 
       action_t = torch.LongTensor(actions_batch).reshape(-1,1)
       new_states_t = torch.FloatTensor(new_states_batch)
       dones_t = torch.BoolTensor(dones_batch)
       
       loss = self.calculate_loss(state_t, action_t, reward_t, new_states_t, 
                                  new_legal_actions_batch, 
                                  rival_new_legal_actions_batch, dones_t) 
       self.state_losses.append(loss.detach().numpy())
       loss.backward() 
       self.__optimizer.step()
       
       
    def calculate_loss(self, state_t, action_t, reward_t, new_states_t, 
                       new_legal_actions_batch, rival_new_legal_actions_batch, 
                       dones_t):
        
        qvals = self.get_action_state_value(state_t, action_t)
                 
        qvals_next = self.__get_target_Q(new_states_t).detach().numpy()
        
        new_legal_actions_filtered = []
        for i in range(len(new_legal_actions_batch)):
            new_legal_actions_bool = np.full(self.__action_size * self.__action_size, True)
            
            new_legal_actions = new_legal_actions_batch[i]
            rival_new_legal_actions = rival_new_legal_actions_batch[i]
            for action in new_legal_actions:
                action_values = []
                for rival_action in rival_new_legal_actions:
                    actions_pair = action + rival_action * self.__action_size
                    action_values.append(qvals_next[i][actions_pair])
                
                # print("action_values")
                # print(action_values)
                min_for_action = np.amin(action_values)
                # print("min_for_Action")
                # print(min_for_action)
                for rival_action in rival_new_legal_actions:
    
                    actions_pair = action + rival_action * self.__action_size 
                    
                    # print("qvals_next[actions_pair]")
                    # print(qvals_next[i][actions_pair])
                    if qvals_next[i][actions_pair] == min_for_action:
                        new_legal_actions_bool[actions_pair] = False
            
            new_legal_actions_filtered.append(new_legal_actions_bool)       
        
        new_legal_actions_t = torch.BoolTensor(new_legal_actions_filtered)
        
        # print("qvals_next")
        # print(qvals_next)
        qvals_next[new_legal_actions_t] = -100000000

        qvals_next = torch.FloatTensor(qvals_next)
        
        qvals_next = torch.max(qvals_next,
                               dim=-1)[0].detach()  
     
        
        qvals_next[dones_t] = 0
        
        expected_qvals = self.__discount * qvals_next + reward_t
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))       
        
        return loss  
    
    
    def update_target_network(self):
        
        self.__target_network.load_state_dict(self.__q_network.state_dict())
        
    # def load(self, filename):
        
    #     self.__q_network.load_state_dict(torch.load(filename + ".pt"))
    #     self.eval()
        
    #     self.__target_network.load_state_dict(self.__q_network.state_dict())
                
    # def save(self, filename):
        
    #     torch.save(self.__q_network.state_dict(), filename + ".pt")
        
 
class MinimaxMultiDQL_Builder(object):

    def __init__(self, state_type, total_episodes, max_epsilon, 
                 min_epsilon, decay_rate, learning_rate, discount,
                 reference = "", previous_nn_filename = None):             
                    
        self.total_episodes = total_episodes
        
        # Minimax only works for two players games, and with zero-sum rewards
        self.num_players = 2
        self.reward_by_win = -1
        
        self.state_type = state_type
        
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate        
        
        batch_size = 256
        memory_size = 1024         
        self.update_target_freq = 100         
                    
        self.chopsticks_phase_mode = state_type.trained_with_chopsticks_phase()
       
        self.memory = [Memory(batch_size, memory_size),
                       Memory(batch_size, memory_size)]
        self.env = gym.make('sushi-go-multi-v0', 
                            state_type = self.state_type,
                            num_players = self.num_players,
                            chopsticks_phase_mode = self.chopsticks_phase_mode, 
                            reward_by_win = self.reward_by_win)
                
        action_size = self.env.action_space.max_space
                
        self.dq_network = [DQNetwork(learning_rate, state_type, action_size,
                                     batch_size, discount),
                           DQNetwork(learning_rate, state_type, action_size,
                                     batch_size, discount)]
        
        if previous_nn_filename is not None:
            
            previous_episodes = previous_nn_filename.split("_")[-2]
            previous_episodes = int(previous_episodes)
                        
            self.dq_network = torch.load(previous_nn_filename + ".pt")
            # self.dq_network.eval()
            
            Q_input = open(previous_nn_filename + ".pkl", 'rb')
            self.state_transf_data = pickle.load(Q_input)
            Q_input.close()
            
        else:            
            previous_episodes = 0
            self.state_transf_data = None
        
                
        self.filename = "MinimaxMultiDQL_"
        self.filename += state_type.__name__ + "_"
        self.filename += "lr" + str(learning_rate) + "_"
        self.filename += "d" + str(discount) + "_"
        self.filename += str(previous_episodes + total_episodes) + "_"
        self.filename += reference
        
        
    def set_params(self,params):
        
        self.state_transf_data = params
        
    def save_params(self):
        
        self.state_transf_data.save("params")
        
    def test_with_random_agent(self, current_batch):
        
        agents_to_test = [
            DeepQLearningTorchAgentPhase1(),
            DoubleDeepQLearningTorchAgentPhase1(),
            DualingDeepQLearningTorchAgentPhase1(),
            MaxminDQLearningTorchAgentPhase1(),
            ]
        
        for player_num in range(2):
                
            games_number = 1000
            victories_by_player = [0,0]
            points_by_player = [0,0]
                                
            for i in range(0, games_number):
                
                agents = []
                agents.append(agents_to_test[i % len(agents_to_test)])
                
                test_env = gym.make('sushi-go-v0', agents = agents,
                                    state_type = self.state_type,
                                    chopsticks_phase_mode = self.chopsticks_phase_mode, 
                                    reward_by_win = self.reward_by_win)
                
                distributions = self.state_type.get_expected_distribution()
                
                state = test_env.reset()
                rival_new_legal_actions = list(range(9))
                
                done = False
            
                while not done:
                            
                    legal_actions = test_env.action_space.available_actions                   
                                    
                    transformed_state = []
    
                    for i, distribution in enumerate(distributions):
                            
                        state_attribute = state[i]
                        
                        if distribution == 'Poisson':
                            
                            # add 1 as all the attributes begin by 0, and is not allowed in the Box-Cox transformation
                            state_attribute = state_attribute + 1  
                            
                            lam = self.state_transf_data.fitted_lambdas[i]
                            
                            if lam == 0:
                                state_attribute = np.log10(state_attribute)
                            else:
                                state_attribute = ((state_attribute ** lam) - 1.) / lam
                                    
                        state_attribute -= self.state_transf_data.means[i]
                        state_attribute /= self.state_transf_data.stDevs[i]
                                        
                        transformed_state.append(state_attribute)                
                         
                    action = self.dq_network[player_num].get_next_action(np.array(transformed_state), 
                                                                legal_actions, 
                                                                rival_new_legal_actions)
                    
                    new_state, reward, done, info = test_env.step(action)   
                    
                    rival_new_legal_actions = info['rival_legal_actions']
                    state = new_state
                    
                winners_list = info['winners']
                scores_list = info['all_scores']
                # print (winners_list)
                # print (scores_list)
                
                for i in range(0,len(scores_list)):
                    if i in winners_list:
                        victories_by_player[i] += 1
                    points_by_player[i] += scores_list[i]
                
                current_batch[player_num].points += info['score']                  
                current_batch[player_num].points_by_victory += info['points_by_victory']
                
            print ("Wins per player: ")
            print (victories_by_player)
            print ("Total points per player: " )
            print (points_by_player)
               
            
        
    
    def run(self):
        
        epsilon = 1                 # Exploration rate
        
        rewards = [[],[]] 
        rewards_by_target_update =  [[],[]]
            
        batches_p1 = [] 
        batches_p2 = []
        current_batch = [BatchInfo(), BatchInfo()]
                
        distributions = self.state_type.get_expected_distribution()

        for episode in range(self.total_episodes):
          
            state = self.env.reset()
            
            done = False
            episode_rewards = [0] * 2
                        
            rival_new_legal_actions = [list(range(9)), list(range(9))]
            
            if episode > 0 and episode % 1000 == 0:
                
                self.test_with_random_agent(current_batch)
                                
                current_batch[0].epsilon_at_end = epsilon
                current_batch[1].epsilon_at_end = epsilon
                
                print(str(episode) + " episodes.")
                print("Rewards:  [" + str(current_batch[0].total_reward) + "," + str(current_batch[1].total_reward) + "]")
                print("Scores: [" + str(current_batch[0].points) + "," + str(current_batch[1].points) + "]")
                print("Wins: [" + str(current_batch[0].points_by_victory) + "," + str(current_batch[1].points_by_victory) + "]")
                print("Epsilon: " + str(current_batch[0].epsilon_at_end))
                
                episodes_batch_id = int(episode / 1000)
                batch_filename = self.filename + "-" + str(episodes_batch_id)
                
                torch.save(self.dq_network, batch_filename + ".pt")
                if self.state_transf_data is not None:
                    self.state_transf_data.save(batch_filename)
                                
                batches_p1.append(current_batch[0])
                batches_p2.append(current_batch[1])
                current_batch = [BatchInfo(), BatchInfo()]
                
                save_batches(batches_p1, batch_filename + "-p1-batches_info.txt")   
                save_batches(batches_p2, batch_filename + "-p2-batches_info.txt")   
                                    
            if episode > 0 and episode % self.update_target_freq == 0:
                # if self.state_transf_data != None:
                #     self.test_with_random_agent(current_batch)
                                    
                self.dq_network[0].update_target_network()
                self.dq_network[1].update_target_network()
                
                rewards_mean_p1 = sum(rewards[0][-self.update_target_freq:]) / self.update_target_freq
                rewards_by_target_update[0].append(rewards_mean_p1)
                print ("Reward over time p1: " +  str(rewards_mean_p1))
                
                rewards_mean_p2 = sum(rewards[1][-self.update_target_freq:]) / self.update_target_freq
                rewards_by_target_update[1].append(rewards_mean_p2)
                print ("Reward over time p2: " +  str(rewards_mean_p2))
                        
            while not done:     
                                        
                legal_actions = self.env.action_space.available_actions   
                actions = []
                             
                for player_num in range(2):
                    
                    exp_exp_tradeoff = random.uniform(0, 1)
                    
                    if self.state_transf_data != None and exp_exp_tradeoff > epsilon:
                        
                        current_batch[player_num].times_explotation += 1 
                        
                        transformed_state = []
    
                        for i, distribution in enumerate(distributions):
                                
                            state_attribute = state[player_num][i]
                            
                            if distribution == 'Poisson':
                                
                                # add 1 as all the attributes begin by 0, and is not allowed in the Box-Cox transformation
                                state_attribute = state_attribute + 1  
                                
                                lam = self.state_transf_data.fitted_lambdas[i]
                                
                                if lam == 0:
                                    state_attribute = np.log10(state_attribute)
                                else:
                                    state_attribute = ((state_attribute ** lam) - 1.) / lam
                                        
                            state_attribute -= self.state_transf_data.means[i]
                            state_attribute /= self.state_transf_data.stDevs[i]
                                            
                            transformed_state.append(state_attribute)                
                             
                        action = self.dq_network[num_player].get_next_action(np.array(transformed_state), 
                                                                             legal_actions[player_num], 
                                                                             rival_new_legal_actions[player_num])
            
                    else:
                        
                        current_batch[player_num].times_exploration += 1
                        action = random.choice(legal_actions[player_num])
                        
                    actions.append(action)
                    
                new_state, reward, done, info = self.env.step(actions)        
                           
                new_legal_actions = self.env.action_space.available_actions
                rival_new_legal_actions = info['rival_legal_actions']
                # print("rival_new_legal_actions")
                # print(rival_new_legal_actions)
                # rival_action = info['rival_action']
                
                # print(action)
                # print(rival_action)
                
                pairs_of_actions = []
                pairs_of_actions.append(actions[0] + actions[1] * self.env.action_space.max_space)
                pairs_of_actions.append(actions[1] + actions[0] * self.env.action_space.max_space)
               
                                    
                self.memory[0].add([state[0], pairs_of_actions[0], reward[0], new_state[0], 
                                 new_legal_actions[0], rival_new_legal_actions[0], 
                                 done])
                self.memory[1].add([state[1], pairs_of_actions[1], reward[1], new_state[1], 
                              new_legal_actions[1], rival_new_legal_actions[1], 
                              done])
                
                               
                # print("last memory")
                # print(self.memory[1].get_buffer()[-1])
                
                episode_rewards[0] += reward[0]
                episode_rewards[1] += reward[1]
                
                state = new_state
            
            if self.memory[0].is_full() and self.state_transf_data == None:            
            
                self.state_transf_data = StateTransformationData(self.state_type)
                               
                buffer = self.memory[0].get_buffer()
                states = np.array([each[0] for each in buffer])
                
                
                for i, distribution in enumerate(distributions):
                    
                    state_attribute = states[:,i]
                    
                    if distribution == 'Poisson':
                        
                        # add 1 as all the attributes begin by 0, and is not allowed in the Box-Cox transformation
                        state_attribute = state_attribute + 1
                        
                        transformed_data, fitted_lambda = stats.boxcox(state_attribute)
                        self.state_transf_data.fitted_lambdas.append(fitted_lambda)
                        self.state_transf_data.means.append(np.mean(transformed_data,0))
                        self.state_transf_data.stDevs.append(np.std(transformed_data,0))
                    
                    else:
                        
                        self.state_transf_data.fitted_lambdas.append(0)
                        self.state_transf_data.means.append(np.mean(state_attribute,0))
                        self.state_transf_data.stDevs.append(np.std(state_attribute,0))
                        
            if self.memory[0].has_enough_experiments() and self.state_transf_data != None:
                      

                for num_player in range(2):                  
                    batch = self.memory[num_player].sample()
                    
                    states_batch = np.array([each[0] for each in batch]).astype(float)
                    actions_batch = np.array([each[1] for each in batch])
                    rewards_batch = np.array([each[2] for each in batch]) 
                    new_states_batch = np.array([each[3] for each in batch]).astype(float)
                    new_actions_batch = np.array([each[4] for each in batch], dtype=object)
                    rival_new_actions_batch = np.array([each[5] for each in batch], dtype=object)
                    dones_batch = np.array([each[6] for each in batch])
                    
                    distributions = self.state_type.get_expected_distribution()
                    
                    for i, distribution in enumerate(distributions):
                            
                        state_attribute = states_batch[:,i]
                        new_state_attribute = new_states_batch[:,i]
                        
                        if distribution == 'Poisson':
                            
                            # add 1 as all the attributes begin by 0, and is not allowed in the Box-Cox transformation
                            state_attribute = state_attribute + 1                
                            state_attribute = stats.boxcox(state_attribute, self.state_transf_data.fitted_lambdas[i])
                            
                            new_state_attribute = new_state_attribute + 1                
                            new_state_attribute = stats.boxcox(new_state_attribute, self.state_transf_data.fitted_lambdas[i])
                        
                                            
                        state_attribute -= self.state_transf_data.means[i]
                        state_attribute /= self.state_transf_data.stDevs[i]
                                        
                        states_batch[:,i] = state_attribute                
                        
                        new_state_attribute -= self.state_transf_data.means[i]
                        new_state_attribute /= self.state_transf_data.stDevs[i]
                        
                        new_states_batch[:,i] = new_state_attribute    
                    
                    self.dq_network[num_player].update(states_batch, actions_batch,
                                           rewards_batch, new_states_batch, 
                                           new_actions_batch, 
                                           rival_new_actions_batch,dones_batch)  
            
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode) 
              
            
            rewards[0].append(episode_rewards[0])
            rewards[1].append(episode_rewards[1])
            
            # current_batch[0].total_reward += episode_rewards[0]         
            # current_batch[0].points += info['score'][0]                   
            # current_batch[0].points_by_victory += info['points_by_victory'][0]
            
            # current_batch[1].total_reward += episode_rewards[1]         
            # current_batch[1].points += info['score'][1]                   
            # current_batch[1].points_by_victory += info['points_by_victory'][1]
        
                
        torch.save(self.dq_network, self.filename + ".pt")
        self.state_transf_data.save(self.filename)
        
        batches_p1.append(current_batch[0])   
        batches_p2.append(current_batch[1])            
        plt.figure(figsize=(36, 48))
          
        plt.subplot(2, 1, 2)
        plt.plot(self.dq_network[0].state_losses, label='loss')
        plt.title('State Loss Function Evolution P1')
        plt.subplot(2, 2, 2)
        plt.plot(self.dq_network[1].state_losses, label='loss')
        plt.title('State Loss Function Evolution P2')
        plt.legend()    
        
        save_batches(batches_p1, self.filename + "-p1-batches_info.txt")   
        save_batches(batches_p2, self.filename + "-p2-batches_info.txt")   
                             
