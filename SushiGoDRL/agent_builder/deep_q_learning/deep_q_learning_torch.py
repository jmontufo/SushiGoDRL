# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
from scipy import stats
import random
import pickle

from agent_builder.utils import *

import torch
import torch.nn as nn

from collections import deque

from states_sushi_go.complete_state_fixed import CompleteState
from agents_sushi_go.random_agent import RandomAgent
from agents_sushi_go.card_lover_agent import SashimiLoverAgent
from agents_sushi_go.card_lover_agent import SashimiSuperLoverAgent
from agents_sushi_go.card_lover_agent import SashimiHaterAgent
from agents_sushi_go.card_lover_agent import TempuraLoverAgent
from agents_sushi_go.card_lover_agent import TempuraSuperLoverAgent
from agents_sushi_go.card_lover_agent import DumplingLoverAgent
from agents_sushi_go.card_lover_agent import DumplingSuperLoverAgent
from agents_sushi_go.card_lover_agent import MakiLoverAgent
from agents_sushi_go.card_lover_agent import MakiSuperLoverAgent
from agents_sushi_go.card_lover_agent import MakiHaterAgent
from agents_sushi_go.card_lover_agent import WasabiLoverAgent
from agents_sushi_go.card_lover_agent import WasabiLoverAtFirstAgent
from agents_sushi_go.card_lover_agent import NigiriLoverAgent
from agents_sushi_go.card_lover_agent import NigiriSuperLoverAgent
from agents_sushi_go.card_lover_agent import PuddingLoverAgent
from agents_sushi_go.card_lover_agent import PuddingSuperLoverAgent
from agents_sushi_go.card_lover_agent import PuddingHaterAgent
from agents_sushi_go.card_lover_agent import ChopstickLoverAgent
from agents_sushi_go.card_lover_agent import ChopstickHaterAgent
from agents_sushi_go.card_lover_agent import ChopstickLoverAtFirstAgent
from agents_sushi_go.q_learning_agent import QLearningAgentPhase1
from agents_sushi_go.q_learning_agent import QLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent import DoubleDeepQLearningAgentPhase1
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
import matplotlib.pyplot as plt
# deregisterCustomGym('sushi-go-v0')
# deregisterCustomGym('sushi-go-multi-v0')
# import gym_sushi_go     



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
    
    def __init__(self, learning_rate, state_type, action_size, batch_size):
        
        super(DQNetwork, self).__init__()
        
        self.__learning_rate = learning_rate
        self.__state_type = state_type        
        self.__batch_size = batch_size

        self.state_losses = []
        
        self.__state_size = state_type.get_number_of_observations()
        self.__action_size = action_size
        
        self.__q_network = self.__define_model()
        
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        

    def __define_model(self):
               
        critic = nn.Sequential(
            torch.nn.Linear(self.__state_size, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.__action_size))
        
        return critic
       
    
    def __get_Q(self, state):
            
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
        
        return self.__q_network(state_t)
        
    
    def get_action_state_value(self, state, action):

        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
                    
        actions_values = self.__q_network(state_t)
        
        return torch.gather(actions_values, 1, action)
    
    def get_next_action(self, state, legal_actions):
        
        q_values = self.__get_Q(state).detach()
                
        legal_actions_values = []
        
        for action in legal_actions:
            legal_actions_values.append(q_values[action])
            
        max_value = np.amax(legal_actions_values)           
        best_actions = np.argwhere(legal_actions_values == max_value).flatten().tolist()        
        chosen_action = legal_actions[random.choice(best_actions)]
        
        return chosen_action
    
    
    def update(self, states_batch, actions_batch, rewards_batch, new_states_batch, new_legal_actions_batch, dones_batch):
           
       self.__optimizer.zero_grad() 
       
       state_t = torch.FloatTensor(states_batch)
       reward_t = torch.FloatTensor(rewards_batch) 
       action_t = torch.LongTensor(actions_batch).reshape(-1,1)
       new_states_t = torch.FloatTensor(new_states_batch)
       new_legal_actions_t = torch.BoolTensor(new_legal_actions_batch)
       dones_t = torch.BoolTensor(dones_batch)
       
       loss = self.calculate_loss(state_t, action_t, reward_t, new_states_t, new_legal_actions_t, dones_t) 
       self.state_losses.append(loss.detach().numpy())
       loss.backward() 
       self.__optimizer.step()
       
       
    def calculate_loss(self, state_t, action_t, reward_t, new_states_t, new_legal_actions_t, dones_t):
        
        qvals = self.get_action_state_value(state_t, action_t)
       
        qvals_next = self.__get_Q(new_states_t)
        qvals_next[new_legal_actions_t] = -100000000

        
        qvals_next = torch.max(qvals_next,
                               dim=-1)[0].detach()  
     
        
        qvals_next[dones_t] = 0
        print("qvals_next")
        print(qvals_next)   
        
        expected_qvals = qvals_next + reward_t
        
          
        # print("qvals")
        # print(qvals)     
        # print("expected_qvals reshape")
        # print(expected_qvals.reshape(-1,1))   
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))       
        
        return loss  
        
    def load(self, filename):
        return
       
                
    def save(self, filename):
        
       return
        
 
class DQLT_Builder(object):

    def __init__(self, num_players, versus_agents, state_type, total_episodes, 
                 max_epsilon, min_epsilon, decay_rate, learning_rate, 
                 reference = "", previous_nn_filename = None):             
                    
        self.total_episodes = total_episodes
        
        self.num_players = num_players
        
        self.state_type = state_type
        
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate        
        
        batch_size = 1024
        # batch_size = 16
        memory_size = 4096         
        # self.update_target_freq = 100         
        
        self.versus_agents = versus_agents
        self.agents = [] 
            
        for i in range(num_players - 1):
            self.agents.append(random.choice(versus_agents))    
            
            
        chopsticks_phase_mode = state_type.trained_with_chopsticks_phase()
       
        self.memory = Memory(batch_size, memory_size)
        self.env = gym.make('sushi-go-v0', agents = self.agents, state_type = state_type,
        chopsticks_phase_mode = chopsticks_phase_mode)
        
        
        action_size = self.env.action_space.n
                
        self.dq_network = DQNetwork(learning_rate, state_type, action_size, batch_size)
        
        if previous_nn_filename is not None:
            
            previous_episodes = previous_nn_filename.split("_")[-2]
            previous_episodes = int(previous_episodes)
            
            self.dq_network.load(previous_nn_filename)
            
            Q_input = open(previous_nn_filename + ".pkl", 'rb')
            self.state_transf_data = pickle.load(Q_input)
            Q_input.close()
            
        else:
            
            previous_episodes = 0
            self.state_transf_data = None
        
                
        self.filename = "Deep_Q_Learning_torch_"
        self.filename += str(num_players) + "p_"
        self.filename += state_type.__name__ + "_"
        self.filename += "lr" + str(learning_rate) + "_"
        self.filename += str(previous_episodes + total_episodes) + "_"
        self.filename += reference
        
        
    def set_params(self,params):
        
        self.state_transf_data = params
        
    def save_params(self):
        
        self.state_transf_data.save("params")
    
    def run(self):
        
        epsilon = 1                 # Exploration rate
        
        rewards = []
        # rewards_by_target_update = []
            
        batches = []
        current_batch = BatchInfo()
                
        distributions = self.state_type.get_expected_distribution()

        for episode in range(self.total_episodes):
            
            for i in range(self.num_players - 1):
                self.agents[i] = random.choice(self.versus_agents) 
            
            state = self.env.reset()
                    
            done = False
            episode_rewards = 0
            
            if episode > 0 and episode % 1000 == 0:
                                
                current_batch.epsilon_at_end = epsilon
                
                print(str(episode) + " episodes.")
                print("Reward: " + str(current_batch.total_reward))
                print("Epsilon: " + str(current_batch.epsilon_at_end))
                
                episodes_batch_id = int(episode / 1000)
                batch_filename = self.filename + "-" + str(episodes_batch_id)
                
                self.dq_network.save(batch_filename)  
                if self.state_transf_data is not None:
                    self.state_transf_data.save(batch_filename)
                                
                batches.append(current_batch)
                current_batch = BatchInfo()
                
                save_batches(batches, batch_filename + "-batches_info.txt")   
                                    
            # if episode > 0 and episode % self.update_target_freq == 0:
             
            #     self.dq_network.update_target_network()
                
            #     rewards_mean = sum(rewards[-self.update_target_freq:]) / self.update_target_freq
            #     rewards_by_target_update.append(rewards_mean)
            #     print ("Score over time: " +  str(rewards_mean))
                        
            while not done:     
                        
                exp_exp_tradeoff = random.uniform(0, 1)
                
                legal_actions = self.env.action_space.available_actions   
                                    
                if self.state_transf_data != None and exp_exp_tradeoff > epsilon:
                    
                    current_batch.times_explotation += 1 
                    
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
                         
                    action = self.dq_network.get_next_action(np.array(transformed_state), legal_actions)
        
                else:
                    
                    current_batch.times_exploration += 1
                    action = random.choice(legal_actions)
                    
                new_state, reward, done, info = self.env.step(action)        
                                                  
                new_legal_actions = self.env.action_space.available_actions
                
                legal_actions_array = np.full(self.env.action_space.n, True)
                
                for action in new_legal_actions:
                    legal_actions_array[action] = False
                                    
                self.memory.add([state, action, reward, new_state, legal_actions_array, done])
                               
                episode_rewards += reward
                
                state = new_state
            
            if self.memory.is_full() and self.state_transf_data == None:            
            
                self.state_transf_data = StateTransformationData(self.state_type)
                
                buffer = self.memory.get_buffer()
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
                        
            if self.memory.has_enough_experiments() and self.state_transf_data != None:
                                        
                batch = self.memory.sample()
                
                states_batch = np.array([each[0] for each in batch]).astype(float)
                actions_batch = np.array([each[1] for each in batch])
                rewards_batch = np.array([each[2] for each in batch]) 
                new_states_batch = np.array([each[3] for each in batch]).astype(float)
                new_actions_batch = np.array([each[4] for each in batch])
                dones_batch = np.array([each[5] for each in batch])
                
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
                
                self.dq_network.update(states_batch, actions_batch, rewards_batch, new_states_batch, new_actions_batch, dones_batch)  
            
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode) 
              
            rewards.append(episode_rewards)
            current_batch.total_reward += episode_rewards
        
        self.dq_network.save(self.filename) 
        self.state_transf_data.save(self.filename)
        
        batches.append(current_batch)            
        plt.figure(figsize=(36, 48))
          
        plt.subplot(2, 1, 2)
        plt.plot(self.dq_network.state_losses, label='loss')
        plt.title('State Loss Function Evolution')
        plt.legend()    
        
        save_batches(batches, self.filename + "-batches_info.txt")   
                             
