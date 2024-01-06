# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:26:39 2023

@author: jmont
"""

import random
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import pickle
from agent_builder.utils import BatchInfo, save_batches
from scipy import stats
import matplotlib.pyplot as plt

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

class ActorCriticNetwork(torch.nn.Module):

    def __init__(self, learning_rate, critic_lr, state_type, action_size, discount):
        """
        Params
        ======
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.input_shape = state_type.get_number_of_observations()
        self.n_outputs = action_size
        self.discount = discount
                    
        self.learning_rate = learning_rate
        
        self.actor = nn.Sequential(
            torch.nn.Linear(self.input_shape, 256, bias=True), 
            torch.nn.Tanh(), 
            torch.nn.Linear(256, self.n_outputs, bias=True),
            torch.nn.Softmax(dim=-1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.losses = []
        self.state_losses = []
        
        self.critic_lr = critic_lr
        
        self.critic = nn.Sequential(
            torch.nn.Linear(self.input_shape, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(64, 64),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_outputs))
        
        self.optimizer_critic = torch.optim.Adam(self.parameters(), lr=critic_lr)
        # print(self.red_lineal)

    def get_action_prob(self, state):
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
        
        return self.actor(state_t)
    
    def get_action_probability(self, state, action):

        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
                    
        actions_probabilities = self.actor(state_t)
        
        # print("actions_probabilities")
        # print(actions_probabilities)
        # print("action")
        # print(action.reshape(-1,1))
        return actions_probabilities.gather(1, action.reshape(-1,1))
    
    def get_next_action(self, state, legal_actions):
        
        action_probs = self.get_action_prob(state).detach().numpy()
        
        is_legal_actions = np.zeros(self.n_outputs)
        for legal_action in legal_actions:
            is_legal_actions[legal_action] = 1
            
        for action in range(self.n_outputs):
            action_probs[action] = action_probs[action] * is_legal_actions[action]            
            
        prob_factor = 1 / sum(action_probs)
        action_probs = [prob_factor * p for p in action_probs]

        action_space = np.arange(self.n_outputs)
          
        if sum(action_probs) == 1:
            action = np.random.choice(action_space, p=action_probs)
        else:
            action = np.random.choice(legal_actions)
        
        return action

    def get_action_state_value(self, state, action):

        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
                    
        actions_values = self.critic(state_t)
        
        return torch.gather(actions_values, 1, action)
    
    def get_qvals(self, state):
        
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
        
        return self.critic(state_t)
    
    def update(self, states_batch, actions_batch, rewards_batch, new_states_batch, new_legal_actions_batch, dones_batch):
        
        state_t = torch.FloatTensor(states_batch)
        action_t = torch.LongTensor(actions_batch).reshape(-1,1)
        
        self.optimizer.zero_grad() 
        loss = self.calculate_actor_loss(state_t, action_t) 
        loss.backward() 
        self.optimizer.step()
        
        self.update_critic(states_batch, actions_batch, rewards_batch, new_states_batch, new_legal_actions_batch, dones_batch)
     
        self.losses.append(loss.detach().numpy())
        # self.update_loss.append(loss.detach().numpy())
    def update_critic(self, states_batch, actions_batch, rewards_batch, new_states_batch, new_legal_actions_batch, dones_batch):
        # print("update")
        # print(rewards_batch)
        state_t = torch.FloatTensor(states_batch)
        reward_t = torch.FloatTensor(rewards_batch) 
        action_t = torch.LongTensor(actions_batch).reshape(-1,1)
        new_states_t = torch.FloatTensor(new_states_batch)
        new_legal_actions_t = torch.BoolTensor(new_legal_actions_batch)
        dones_t = torch.BoolTensor(dones_batch)
        
        self.optimizer_critic.zero_grad() 
        loss = self.calculate_critic_loss(state_t, action_t, reward_t, new_states_t, new_legal_actions_t, dones_t) 
  
        loss.backward() 
        self.optimizer_critic.step()
        
        self.state_losses.append(loss.detach().numpy())
        # self.update_loss.append(loss.detach().numpy())
              
    def calculate_actor_loss(self, state_t, action_t):
        
        # print("state_t")
        # print(state_t.shape)
        # print("action_t")
        # print(action_t.shape)
        # print("calculate_loss")
        probs = self.get_action_probability(state_t, action_t)
        # print("advantage_t")
        # print(reward_t)
        # print("probs")
        # print(probs)
        logprob = - torch.log(probs)
        # 
        # print("logprob")
        # print(logprob)
        
        # print("action_t")
        # print(action_t)
        selected_logprobs = self.get_action_state_value(state_t, action_t).detach() * logprob
        # print("selected_logprobs")
        # print(selected_logprobs)
        loss = selected_logprobs.mean()
        # print("loss")
        # print(loss)
        
        # print("loss\n" + str(loss))
        return loss 
    
    
                
        # action_probs = self.get_action_prob(state_t)
        # print("action_probs")
        # print(action_probs)
        # logprob = torch.log(action_probs)
        # print("logprob")
        # print(logprob)
        # action_state_value = self.get_action_state_value(state_t, action_t).detach()
        # print("action_state_value")
        # print(action_state_value)
        # action_logprob = logprob[np.arange(len(action_t)), action_t]        
        # print("action_logprob")
        # print(action_logprob)
        # selected_logprobs =  action_state_value * action_logprob
        # print("selected_logprobs")
        # print(selected_logprobs)
        
        # loss = -selected_logprobs.mean()
        # # print("loss\n" + str(loss))
        return loss     
    
    
    def calculate_critic_loss(self, state_t, action_t, reward_t, new_states_t, new_legal_actions_t, dones_t):
        
        # print("state_t")
        # print(state_t)
        # print("action_t")
        # print(action_t)
        # print("new_states_t")
        # print(new_states_t)
        qvals = self.get_action_state_value(state_t, action_t)
        
        # print("qvals")
        # print(qvals)
                
        qvals_next = self.get_qvals(new_states_t)  
        # print("qvals_next")
        # print(qvals_next)      
        # print("new_legal_actions_t")
        # print(new_legal_actions_t)      
        qvals_next[new_legal_actions_t] = -100000000
        # print("qvals_next2")
        # print(qvals_next)      
        qvals_next = torch.max(qvals_next,
                               dim=-1)[0].detach()        
        qvals_next[dones_t] = 0
        
                       
        expected_qvals = (self.discount * qvals_next) + reward_t
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))       
        
        return loss     
        
    def load(self, filename):
        
       return
        
                
    def save(self, filename):
        
       return
    
class AC_Builder(object):

    def __init__(self, num_players, versus_agents, state_type, total_episodes, 
                 learning_rate, discount, reference = "", 
                 previous_nn_filename = None, reward_by_win = 0):             
                    
        self.total_episodes = total_episodes
        
        self.num_players = num_players
        
        self.state_type = state_type
        
        self.versus_agents = versus_agents
        self.agents = [] 
        self.discount = discount
        self.reward_by_win = reward_by_win 
        
        batch_size = 256
        memory_size = 1024            
        self.memory = Memory(batch_size, memory_size)
        
        for i in range(num_players - 1):
            self.agents.append(random.choice(versus_agents))    
                        
        chopsticks_phase_mode = state_type.trained_with_chopsticks_phase()
        
        self.env = gym.make('sushi-go-v0', agents = self.agents, state_type = state_type,
        chopsticks_phase_mode = chopsticks_phase_mode, reward_by_win = self.reward_by_win)
        
        action_size = self.env.action_space.n
                
        self.ac_network = ActorCriticNetwork(learning_rate, 0.001, state_type, action_size, discount)
        
        if previous_nn_filename is not None:
            
            previous_episodes = previous_nn_filename.split("_")[-2]
            previous_episodes = int(previous_episodes)
            
            self.ac_network = torch.load(previous_nn_filename + ".pt")
                        
            Q_input = open(previous_nn_filename + ".pkl", 'rb')
            self.state_transf_data = pickle.load(Q_input)
            Q_input.close()
            
        else:
            
            previous_episodes = 0
            self.state_transf_data = None
        
                
        self.filename = "AC_"
        self.filename += str(num_players) + "p_"
        self.filename += state_type.__name__ + "_"
        self.filename += "lr" + str(learning_rate) + "_"
        self.filename += "d" + str(discount) + "_"
        self.filename += "wr" + str(reward_by_win) + "_"
        self.filename += str(previous_episodes + total_episodes) + "_"
        self.filename += reference
        
        
    def set_params(self,params):
        
        self.state_transf_data = params
        
    def save_params(self):
        
        self.state_transf_data.save("params")
    
    def run(self):
              
        rewards = []
            
        batches = []   
        current_batch = BatchInfo()             
        distributions = self.state_type.get_expected_distribution()
                  
        for episode in range(self.total_episodes):
            # print("episode\n" + str(episode))
            for i in range(self.num_players - 1):
                self.agents[i] = random.choice(self.versus_agents) 
            
            state = self.env.reset()
                    
            done = False
            episode_rewards = 0
            
            if episode > 0 and episode % 1000 == 0:
                                                
                print(str(episode) + " episodes.")
                print("Reward: " + str(current_batch.total_reward))
                print("Score: " + str(current_batch.points))
                print("Wins: " + str(current_batch.points_by_victory))
                
                episodes_batch_id = int(episode / 1000)
                batch_filename = self.filename + "-" + str(episodes_batch_id)
                                
                torch.save(self.ac_network, batch_filename + ".pt")
                
                if self.state_transf_data is not None:
                    self.state_transf_data.save(batch_filename)
                                
                batches.append(current_batch)
                current_batch = BatchInfo()
                
                save_batches(batches, batch_filename + "-batches_info.txt")   
                
            while not done:                   
                        
                legal_actions = self.env.action_space.available_actions                   
                                    
                if self.state_transf_data != None:
                                    
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
                    # print("aaaaa\n" + str(np.array(transformed_state)) )
                    action = self.ac_network.get_next_action(np.array(transformed_state), legal_actions)
                else:
                    action = random.choice(legal_actions)
        
                    
                new_state, reward, done, info = self.env.step(action)        
                                                                                      
                new_legal_actions = self.env.action_space.available_actions
                legal_actions_array = np.full(self.env.action_space.n, True)
                
                for legal_action in new_legal_actions:
                    legal_actions_array[legal_action] = False
                                    
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
              
                self.ac_network.update(states_batch, actions_batch, rewards_batch, new_states_batch, new_actions_batch, dones_batch)  
            
                rewards.append(episode_rewards)
                current_batch.total_reward += episode_rewards
                current_batch.points += info['score']                      
                current_batch.points_by_victory += info['points_by_victory']
         
        
        torch.save(self.ac_network, self.filename + ".pt")
        self.state_transf_data.save(self.filename)
        
        batches.append(current_batch)                
        plt.figure(figsize=(36, 48))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.ac_network.losses, label='loss')
        plt.title('Gradient Loss Function Evolution')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.ac_network.state_losses, label='loss')
        plt.title('State Loss Function Evolution')
        plt.legend()
        
        save_batches(batches, self.filename + "-batches_info.txt")  