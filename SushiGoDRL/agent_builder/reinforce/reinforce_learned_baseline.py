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

class PolicyNet(torch.nn.Module):

    def __init__(self, learning_rate, state_type, action_size):
        """
        Params
        ======
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        """
        super(PolicyNet, self).__init__()
        
        self.input_shape = state_type.get_number_of_observations()
        self.n_outputs = action_size
                    
        self.learning_rate = learning_rate
        
        self.red_lineal = nn.Sequential(
            torch.nn.Linear(self.input_shape, 256), 
            torch.nn.Tanh(), 
            # torch.nn.Linear(64, 64, bias=True),
            # torch.nn.ReLU(), 
            torch.nn.Linear(256, self.n_outputs),
            torch.nn.Softmax(dim=-1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.losses = []
    
    def get_action_prob(self, state):
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
        
        # print("state_t")
        # print(state_t)
        # print("action_probs")
        # print(self.red_lineal(state_t))
        return self.red_lineal(state_t)
    
    def get_action_probability(self, state, action):

        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
                    
        actions_probabilities = self.red_lineal(state_t)
        
        # print("actions_probabilities")
        # print(actions_probabilities)
        # print("action")
        # print(action.reshape(-1,1))
        return actions_probabilities.gather(1, action.reshape(-1,1))
    
    def get_next_action(self, state, legal_actions):
        
        # print("get_next_action")
        action_probs = self.get_action_prob(state).detach().numpy()
        # print("action_probs")
        # print(action_probs)
        # print("legal_actions")
        # print(legal_actions)
        
        is_legal_actions = np.zeros(self.n_outputs)
        for legal_action in legal_actions:
            is_legal_actions[legal_action] = 1
            
        for action in range(self.n_outputs):
            action_probs[action] = action_probs[action] * is_legal_actions[action]            
            
        # print("action_probs 2")
        # print(action_probs)
        prob_factor = 1 / sum(action_probs)
        action_probs = [prob_factor * p for p in action_probs]
        # print("prob_factor")
        # print(prob_factor)

        # print("action_probs 3")
        # print(action_probs)
        action_space = np.arange(self.n_outputs)
          
        if sum(action_probs) == 1:
            action = np.random.choice(action_space, p=action_probs)
        else:
            action = np.random.choice(legal_actions)
        
        return action

   
    
    def update(self, states_batch, actions_batch, rewards_batch, advantage_batch, has_decided_batch):
        # print("update")
        # print("states_batch")
        # print(states_batch)
        # print("actions_batch")
        # print(actions_batch)
        # print("rewards_batch")
        # print(rewards_batch)
        # print("advantage_batch")
        # print(advantage_batch)
        # print("has_decided_batch")
        # print(has_decided_batch)
        # print("states_batch[has_decided_batch]")
        # print(states_batch[has_decided_batch])
        
        
        state_t = torch.FloatTensor(states_batch[has_decided_batch])
        advantage_t = torch.FloatTensor(advantage_batch[has_decided_batch])
        action_t = torch.LongTensor(actions_batch[has_decided_batch])
        loss = self.calculate_loss(state_t, action_t, advantage_t) 
        # print("loss")
        # print(loss)
        
        self.optimizer.zero_grad() 
        loss.backward()  
        nn.utils.clip_grad_value_(self.red_lineal.parameters(), clip_value=0.5)
        
        # nn.utils.clip_grad_norm_(self.red_lineal.parameters(), max_norm=0.5, norm_type=2)

        self.optimizer.step()
    
        self.losses.append(loss.detach().numpy())
        
              
    def calculate_loss(self, state_t, action_t, reward_t):
        
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
        selected_logprobs = reward_t * logprob
        # print("selected_logprobs")
        # print(selected_logprobs)
        loss = selected_logprobs.mean()
        # print("loss")
        # print(loss)
        
        # print("loss\n" + str(loss))
        return loss 
    
    
    def load(self, filename):
        
       return
        
                
    def save(self, filename):
        
       return
    
    
class LearnedStateNet(torch.nn.Module):

    def __init__(self, state_lr, state_type):
        """
        Params
        ======
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        """
        super(LearnedStateNet, self).__init__()
        
        self.input_shape = state_type.get_number_of_observations()
                    
        self.state_losses = []
        
        self.state_lr = state_lr
        
        self.red_state_value = nn.Sequential(
            torch.nn.Linear(self.input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1))
        
        self.optimizer_state_value = torch.optim.Adam(self.parameters(), lr=state_lr)

    def get_state_value(self, state):
            
    #         print("get_action_value")
    #         print("\tstate")
    #         print(state)
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state)
            
    #         print("\tresult")
    #         print(self.red_critic(state_t))
        return self.red_state_value(state_t)
    
    def update(self, states_batch, rewards_batch):
        # print("update")
        # print(rewards_batch)
        state_t = torch.FloatTensor(states_batch)
        reward_t = torch.FloatTensor(rewards_batch)
        loss = self.calculate_loss_state_value(state_t, reward_t) 
        self.optimizer_state_value.zero_grad() 
        loss.backward() 
        self.optimizer_state_value.step()
         
        
        self.state_losses.append(loss.detach().numpy())
    
    def calculate_loss_state_value(self, state_t, reward_t):
        
        # print("state_t")
        # print(state_t)
        # print("reward_t")
        # print(reward_t)
        
        qvals = self.get_state_value(state_t)
        # print("qvals")
        # print(qvals)
        
        loss = torch.nn.MSELoss()(qvals, reward_t.reshape(-1,1))
        # print("loss")
        # print(loss)
        return loss     
    
    def load(self, filename):
        
       return
        
                
    def save(self, filename):
        
       return
       
        
class PGRLB_Builder(object):

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
            
        self.memory = Memory(0, 4096)
        
        for i in range(num_players - 1):
            self.agents.append(random.choice(versus_agents))    
                        
        chopsticks_phase_mode = state_type.trained_with_chopsticks_phase()
        
        self.env = gym.make('sushi-go-v0', agents = self.agents, state_type = state_type,
        chopsticks_phase_mode = chopsticks_phase_mode, reward_by_win = self.reward_by_win)
        
        action_size = self.env.action_space.n
                
        self.policy_network = PolicyNet(learning_rate, state_type, action_size)
        self.state_network = LearnedStateNet(0.001, state_type)
        
        if previous_nn_filename is not None:
            
            previous_episodes = previous_nn_filename.split("_")[-2]
            previous_episodes = int(previous_episodes)
            
            self.policy_network = torch.load(previous_nn_filename + "_p.pt")
            self.state_network = torch.load(previous_nn_filename + "_s.pt")
            
            Q_input = open(previous_nn_filename + ".pkl", 'rb')
            self.state_transf_data = pickle.load(Q_input)
            Q_input.close()
            
        else:
            
            previous_episodes = 0
            self.state_transf_data = None
        
                
        self.filename = "PG_Reinforce_learned_bl"
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
                
                torch.save(self.policy_network, batch_filename + "_p.pt")
                torch.save(self.state_network, batch_filename + "_s.pt")
                                
                if self.state_transf_data is not None:
                    self.state_transf_data.save(batch_filename)
                                
                batches.append(current_batch)
                current_batch = BatchInfo()
                
                save_batches(batches, batch_filename + "-batches_info.txt")   
                        
                
            trajectory = []       
            state_trajectory = []
            while not done:    
                
                # print("state")                            
                # print(state)      
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
                    # print("transformed_state")                            
                    # print(transformed_state)      
                    action = self.policy_network.get_next_action(np.array(transformed_state), legal_actions)
                else:
                    action = random.choice(legal_actions)
        
                    
                new_state, reward, done, info = self.env.step(action)        
                                
                trajectory.append([state, action, reward, len(legal_actions) > 1])                
                # print("trajectory")
                # print(trajectory)
                if not self.memory.is_full():                    
                    self.memory.add(state)
                               
                episode_rewards += reward
                
                state = new_state
            
            if self.state_transf_data != None:
                                                        
                states_batch = np.array([each[0] for each in trajectory]).astype(float)
                actions_batch = np.array([each[1] for each in trajectory])
                rewards_batch = np.array([each[2] for each in trajectory]) 
                has_decided_batch = np.array([each[3] for each in trajectory]) 
                # print("states_batch")
                # print(states_batch)
                r = 0
                for i in reversed(range(len(rewards_batch))):
                   # compute the return
                   r = rewards_batch[i] + self.discount * r
                   rewards_batch[i] = r
                 
                # print("rewards_batch")
                # print(rewards_batch)
                state_pred = (self.state_network.get_state_value(states_batch).detach().numpy())
                # print("state_pred")
                # print(state_pred)
                advantage_batch = rewards_batch.reshape(-1,1) - state_pred
                
                # print("advantage_batch")
                # print(advantage_batch)
                # print("trajectory" + str(trajectory))
        
                # print("rewards_batch" + str(rewards_batch))
                # print("rewards_batch\n\n" + str(rewards_batch))
                
                distributions = self.state_type.get_expected_distribution()
                
                for i, distribution in enumerate(distributions):
                        
                    state_attribute = states_batch[:,i]
                    
                    if distribution == 'Poisson':
                        
                        # add 1 as all the attributes begin by 0, and is not allowed in the Box-Cox transformation
                        state_attribute = state_attribute + 1                
                        state_attribute = stats.boxcox(state_attribute, self.state_transf_data.fitted_lambdas[i])
                                                                
                    state_attribute -= self.state_transf_data.means[i]
                    state_attribute /= self.state_transf_data.stDevs[i]
                                    
                    states_batch[:,i] = state_attribute                
                # print("aqui\n\n")
                
        
                self.policy_network.update(states_batch, actions_batch, rewards_batch, advantage_batch, has_decided_batch)              
                self.state_network.update(states_batch, rewards_batch)  
            
                rewards.append(episode_rewards)
                current_batch.total_reward += episode_rewards
                current_batch.points += info['score']                      
                current_batch.points_by_victory += info['points_by_victory']

                # print("episode_rewards" + str(episode_rewards))                
            elif self.memory.is_full():            
            
                self.state_transf_data = StateTransformationData(self.state_type)
                
                buffer = self.memory.get_buffer()
                states = np.array(buffer)                
                
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
                        
            trajectory = []
                
        torch.save(self.policy_network, self.filename + "_p.pt")
        torch.save(self.state_network, self.filename + "_s.pt")
        self.state_transf_data.save(self.filename)
        
        batches.append(current_batch)                
        plt.figure(figsize=(36, 48))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.policy_network.losses, label='loss')
        plt.title('Gradient Loss Function Evolution')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.state_network.state_losses, label='loss')
        plt.title('State Loss Function Evolution')
        plt.legend()
        
        save_batches(batches, self.filename + "-batches_info.txt")  