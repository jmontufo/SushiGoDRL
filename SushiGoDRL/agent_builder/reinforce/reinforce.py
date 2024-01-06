# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
from scipy import stats
import random
import pickle
import gym

from agent_builder.utils import BatchInfo, save_batches

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

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
# from agents_sushi_go.q_learning.q_learning_agent import QLearningAgentPhase1
# from agents_sushi_go.q_learning.q_learning_agent import QLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent import DoubleDeepQLearningAgentPhase1
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
# deregisterCustomGym('sushi-go-v0')
# deregisterCustomGym('sushi-go-multi-v0')
# import gym_sushi_go     

num_players = 2
versus_agents = [RandomAgent()]

state_type = CompleteState.get_type_from_num_players(num_players)
total_episodes = 20000
learning_rate = 1     
max_epsilon = 1             
min_epsilon = 0.05            
decay_rate = 0.00012   
        
reference = "Prueba3"     

previous_nn_filename = None 

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

class PGNetwork(object):
    
    def __init__(self, learning_rate, state_type, action_size, discount):
        
        self.__learning_rate = learning_rate
        self.__discount = discount
        self.__state_type = state_type

        self.__state_size = state_type.get_number_of_observations()
        self.__action_size = action_size
        
        self.__pg_network = self.__define_model()
        # self.losses = []
        

    def __define_model(self):
        
        model_input = Input(shape=(self.__state_size,), name='model_input')
        x = Dense(32, activation='relu')(model_input)
        x = Dense(32, activation='relu')(x)
        model_output = Dense(self.__action_size, activation='softmax' , name='model_output')(x)
        model = Model(model_input, model_output)
        self.optimizer = Adam(self.__learning_rate)
     
        
        print(model.summary())
        
        return model
       

    def __get_actions_probs(self, state):
        
        adapted_state = state[np.newaxis, :]
            
        return self.__pg_network.predict(adapted_state)[0]
    
    
    def get_next_action(self, state, legal_actions):       
        
        action_probs = self.__get_actions_probs(state)
        
        non_legal_actions = np.zeros(self.__action_size)
        for legal_action in legal_actions:
            non_legal_actions[legal_action] = 1
            
        for action in range(self.__action_size):
            action_probs[action] = action_probs[action] * non_legal_actions[action]            
            
        prob_factor = 1 / sum(action_probs)
        action_probs = [prob_factor * p for p in action_probs]

        action_space = np.arange(self.__action_size)
                
        action = np.random.choice(action_space, p=action_probs)
        
        return action
        
    
    def update(self, states_batch, actions_batch, rewards_batch):
        print(len(actions_batch))
        states = np.array(states_batch)
        actions = np.array(actions_batch)
        rewards = np.array(rewards_batch)
        
        mean = rewards.mean()
        std = rewards.std() if rewards.std() > 0 else 1
        rewards = (rewards - mean) / std
        
        with tf.GradientTape() as tape:
            neg_log_probs = sparse_categorical_crossentropy(y_true=actions, y_pred=self.__pg_network(states))
            loss = neg_log_probs * rewards
            
        gradients = tape.gradient(loss, self.__pg_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.__pg_network.trainable_variables))
           
        # self.losses.append(loss.numpy().mean())                
        
    def load(self, filename):
        
        json_file = open(filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        self.__pg_network = model_from_json(loaded_model_json)
        self.__pg_network.load_weights(filename + ".h5")     
        
        self.optimizer = Adam(self.__learning_rate)
        return
                
    def save(self, filename):
        
        model_json = self.__pg_network.to_json()
        
        with open(filename + ".json", "w") as json_file: 
            json_file.write(model_json)
            
        self.__pg_network.save_weights(filename + ".h5")
        return 
        
 
class PG_Reinforce_Builder(object):

    def __init__(self, num_players, versus_agents, state_type, total_episodes, 
                 learning_rate, discount, reference = "", previous_nn_filename = None):             
                    
        self.total_episodes = total_episodes
        
        self.num_players = num_players
        
        self.state_type = state_type
        
        self.versus_agents = versus_agents
        self.agents = [] 
        self.discount = discount
            
        self.memory = Memory(0, 1024)
        
        for i in range(num_players - 1):
            self.agents.append(random.choice(versus_agents))    
                        
        chopsticks_phase_mode = state_type.trained_with_chopsticks_phase()
        
        self.env = gym.make('sushi-go-v0', agents = self.agents, state_type = state_type,
        chopsticks_phase_mode = chopsticks_phase_mode)
        
        action_size = self.env.action_space.n
                
        self.pg_network = PGNetwork(learning_rate, state_type, action_size, 
                                    discount)
        
        if previous_nn_filename is not None:
            
            previous_episodes = previous_nn_filename.split("_")[-2]
            previous_episodes = int(previous_episodes)
            
            self.pg_network.load(previous_nn_filename)
            
            Q_input = open(previous_nn_filename + ".pkl", 'rb')
            self.state_transf_data = pickle.load(Q_input)
            Q_input.close()
            
        else:
            
            previous_episodes = 0
            self.state_transf_data = None
        
                
        self.filename = "PG_Reinforce_keras"
        self.filename += str(num_players) + "p_"
        self.filename += state_type.__name__ + "_"
        self.filename += "lr" + str(learning_rate) + "_"
        self.filename += "d" + str(discount) + "_"
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
            
            for i in range(self.num_players - 1):
                self.agents[i] = random.choice(self.versus_agents) 
            
            state = self.env.reset()
                    
            done = False
            episode_rewards = 0
            
            if episode > 0 and episode % 1000 == 0:
                                                
                print(str(episode) + " episodes.")
                print("Reward: " + str(current_batch.total_reward))
                
                episodes_batch_id = int(episode / 1000)
                batch_filename = self.filename + "-" + str(episodes_batch_id)
                
                self.pg_network.save(batch_filename)  
                if self.state_transf_data is not None:
                    self.state_transf_data.save(batch_filename)
                                
                batches.append(current_batch)
                current_batch = BatchInfo()
                
                save_batches(batches, batch_filename + "-batches_info.txt")   
                        
                
            trajectory = []           
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
                             
                    action = self.pg_network.get_next_action(np.array(transformed_state), legal_actions)
                else:
                    action = random.choice(legal_actions)
        
                    
                new_state, reward, done, info = self.env.step(action)        
                                                                                      
                trajectory.append([state, action, reward])
                
                if not self.memory.is_full():                    
                    self.memory.add(state)
                               
                episode_rewards += reward
                
                state = new_state
            
            if self.state_transf_data != None:
                                                        
                states_batch = np.array([each[0] for each in trajectory]).astype(float)
                actions_batch = np.array([each[1] for each in trajectory])
                rewards_batch = np.array([each[2] for each in trajectory]) 
                
                r = 0
                for i in reversed(range(len(rewards_batch))):
                   # compute the return
                   r = rewards_batch[i] + self.discount * r
                   rewards_batch[i] = r
                   
                # print("trajectory" + str(trajectory))
                
                # print("rewards_batch" + str(rewards_batch))
                
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
                
                self.pg_network.update(states_batch, actions_batch, rewards_batch)  
            
                rewards.append(episode_rewards)
                current_batch.total_reward += episode_rewards

                print("episode_rewards" + str(episode_rewards))                
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
        
        self.pg_network.save(self.filename) 
        self.state_transf_data.save(self.filename)
        
        batches.append(current_batch)                
        # plt.figure(figsize=(36, 48))
        
        # plt.subplot(2, 1, 1)
        # plt.plot(self.pg_network.losses, label='loss')
        # plt.title('Gradient Loss Function Evolution')
        # plt.legend()
        # save_batches(batches, self.filename + "-batches_info.txt")   
                             
