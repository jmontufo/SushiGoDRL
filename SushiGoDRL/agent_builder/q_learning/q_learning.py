# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
import random
import pickle

from agent_builder.utils import *

from states_sushi_go.player_state import PlayerSimplState
from states_sushi_go.with_phase.player_state import PlayerSimplWithPhaseState
from states_sushi_go.game_state import GameState

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
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase1
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase2
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase3
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase3
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase3
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3

import gym
from gym_sushi_go.envs.single_env import SushiGoEnv

num_players = 2

versus_agents = [
                RandomAgent(),
                ChopstickLoverAgent(),
                ChopstickHaterAgent(),
                ChopstickLoverAtFirstAgent(),
                DumplingLoverAgent(),
                MakiLoverAgent(),
                MakiHaterAgent(),
                NigiriLoverAgent(),
                PuddingLoverAgent(),
                PuddingHaterAgent(),
                SashimiLoverAgent(),
                SashimiHaterAgent(),
                TempuraLoverAgent(),
                WasabiLoverAgent(),
                WasabiLoverAtFirstAgent(),
                # DumplingSuperLoverAgent(),
                # MakiSuperLoverAgent(),
                # NigiriSuperLoverAgent(),
                # PuddingSuperLoverAgent(),
                # SashimiSuperLoverAgent(),
                # TempuraSuperLoverAgent(),
                # QLearningAgentPhase1(),
                # MCTreeSearchAgentPhase1(),
                # DeepQLearningAgentPhase1(),
                # DoubleDeepQLearningAgentPhase1(),
                # QLearningAgentPhase2(),
                # MCTreeSearchAgentPhase2(),
                # DeepQLearningAgentPhase2(),
                # DoubleDeepQLearningAgentPhase2(),
                # QLearningAgentPhase3(),
                # MCTreeSearchAgentPhase3(),
                # DeepQLearningAgentPhase3(),
                # DoubleDeepQLearningAgentPhase3()
                ]

state_type = PlayerSimplWithPhaseState
total_episodes = 10000 
max_epsilon = 1             
min_epsilon = 0.05            
decay_rate = 0.0005       
learning_rate = 0.7           
discount = 1           
reference = "Phase4_test"      
reward_by_win = 0
chopsticks_phase_mode = True

previous_table_filename = None
# previous_table_filename = "Q_Learning_2p_PlayerSimplState_lr0.3_dis1_25000_Phase4_test.pkl" 


class QL_Builder(object):
    
    def __init__(self, num_players, versus_agents, state_type, total_episodes, 
                 max_epsilon, min_epsilon, decay_rate, learning_rate, discount, 
                 reference, previous_table_filename = None):
        
        self.reference = reference
        
        self.num_players = num_players
        
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        self.learning_rate = learning_rate
        self.discount = discount
        
        self.total_episodes = total_episodes
        
        self.versus_agents = versus_agents
        self.agents = [] 
        
        for i in range(num_players - 1):
            self.agents.append(random.choice(versus_agents))  
            
        self.env = gym.make('sushi-go-v0', agents = self.agents, 
                            state_type = state_type, reward_by_win = reward_by_win,
                            chopsticks_phase_mode = chopsticks_phase_mode)
        
        self.action_size = self.env.action_space.n
        self.state_size = self.env.state_type.get_total_numbers()
        
        if previous_table_filename is not None:
            previous_episodes = previous_table_filename[:-4]
            previous_episodes = previous_episodes.split("_")[-2]
            previous_episodes = int(previous_episodes)
            
            Q_input = open(previous_table_filename, 'rb')
            self.qtable = pickle.load(Q_input)
            Q_input.close()
            
        else:
            
            previous_episodes = 0
            self.qtable = np.zeros((self.state_size, self.action_size))            
        
        self.filename = "Q_Learning_"
        self.filename += str(num_players) + "p_"
        self.filename += state_type.__name__ + "_"
        self.filename += "lr" + str(learning_rate) + "_"
        self.filename += "dis" + str(discount) + "_"
        self.filename += str(previous_episodes + total_episodes) + "_"
        self.filename += reference
        
    def run(self):      
        
        epsilon = self.max_epsilon           
        
        n_table = np.zeros((self.state_size, self.action_size))
            
        batches = []
        current_batch = BatchInfo()
        
        for episode in range(self.total_episodes):
            
            for i in range(self.num_players - 1):
                self.agents[i] = random.choice(self.versus_agents)    
                
            state = self.env.reset()                
            
            if episode % 1000 == 0 and episode > 0:
                                
                current_batch.epsilon_at_end = epsilon
                
                print(str(episode) + " episodes.")
                print("Reward: " + str(current_batch.total_reward))
                print("Epsilon: " + str(current_batch.epsilon_at_end))
                
                episodes_batch_id = int(episode / 1000)                    
                batch_filename = self.filename + "-" + str(episodes_batch_id)   
                
                self.save_qtable(batch_filename)
                
                batches.append(current_batch)                
                current_batch = BatchInfo()
              
            done = False
            episode_rewards = 0  
            
            while not done:
                
                state_object = self.env.state_type.build_by_observation(state)
                state_number = state_object.get_state_number()
            
                exp_exp_tradeoff = random.uniform(0, 1)
                
                legal_actions = self.env.action_space.available_actions       
                        
                if exp_exp_tradeoff > epsilon:
                    
                    legal_actions_qtable_values = []
                    
                    for action in legal_actions:
                        legal_actions_qtable_values.append(self.qtable[state_number,action])
                        
                    best_actions = np.argwhere(legal_actions_qtable_values == np.amax(legal_actions_qtable_values)).flatten().tolist()
                    action = legal_actions[random.choice(best_actions)]
                                    
                    current_batch.times_explotation += 1
        
                else:
                    
                    action = self.env.action_space.sample()
                    
                    current_batch.times_exploration += 1
                    
        
                new_state, reward, done, info = self.env.step(action)
                
                new_state_object = self.env.state_type.build_by_observation(new_state)
                new_state_number = new_state_object.get_state_number()
                                
                new_legal_actions = self.env.action_space.available_actions
                new_legal_actions_qtable_values = []
                
                if not done:
                    
                    for new_action in new_legal_actions:
                        new_legal_actions_qtable_values.append(self.qtable[new_state_number,new_action])
               
                    self.qtable[state_number, action] = self.qtable[state_number, action] + self.learning_rate * (reward + self.discount * np.max(new_legal_actions_qtable_values) - self.qtable[state_number, action])
               
                else:
                    self.qtable[state_number, action] = self.qtable[state_number, action] + self.learning_rate * (reward - self.qtable[state_number, action])
                        
                n_table[state_number, action] += 1
                
                episode_rewards += reward
                
                state = new_state
                
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode) 
                     
            current_batch.total_reward += episode_rewards
                
    
        current_batch.epsilon_at_end = epsilon    
        batches.append(current_batch)       
        
        print(str(self.total_episodes) + " episodes.")
        print("Reward: " + str(current_batch.total_reward))
        print("Epsilon: " + str(current_batch.epsilon_at_end))        
        
        self.save_qtable(self.filename)
        QL_Builder.save_as_text(self.qtable, self.filename)        
        QL_Builder.save_as_text(n_table, self.filename + "-counter.txt")
        save_batches(batches, self.filename + "-batches_info.txt")       
               
    def save_qtable(self, filename):
        
        output = open(filename + ".pkl", "wb")
        pickle.dump(self.qtable,output)
        output.close()
     
    def save_as_text(table, filename):
        
        f = open(filename + ".txt", "w")
        for row in table:
            for value in row:
                f.write("{:.2f}".format(value) + "\t")
            f.write("\n")
        f.close()
        
        
builder = QL_Builder(num_players, versus_agents, state_type, total_episodes, 
                 max_epsilon, min_epsilon, decay_rate, learning_rate, discount, 
                 reference, previous_table_filename)

builder.run()