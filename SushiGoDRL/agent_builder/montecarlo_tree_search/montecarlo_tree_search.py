# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
import random
import pickle
from collections import deque

from agent_builder.utils import *

from states_sushi_go.game_state import GameState
from states_sushi_go.player_state import PlayerSimplState
from states_sushi_go.player_state import PlayerSimpleStateV2
from states_sushi_go.player_state import PlayerMidStateV2

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
from agents_sushi_go.q_learning_agent import QLearningAgentPhase3
from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase3
from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase3
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3
            
deregisterCustomGym('sushi-go-v0')
deregisterCustomGym('sushi-go-multi-v0')
import gym_sushi_go     
        
num_players = 2

versus_agents = [
                RandomAgent(),
                ChopstickLoverAgent(),
                ChopstickHaterAgent(),
                ChopstickLoverAtFirstAgent(),
                DumplingLoverAgent(),
                DumplingSuperLoverAgent(),
                MakiLoverAgent(),
                MakiSuperLoverAgent(),
                MakiHaterAgent(),
                NigiriLoverAgent(),
                NigiriSuperLoverAgent(),
                PuddingLoverAgent(),
                PuddingSuperLoverAgent(),
                PuddingHaterAgent(),
                SashimiLoverAgent(),
                SashimiSuperLoverAgent(),
                SashimiHaterAgent(),
                TempuraLoverAgent(),
                TempuraSuperLoverAgent(),
                WasabiLoverAgent(),
                WasabiLoverAtFirstAgent(),
                QLearningAgentPhase1(),
                MCTreeSearchAgentPhase1(),
                DeepQLearningAgentPhase1(),
                DoubleDeepQLearningAgentPhase1(),
                QLearningAgentPhase2(),
                MCTreeSearchAgentPhase2(),
                DeepQLearningAgentPhase2(),
                DoubleDeepQLearningAgentPhase2(),
                QLearningAgentPhase3(),
                MCTreeSearchAgentPhase3(),
                DeepQLearningAgentPhase3(),
                DoubleDeepQLearningAgentPhase3()
                ]

state_type = PlayerSimplState
total_episodes = 20000
epsilon = 2.1     
learning_rate = None               
reference = "Phase4"                     

previous_tree_filename = None 
#previous_tree_filename = "MonteCarlo_Tree_Search_2p_PlayerSimplState_eps2.1_20000_Phase3.pkl"

class MCTS_Builder(object):
    
    # mc_tree
    # {
    #      state 0:
    #           actions
    #           {
    #               action 1: (value, n)
    #               action 2: (value, n)
    #               ...
    #           },
    #           n --> Occurrences of the state
    #      state 3:
    #           actions
    #           {
    #               action 2: (value, n)
    #               action 4: (value, n)
    #               ...
    #           },
    #           n --> Occurrences of the state
    #      ....
    # }
    
    def __init__(self, num_players, agents_types, state_type, total_episodes, 
                 epsilon, learning_rate, reference, previous_tree_filename = None):
                
        self.reference = reference
        
        self.num_players = num_players
        
        self.epsilon = epsilon
        
        self.learning_rate = learning_rate
                
        self.total_episodes = total_episodes
        
        self.versus_agents = versus_agents
        self.agents = [] 
        
        for i in range(num_players - 1):
            self.agents.append(random.choice(versus_agents))  
            
        self.env = gym.make('sushi-go-v0', agents = self.agents, 
                            state_type = state_type)
        
        self.action_size = self.env.action_space.n
        self.state_size = self.env.state_type.get_total_numbers()
        
        if previous_tree_filename is not None:
            previous_episodes = previous_tree_filename[:-4]
            previous_episodes = previous_episodes.split("_")[-2]
            previous_episodes = int(previous_episodes)
            
            mc_input = open(previous_tree_filename, 'rb')
            self.mc_tree = pickle.load(mc_input)
            mc_input.close()
            
        else:
            
            previous_episodes = 0
            self.mc_tree = dict()          
        
        self.filename = "MonteCarlo_Tree_Search_"
        self.filename += str(num_players) + "p_"
        self.filename += state_type.__name__ + "_"
        self.filename += "eps" + str(epsilon) + "_"
        if learning_rate is not None:            
            self.filename += "lr" + str(learning_rate) + "_"
        self.filename += str(previous_episodes + total_episodes) + "_"
        self.filename += reference
      
    def run(self):
                   
        MAX_INT = 10000000
        
        batches = []
        current_batch = BatchInfo()
        
        for episode in range(self.total_episodes):        
                        
            for i in range(self.num_players - 1):
                self.agents[i] = random.choice(versus_agents)        
            
            state = self.env.reset()                
            
            if episode % 1000 == 0 and episode > 0:                
                
                print(str(episode) + " episodes.")
                print("Reward: " + str(current_batch.total_reward))
                
                episodes_batch_id = int(episode / 1000)                    
                batch_filename = self.filename + "-" + str(episodes_batch_id)   
                
                self.save_tree(batch_filename)
                
                batches.append(current_batch)
                current_batch = BatchInfo()                
                
            done = False
            episode_rewards = 0
            episode_log = deque()
            
            while not done:
                
                state_object = state_type.build_by_observation(state)
                state_number = state_object.get_state_number()
                
                state_node = [dict(), 0]
                
                if state_number not in self.mc_tree:
                    self.mc_tree[state_number] = state_node
                else:
                    state_node = self.mc_tree[state_number]
                    
                state_action_nodes = state_node[0]
                state_n = state_node[1]
            
                legal_actions = self.env.action_space.available_actions   
                
                actions_epsilon = []
                actions_values = []
                
                for legal_action in legal_actions:
                    
                    if legal_action in state_action_nodes:
                        
                        action_node = state_action_nodes[legal_action]
                        action_value = action_node[0]
                        action_n = action_node[1]
                        
                        a = np.math.log(state_n)
                        b = a / action_n
                        c = np.math.sqrt(b)
                        
                        exploration_rate = self.epsilon * c
                        
                        actions_epsilon.append(action_value + exploration_rate)
                        
                        actions_values.append(action_value)
                        
                    else:
                        
                        actions_epsilon.append(MAX_INT)
                        actions_values.append(-1*MAX_INT)
                    
                best_epsilons = np.argwhere(actions_epsilon == np.amax(actions_epsilon)).flatten().tolist()
                
                chosen_epsilon = random.choice(best_epsilons)            
                action = legal_actions[chosen_epsilon]
                
                if actions_values[chosen_epsilon] > -1 * MAX_INT and actions_values[chosen_epsilon] == np.amax(actions_values):
                    current_batch.times_explotation += 1
                else:            
                    current_batch.times_exploration += 1
                
                if action not in state_action_nodes:
                    state_action_nodes[action] = [0,0]
                    
                new_state, reward, done, info = self.env.step(action)
                                        
                episode_log.append((state_number, action, reward))       
                
                episode_rewards += reward                
                state = new_state
                
                if done == True: 
                    
                    accumulated_reward = 0
                    
                    while len(episode_log) > 0:
                        
                        game_turn = episode_log.pop()
                        
                        state = game_turn[0]
                        action = game_turn[1]
                        reward = game_turn[2]
                                        
                        accumulated_reward += reward 
                        
                        state_node = self.mc_tree[state]
                        action_node = state_node[0][action]
                        
                        state_node[1] += 1
                        action_node[1] += 1
                        
                        if self.learning_rate == None:
                            alpha = 1 / action_node[1]
                        else:
                            alpha = self.learning_rate
                            
                        action_node[0] += (alpha * (accumulated_reward - action_node[0]))
                    
                    break
                
            current_batch.total_reward += episode_rewards        
        
        batches.append(current_batch)
                
        self.save_tree(self.filename)        
        save_batches(batches, self.filename + "-batches_info.txt")       
    
    def save_tree(self, filename):
        
        output = open(filename + ".pkl", "wb")
        pickle.dump(self.mc_tree,output)
        output.close()


builder = MCTS_Builder(num_players, versus_agents, state_type, total_episodes, 
                       epsilon, learning_rate, reference, previous_tree_filename)

builder.run()