# -*- coding: utf-8 -*-


import pickle
import numpy as np
import random
from collections import deque

from agents_sushi_go.agent import Agent

from states_sushi_go.game_state import GameState
from states_sushi_go.player_state import PlayerSimplState
from states_sushi_go.player_state import PlayerFullState
from states_sushi_go.player_state import PlayerFullStateV2

class MCTreeSearchAgent(Agent):
    
    def __init__(self, player, filename, state_type, learning_rate = None):
        
        super(MCTreeSearchAgent, self).__init__(player, filename)
        
        mc_input = open(filename, 'rb')
        self.mc_tree = pickle.load(mc_input)
        mc_input.close()
        
        self.state_type = state_type
        self.learning_rate = learning_rate
                
        self.episode_log = deque()
        self.rewards_log = deque()

    def choose_action(self, legal_actions):
             
        state = self.state_type.build_by_player(self.get_player())
        
        state_number = state.get_state_number()
                
        if state_number not in self.mc_tree:
            return random.choice(legal_actions)
        else:
            state_node = self.mc_tree[state_number]
            
        state_action_nodes = state_node[0]
            
        actions_value = []
                
        for legal_action in legal_actions:
            
            legal_action_number = legal_action.get_action_number()
            
            if legal_action_number in state_action_nodes:
                
                action_node = state_action_nodes[legal_action_number]
                action_value = action_node[0]
                
                actions_value.append(action_value)
                
            else:
                actions_value.append(0)
                        
        better_actions = np.argwhere(actions_value == np.amax(actions_value)).flatten().tolist()
            
        action = legal_actions[random.choice(better_actions)]
        self.episode_log.append((state_number, action))
        
        return action
    
    def learn_from_previous_action(self, reward, done, new_legal_actions = None):
        
        self.rewards_log.append(reward)
        
        if done:
            
            accumulated_reward = 0
            
            while len(self.episode_log) > 0:
                
                game_turn = self.episode_log.pop()
                
                state = game_turn[0]
                action = game_turn[1].get_action_number()
                reward = self.rewards_log.pop()
                                
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
                    
    
    def save_training(self):
        
        output = open(self.filename, "wb")
        pickle.dump(self.mc_tree,output)
        output.close()
    

class MCTreeSearchAgentPhase1(MCTreeSearchAgent):
    
    def __init__(self, player = None):
        
        super(MCTreeSearchAgentPhase1, self).__init__(player, 'MonteCarlo_Tree_Search_2p_PlayerSimplState_eps2.5_20000_Phase1.pkl', PlayerSimplState)

class MCTreeSearchAgentPhase2(MCTreeSearchAgent):
    
    def __init__(self, player = None):
        
        super(MCTreeSearchAgentPhase2, self).__init__(player, 'MonteCarlo_Tree_Search_2p_PlayerSimplState_eps2.1_40000_Phase2.pkl', PlayerSimplState)

class MCTreeSearchAgentPhase3(MCTreeSearchAgent):
    
    def __init__(self, player = None):
        
        super(MCTreeSearchAgentPhase3, self).__init__(player, 'MonteCarlo_Tree_Search_2p_PlayerSimplState_eps2.1_20000_Phase3.pkl', PlayerSimplState)

class MCTreeSearchAgentPhase4(MCTreeSearchAgent):
    
    def __init__(self, player = None):
        
        super(MCTreeSearchAgentPhase4, self).__init__(player, 'MonteCarlo_Tree_Search_2p_PlayerSimplState_eps2.1_20000_Phase4.pkl', PlayerSimplState)
