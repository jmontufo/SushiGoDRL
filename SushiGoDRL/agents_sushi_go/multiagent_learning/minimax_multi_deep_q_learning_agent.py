# -*- coding: utf-8 -*-


import pickle
import random
import numpy as np

from agents_sushi_go.agent import Agent

from states_sushi_go.player_state import PlayerFullState
from states_sushi_go.complete_state import TwoPlayersCompleteState
from states_sushi_go.game_state import GameState
from states_sushi_go.with_phase.complete_state import CompleteMemoryState

import torch
import torch.nn as nn

class MinimaxMultiDQLearningAgent(Agent):
    
    def __init__(self, player, filename, state_type, learning_rate, winner_num):
        
        super(MinimaxMultiDQLearningAgent, self).__init__(player, filename)
                
        self.dq_network = torch.load(filename + ".pt")
        self.winner_num = winner_num
        
        self.state_type = state_type
        self.learning_rate = learning_rate
        
        Q_input = open(filename + ".pkl", 'rb')
        self.state_transf_data = pickle.load(Q_input)
        Q_input.close()
        
        
        self.last_action_taken = None
        self.last_state = None
        
        self.distributions =  state_type.get_expected_distribution()
        
    # def begin_training(self):
        
    #     self.q_network.compile(loss='mse', optimizer='adam')


    def choose_action(self, legal_actions, rival_legal_actions):
             
        state = self.state_type.build_by_player(self.get_player())
        observation = state.get_as_observation()            
        observation = self.standardize_state(observation)
        
        adapted_state = observation.reshape(-1, len(observation))
        
        # print("adapted_state")
        # print(adapted_state)
        action = self.dq_network[self.winner_num].get_next_action(np.array(adapted_state), 
                                                   legal_actions, 
                                                   rival_legal_actions)
    
                                       
        self.last_action_taken = action
        self.last_state = state
        
        return action
    
    def standardize_state(self, state):
        
        transformed_state = []
        
        for i, distribution in enumerate(self.distributions):
                
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
            
        return np.array(transformed_state)        
       
    def learn_from_previous_action(self, reward, done, new_legal_actions):
    
    #     observation = self.last_state.get_as_observation()
    #     adapted_state = observation.reshape(-1, len(observation))
    
    #     new_state = self.state_type.build_by_player(self.get_player())
    #     new_observation = new_state.get_as_observation()
    #     adapted_new_state = new_observation.reshape(-1, len(new_observation))
        
    #     target = self.q_network.predict(adapted_state)[0]     
    #     new_states_values = self.q_network.predict(adapted_new_state)[0]     
                                      
    #     new_value = reward
        
    #     if not done:
                                        
    #         new_legal_actions_values = []
            
    #         for new_legal_action in new_legal_actions:
    #             new_legal_actions_values.append(new_states_values[new_legal_action.get_action_number()])            
            
    #         new_value += self.learning_rate * np.amax(new_legal_actions_values)
                        
    #     target[self.last_action_taken.get_action_number()] = new_value
            
    #     self.q_network.fit(adapted_state, target.reshape(-1, len(target)), epochs=1) 
        return
    
    def save_training(self):
        
    #     model_json = self.q_network.to_json()
        
    #     with open(self.filename + ".json", "w") as json_file: 
    #         json_file.write(model_json)
            
    #     self.q_network.save_weights(self.filename + ".h5")
        return
        
    
class DeepQLearningTorchAgentPhase1(DeepQLearningTorchAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningTorchAgentPhase1, self).__init__(player, "DQLt_2p_CompleteMemoryState_lr0.001_d0.95_wr-1_20000_Phase1-18", CompleteMemoryState, 0.0005)
      
    def trained_with_chopsticks_phase(self):
        return True  
# class DeepQLearningAgentPhase2(DeepQLearningAgent):
    
#     def __init__(self, player = None):
        
#         super(DeepQLearningAgentPhase2, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase2", TwoPlayersCompleteState, 1)
 
# class DeepQLearningAgentPhase3(DeepQLearningAgent):
    
#     def __init__(self, player = None):
        
#         super(DeepQLearningAgentPhase3, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase3", TwoPlayersCompleteState, 1)
        
# class DoubleDeepQLearningAgentPhase1(DeepQLearningAgent):
    
#     def __init__(self, player = None):
        
#         super(DoubleDeepQLearningAgentPhase1, self).__init__(player, "Double_Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase1", TwoPlayersCompleteState, 1)
    
