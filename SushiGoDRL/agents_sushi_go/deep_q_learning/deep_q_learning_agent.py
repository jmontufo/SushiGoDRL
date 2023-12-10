# -*- coding: utf-8 -*-


import pickle
import random
import numpy as np

from agents_sushi_go.agent import Agent

from states_sushi_go.player_state import PlayerFullState
from states_sushi_go.complete_state import TwoPlayersCompleteState
from states_sushi_go.game_state import GameState

from tensorflow.keras.models import model_from_json

class DeepQLearningAgent(Agent):
    
    def __init__(self, player, filename, state_type, learning_rate):
        
        super(DeepQLearningAgent, self).__init__(player, filename)
        
        json_file = open(filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.q_network = model_from_json(loaded_model_json)
        # load weights into new model
        self.q_network.load_weights(filename + ".h5")
        
        self.state_type = state_type
        self.learning_rate = learning_rate
        
        self.last_action_taken = None
        self.last_state = None
        
    def begin_training(self):
        
        self.q_network.compile(loss='mse', optimizer='adam')


    def choose_action(self, legal_actions):
             
        state = self.state_type.build_by_player(self.get_player())
        observation = state.get_as_observation()
        adapted_state = observation.reshape(-1, len(observation))
        
        q_values = self.q_network.predict(adapted_state)[0]
    
        legal_actions_values = []
        
        for action in legal_actions:
            action_number = action.get_action_number()
            legal_actions_values.append(q_values[action_number])
            
        max_value = np.amax(legal_actions_values)           
        best_actions = np.argwhere(legal_actions_values == max_value).flatten().tolist()        
        chosen_action = legal_actions[random.choice(best_actions)]
                
        self.last_action_taken = chosen_action
        self.last_state = state
        
        return chosen_action
       
    def learn_from_previous_action(self, reward, done, new_legal_actions):
    
        observation = self.last_state.get_as_observation()
        adapted_state = observation.reshape(-1, len(observation))
    
        new_state = self.state_type.build_by_player(self.get_player())
        new_observation = new_state.get_as_observation()
        adapted_new_state = new_observation.reshape(-1, len(new_observation))
        
        target = self.q_network.predict(adapted_state)[0]     
        new_states_values = self.q_network.predict(adapted_new_state)[0]     
                                      
        new_value = reward
        
        if not done:
                                        
            new_legal_actions_values = []
            
            for new_legal_action in new_legal_actions:
                new_legal_actions_values.append(new_states_values[new_legal_action.get_action_number()])            
            
            new_value += self.learning_rate * np.amax(new_legal_actions_values)
                        
        target[self.last_action_taken.get_action_number()] = new_value
            
        self.q_network.fit(adapted_state, target.reshape(-1, len(target)), epochs=1) 
    
    def save_training(self):
        
        model_json = self.q_network.to_json()
        
        with open(self.filename + ".json", "w") as json_file: 
            json_file.write(model_json)
            
        self.q_network.save_weights(self.filename + ".h5")
        
    
class DeepQLearningAgentPhase1(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase1, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_26000_Phase1", TwoPlayersCompleteState, 1)
        
class DeepQLearningAgentPhase2(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase2, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase2", TwoPlayersCompleteState, 1)
 
class DeepQLearningAgentPhase3(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase3, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase3", TwoPlayersCompleteState, 1)
        
class DoubleDeepQLearningAgentPhase1(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DoubleDeepQLearningAgentPhase1, self).__init__(player, "Double_Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase1", TwoPlayersCompleteState, 1)
    
