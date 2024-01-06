# -*- coding: utf-8 -*-


import pickle
import random
import numpy as np

from agents_sushi_go.agent import Agent

from states_sushi_go.player_state import PlayerFullState
from states_sushi_go.complete_state_fixed import TwoPlayersCompleteState
from states_sushi_go.game_state import GameState

from tensorflow.keras.models import model_from_json

class DeepQLearningAgent(Agent):
    
    def __init__(self, player, filename, state_type, learning_rate):
        
        # filename = "/kaggle/input/sushigodrl/" + filename    
        
        super(DeepQLearningAgent, self).__init__(player, filename)
        
        json_file = open(filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.q_network = model_from_json(loaded_model_json)
        # load weights into new model
        self.q_network.load_weights(filename + ".h5")
                
        Q_input = open(filename + ".pkl", 'rb')
        self.state_transf_data = pickle.load(Q_input)
        Q_input.close()
        
        self.state_type = state_type
        self.learning_rate = learning_rate
        
        self.last_action_taken = None
        self.last_state = None
        
        self.distributions =  state_type.get_expected_distribution()
        
    def begin_training(self):
        
        # self.q_network.compile(loss='mse', optimizer='adam')
        return

    def choose_action(self, legal_actions, rival_legal_actions = None):
             
        state = self.state_type.build_by_player(self.get_player())
        observation = state.get_as_observation().astype(float)                
        observation = self.standardize_state(observation)
        
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
        self.last_state = adapted_state
        
        return chosen_action
       
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        
        # new_state = self.state_type.build_by_player(self.get_player())
        # new_observation = new_state.get_as_observation()
        # adapted_new_state = new_observation.reshape(-1, len(new_observation)).astype(float)
                
        # new_state = self.standardize_state(adapted_new_state)
        
        # target = self.q_network.predict(self.last_state)[0]     
        # new_states_values = self.q_network.predict(new_state)[0]     
                                      
        # new_value = reward
        
        # if not done:
                                        
        #     new_legal_actions_values = []
            
        #     for new_legal_action in new_legal_actions:
        #         new_legal_actions_values.append(new_states_values[new_legal_action.get_action_number()])            
            
        #     new_value += self.learning_rate * np.amax(new_legal_actions_values)
                        
        # target[self.last_action_taken.get_action_number()] = new_value
            
        # self.q_network.fit(self.last_state, target.reshape(-1, len(target)), epochs=1) 
        return 
    def save_training(self):
        
        model_json = self.q_network.to_json()
        
        with open(self.filename + ".json", "w") as json_file: 
            json_file.write(model_json)
            
        self.q_network.save_weights(self.filename + ".h5")
        
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
    
class DeepQLearningAgentPhase1(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase1, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_26000_Phase1", TwoPlayersCompleteState, 1)
      
    def trained_with_chopsticks_phase(self):
        return False 
    
class DeepQLearningAgentPhase2(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase2, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase2", TwoPlayersCompleteState, 1)
          
    def trained_with_chopsticks_phase(self):
        return False 
class DeepQLearningAgentPhase3(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase3, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_25000_Phase3", TwoPlayersCompleteState, 1)
              
    def trained_with_chopsticks_phase(self):
        return False 
class DeepQLearningAgentPhase4(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DeepQLearningAgentPhase4, self).__init__(player, "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_25000_Phase4", TwoPlayersCompleteState, 1)
          
    def trained_with_chopsticks_phase(self):
        return False 
class DoubleDeepQLearningAgentPhase1(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DoubleDeepQLearningAgentPhase1, self).__init__(player, "Double_Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_23000_Phase1", TwoPlayersCompleteState, 1)
      
    def trained_with_chopsticks_phase(self):
        return False 
class DoubleDeepQLearningAgentPhase2(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DoubleDeepQLearningAgentPhase2, self).__init__(player, "Double_Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase2", TwoPlayersCompleteState, 1)
      
    def trained_with_chopsticks_phase(self):
        return False 
class DoubleDeepQLearningAgentPhase3(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DoubleDeepQLearningAgentPhase3, self).__init__(player, "Double_Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase3", TwoPlayersCompleteState, 1)
      
    def trained_with_chopsticks_phase(self):
        return False 
class DoubleDeepQLearningAgentPhase4(DeepQLearningAgent):
    
    def __init__(self, player = None):
        
        super(DoubleDeepQLearningAgentPhase4, self).__init__(player, "Double_Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Phase4", TwoPlayersCompleteState, 1)
      
    def trained_with_chopsticks_phase(self):
        return False 
