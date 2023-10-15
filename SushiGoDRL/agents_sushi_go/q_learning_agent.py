# -*- coding: utf-8 -*-


import pickle
import numpy as np

from agents_sushi_go.agent import Agent

from states_sushi_go.table_state import TableState
from states_sushi_go.player_state import PlayerSimplState

class QLearningAgent(Agent):
    
    def __init__(self, player, filename, state_type, learning_rate, discount):
        
        super(QLearningAgent, self).__init__(player, filename)
        
        Q_input = open(filename, 'rb')
        self.q_table = pickle.load(Q_input)
        Q_input.close()
        
        self.state_type = state_type
        self.learning_rate = learning_rate
        self.discount = discount
        
        self.last_action_taken = None
        self.last_state_number = None

    def choose_action(self, legal_actions):
             
        state = self.state_type.build_by_player(self.get_player())
        
        state_number = state.get_state_number()
        
        legal_actions_qtable_values = []
        for action in legal_actions:
            action_number = action.get_action_number()
            legal_actions_qtable_values.append(self.q_table[state_number,action_number])
        
        self.last_action_taken = legal_actions[np.argmax(legal_actions_qtable_values)]
        self.last_state_number = state_number    
        
        return self.last_action_taken
    
    def learn_from_previous_action(self, reward, done, new_legal_actions):
           
        state_number = self.last_state_number
        action = self.last_action_taken.get_action_number()
        
        new_state = self.state_type.build_by_player(self.get_player())        
        new_state_number = new_state.get_state_number()
                        
        new_legal_actions_qtable_values = []
        
        if not done:
            
            for new_action in new_legal_actions:
                new_legal_actions_qtable_values.append(self.q_table[new_state_number,new_action.get_action_number()])
       
            self.q_table[state_number, action] = self.q_table[state_number, action] + self.learning_rate * (reward + self.discount * np.max(new_legal_actions_qtable_values) - self.q_table[state_number, action])
       
        else:
            self.q_table[state_number, action] = self.q_table[state_number, action] + self.learning_rate * (reward - self.q_table[state_number, action])
                        
    
    def save_training(self):
        
        output = open(self.filename, "wb")
        pickle.dump(self.q_table,output)
        output.close()     

class QLearningAgentTest(QLearningAgent):
    
    def __init__(self, player = None):
        
        super(QLearningAgentTest, self).__init__(player, 'Q_values_test.pkl', TableState, 0.3, 1)
    
    
class QLearningAgentTestMulti(QLearningAgent):
    
    def __init__(self, player = None):
        
        super(QLearningAgentTestMulti, self).__init__(player, 'Q_values_test_multi.pkl', TableState, 0.3, 1)

class QLearningAgentPhase1(QLearningAgent):
    
    def __init__(self, player = None):
        
        super(QLearningAgentPhase1, self).__init__(player, 'Q_Learning_2p_PlayerSimplState_lr0.3_dis1_40000_Phase1.pkl', PlayerSimplState, 0.3, 1)
          
class QLearningAgentPhase2(QLearningAgent):
    
    def __init__(self, player = None):
        
        super(QLearningAgentPhase2, self).__init__(player, 'Q_Learning_2p_PlayerSimplState_lr0.3_dis1_40000_Phase2.pkl', PlayerSimplState, 0.3, 1)
         
class QLearningAgentPhase3(QLearningAgent):
    
    def __init__(self, player = None):
        
        super(QLearningAgentPhase3, self).__init__(player, 'Q_Learning_2p_PlayerSimplState_lr0.3_dis1_20000_Phase3.pkl', PlayerSimplState, 0.3, 1)
             
class QLearningAgentPhase4(QLearningAgent):
    
    def __init__(self, player = None):
        
        super(QLearningAgentPhase4, self).__init__(player, 'Q_Learning_2p_PlayerSimplState_lr0.3_dis1_35000_Phase4.pkl', PlayerSimplState, 0.3, 1)
           
