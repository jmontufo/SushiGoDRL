# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:23:32 2023

@author: jmont
"""

import numpy as np


from states_sushi_go.state import State
from states_sushi_go.with_phase.player_state import PlayerCompleteState, OtherPlayerCompleteState
from states_sushi_go.table_state import TableFullState

# PlayerCompleteState
# OtherPlayerCompleteState
# OtherPlayerCompleteState
class CompleteMemoryState(State):    
    
    def __init__(self):         

        self.__player_state = None
        self.__previous_player_state = None
        self.__next_player_state = None
        
    def get_player_state(self):
        
        return self.__player_state
        
    def __set_player_state(self, player_state):
        
        self.__player_state = player_state   
        
    def get_previous_player_state(self):
        
        return self.__previous_player_state
        
    def __set_previous_player_state(self, previous_player_state):
        
        self.__previous_player_state = previous_player_state   
        
    def get_next_player_state(self):
        
        return self.__next_player_state
        
    def __set_next_player_state(self, next_player_state):
        
        self.__next_player_state = next_player_state   
        
    def __get_values_from_player(self, player):
        
        game = player.get_game()
        
        player_position = player.get_position()
        next_player_position = (player_position + 1) % game.get_num_players()
        previous_player_position = (player_position - 1) % game.get_num_players()
        next_player = game.get_player(next_player_position)
        previous_player = game.get_player(previous_player_position)
        
        self.__set_player_state(PlayerCompleteState.build_by_player(player)) 
                   
        if player.can_remember_cards_of_position(next_player_position):
            next_player_state = OtherPlayerCompleteState.build_by_player(next_player)
        else:
            next_player_state = OtherPlayerCompleteState.build_when_unknown_hand(next_player)
        self.__set_next_player_state(next_player_state)
            
        if player.can_remember_cards_of_position(previous_player_position):
            previous_player_state = OtherPlayerCompleteState.build_by_player(previous_player)
        else:
            previous_player_state = OtherPlayerCompleteState.build_when_unknown_hand(previous_player)          
        self.__set_previous_player_state(previous_player_state)
                               
    def __get_values_from_observation(self, observation):
        
        pos_ini = 0
        pos = PlayerCompleteState.get_number_of_observations()
        
        self.__set_player_state(PlayerCompleteState.build_by_observation(observation[pos_ini:pos]))
                    
        pos_ini = pos
        pos = pos_ini + OtherPlayerCompleteState.get_number_of_observations()
                        
        next_player_state = OtherPlayerCompleteState.build_by_observation(observation[pos_ini:pos])            
        self.__set_next_player_state(next_player_state)
                    
        pos_ini = pos
        pos = pos_ini + OtherPlayerCompleteState.get_number_of_observations()
                        
        previous_player_state = OtherPlayerCompleteState.build_by_observation(observation[pos_ini:pos])            
        self.__set_previous_player_state(previous_player_state)
        
    def get_as_observation(self):
               
        player_observation = self.get_player_state().get_as_observation()   
        next_player_observation = self.get_next_player_state().get_as_observation() 
        previous_player_observation = self.get_previous_player_state().get_as_observation()   
        
        return np.concatenate((player_observation, next_player_observation, previous_player_observation))     
    
    def build_by_player(player):
                
        complete_state = CompleteMemoryState()        
        complete_state.__get_values_from_player(player)
        
        return complete_state
    
    def build_by_observation(observation):
        
        complete_state = CompleteMemoryState()        
        complete_state.__get_values_from_observation(observation)
        
        return complete_state
                        
    def get_low_values():
        
        low_values = list()
        
        low_values.append(PlayerCompleteState.get_low_values())
        low_values.append(OtherPlayerCompleteState.get_low_values())
        low_values.append(OtherPlayerCompleteState.get_low_values())
        
        return np.concatenate(low_values)       
        
     
    def get_high_values():
        
        high_values = list()
        
        high_values.append(PlayerCompleteState.get_high_values())
        high_values.append(OtherPlayerCompleteState.get_high_values())
        high_values.append(OtherPlayerCompleteState.get_high_values())
        
        return np.concatenate(high_values)  
        
    def get_number_of_observations():
    
        return PlayerCompleteState.get_number_of_observations() + (OtherPlayerCompleteState.get_number_of_observations() * 2)
       
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(PlayerCompleteState.get_expected_distribution())
        distribution.append(OtherPlayerCompleteState.get_expected_distribution())
        distribution.append(OtherPlayerCompleteState.get_expected_distribution())
                    
        return np.concatenate(distribution)
    
    def trained_with_chopsticks_phase():
        return True