# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:49:08 2020

@author: Jose Montufo
"""


import numpy as np


from states_sushi_go.state import State
from states_sushi_go.player_state import PlayerFullState
from states_sushi_go.table_state import TableFullState

class CompleteState(State):    
    
    def __init__(self, num_players):         

        self.__player_state = None
        self.__num_players = num_players - 1 # do not count main player
        self.__other_players_tables = []
        
    def get_player_state(self):
        
        return self.__player_state
        
    def __set_player_state(self, player_state):
        
        self.__player_state = player_state   
        
    def get_other_players_tables(self):
        
        return self.__other_players_tables
        
    def __add_other_player_table_state(self, other_player_table_state):
        
        self.__other_players_tables.append(other_player_table_state)   
           
        
    def __get_values_from_player(self, player):
        
        self.__set_player_state(PlayerFullState.build_by_player(player))        
        
        game = player.get_game()
        
        player_position = player.get_position()
        
        for i in range(self.__num_players):
            
            rival_position = (i + player_position + 1) % (self.__num_players + 1)
            
            if rival_position != player_position:
                other_player_state = TableFullState.build_by_player(game.get_player(rival_position))
                self.__add_other_player_table_state(other_player_state)
               
    def get_values_from_number(self, number):
             
        player_state_total_values = PlayerFullState.get_total_numbers()
        player_state_number = number % player_state_total_values   
    
        self.__set_player_state(PlayerFullState.build_by_number(player_state_number))
        
        number = int(number / player_state_total_values) 
        
        other_players_state_total_values = TableFullState.get_total_numbers()
                
        self.__other_players_tables = []
        
        for player_position in range(self.__num_players):
            
            other_player_state_number = number % other_players_state_total_values 
            other_player_state = TableFullState.build_by_number(other_player_state_number)
    
            self.__add_other_player_table_state(other_player_state)
        
            number = int(number / other_players_state_total_values) 

    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_player_state().get_state_number() * base
        base *= PlayerFullState.get_total_numbers()
        
        for other_player_table_state in self.get_other_players_tables():
            accumulated_value += other_player_table_state.get_state_number() * base
            base *= TableFullState.get_total_numbers()    
                
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        pos_ini = 0
        pos = PlayerFullState.get_number_of_observations()
        
        self.__set_player_state(PlayerFullState.build_by_observation(observation[pos_ini:pos]))
        
        self.__other_players_tables = []
        
        for player_position in range(self.__num_players): 
            
            pos_ini = pos
            pos = pos_ini + TableFullState.get_number_of_observations()
                        
            other_player_state = TableFullState.build_by_observation(observation[pos_ini:pos])            
            self.__add_other_player_table_state(other_player_state)
        
    def get_as_observation(self):
        
        observations = list()
        
        observations.append(self.get_player_state().get_as_observation())
                    
        for other_player_table_state in self.get_other_players_tables():
            other_player_observation = other_player_table_state.get_as_observation()  
            observations.append(other_player_observation)
        
        return np.concatenate(observations)       
    
    def build_by_player(player):
                
        complete_state = CompleteState(player.get_game().get_num_players())        
        complete_state.__get_values_from_player(player)
        
        return complete_state
    
    def build_by_number(number, num_players):
        
        complete_state = CompleteState(num_players)        
        complete_state.get_values_from_number(number)
        
        return complete_state
    
    def build_by_observation(observation, num_players):
        
        complete_state = CompleteState(num_players)        
        complete_state.__get_values_from_observation(observation)
        
        return complete_state
        
    def get_total_numbers(num_players):
        
        base = 1        
        base *= PlayerFullState.get_total_numbers()
        base *= TableFullState.get_total_numbers() * (num_players - 1)
        
        return base
                
    def get_low_values(num_players):
        
        low_values = list()
        
        low_values.append(PlayerFullState.get_low_values())
                    
        for i in range(num_players - 1): 
            other_player_low_values = TableFullState.get_low_values()  
            low_values.append(other_player_low_values)
        
        return np.concatenate(low_values)       
        
     
    def get_high_values(num_players):
        
        hign_values = list()
        
        hign_values.append(PlayerFullState.get_high_values())
                    
        for i in range(num_players - 1): 
            other_player_high_values = TableFullState.get_high_values()  
            hign_values.append(other_player_high_values)
        
        return np.concatenate(hign_values)  
        
    def get_number_of_observations(num_players):
    
        return PlayerFullState.get_number_of_observations() + (TableFullState.get_number_of_observations() * (num_players - 1))
    
    def get_type_from_num_players(num_players):
        
        aux = [TwoPlayersCompleteState, 
               ThreePlayersCompleteState,
               FourPlayersCompleteState,
               FivePlayersCompleteState]
        
        return aux[num_players - 2]
    
    def get_expected_distribution(num_players):
        
        distribution = list()
        
        distribution.append(PlayerFullState.get_expected_distribution())
                    
        for i in range(num_players - 1): 
            other_player_distribution = TableFullState.get_expected_distribution()  
            distribution.append(other_player_distribution)
        
        return np.concatenate(distribution)
    
class TwoPlayersCompleteState(CompleteState):    
    
    num_players = 2
    
    def __init__(self):
        super(TwoPlayersCompleteState,self).__init__(TwoPlayersCompleteState.num_players)    
        
    def build_by_number(number):
        
        return CompleteState.build_by_number(number, TwoPlayersCompleteState.num_players)
    
    def build_by_observation(observation):
    
        return CompleteState.build_by_observation(observation, TwoPlayersCompleteState.num_players)
    
    def get_total_numbers():
        
        return CompleteState.get_total_numbers(TwoPlayersCompleteState.num_players)
    
    def get_low_values():
        
        return CompleteState.get_low_values(TwoPlayersCompleteState.num_players)
    
    def get_high_values():
        
        return CompleteState.get_high_values(TwoPlayersCompleteState.num_players)
    
    def get_number_of_observations():
            
        return CompleteState.get_number_of_observations(TwoPlayersCompleteState.num_players)    
    
    def get_expected_distribution():
            
        return CompleteState.get_expected_distribution(TwoPlayersCompleteState.num_players)
    
class ThreePlayersCompleteState(CompleteState):    
   
    num_players = 3
    
    def __init__(self):
        super().__init__(ThreePlayersCompleteState.num_players)    
        
    def build_by_number(number):
        
        return CompleteState.build_by_number(number, ThreePlayersCompleteState.num_players)
    
    def build_by_observation(observation):
    
        return CompleteState.build_by_observation(observation, ThreePlayersCompleteState.num_players)
    
    def get_total_numbers():
        
        return CompleteState.get_total_numbers(ThreePlayersCompleteState.num_players)
    
    def get_low_values():
        
        return CompleteState.get_low_values(ThreePlayersCompleteState.num_players)
    
    def get_high_values():
        
        return CompleteState.get_high_values(ThreePlayersCompleteState.num_players)
    
    def get_number_of_observations():
            
        return CompleteState.get_number_of_observations(ThreePlayersCompleteState.num_players)
    
    def get_expected_distribution():
            
        return CompleteState.get_expected_distribution(ThreePlayersCompleteState.num_players)
    
class FourPlayersCompleteState(CompleteState):    
    
    num_players = 4
    
    def __init__(self):
        super().__init__(FourPlayersCompleteState.num_players)    
    
    def build_by_number(number):
        
        return CompleteState.build_by_number(number, FourPlayersCompleteState.num_players)
    
    def build_by_observation(observation):
    
        return CompleteState.build_by_observation(observation, FourPlayersCompleteState.num_players)
        
    def get_total_numbers():
        
        return CompleteState.get_total_numbers(FourPlayersCompleteState.num_players)
    
    def get_low_values():
        
        return CompleteState.get_low_values(FourPlayersCompleteState.num_players)
    
    def get_high_values():
        
        return CompleteState.get_high_values(FourPlayersCompleteState.num_players)
    
    def get_number_of_observations():
            
        return CompleteState.get_number_of_observations(FourPlayersCompleteState.num_players)       
    
    def get_expected_distribution():
            
        return CompleteState.get_expected_distribution(FourPlayersCompleteState.num_players)
    
class FivePlayersCompleteState(CompleteState):    
    
    num_players = 5
    
    def __init__(self):
        super().__init__(FivePlayersCompleteState.num_players)    
    
    def build_by_number(number):
        
        return CompleteState.build_by_number(number, FivePlayersCompleteState.num_players)
    
    def build_by_observation(observation):
    
        return CompleteState.build_by_observation(observation, FivePlayersCompleteState.num_players)
     
    def get_total_numbers():
        
        return CompleteState.get_total_numbers(FivePlayersCompleteState.num_players)
    
    def get_low_values():
        
        return CompleteState.get_low_values(FivePlayersCompleteState.num_players)
    
    def get_high_values():
        
        return CompleteState.get_high_values(FivePlayersCompleteState.num_players)
    
    def get_number_of_observations():

        return CompleteState.get_number_of_observations(FivePlayersCompleteState.num_players) 
    
    def get_expected_distribution():
            
        return CompleteState.get_expected_distribution(FivePlayersCompleteState.num_players)
    