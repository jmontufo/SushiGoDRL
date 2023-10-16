# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np

from states_sushi_go.state import State

class GameState(State):
    
    current_turn_values = 10
    current_round_values = 3
    
    def __init__(self):
        
        self.__current_turn = 0
        self.__current_round = 0
        
    def get_current_turn(self):
        
        return self.__current_turn
        
    def __set_current_turn(self, current_turn):
        
        self.__current_turn = current_turn
        
    def get_current_round(self):
        
        return self.__current_round
        
    def __set_current_round(self, current_round):
        
        self.__current_round = current_round
                        
    def __get_values_from_game(self, game):
        
        self.__set_current_turn(game.get_turn())
        self.__set_current_round(game.get_round())
                    
    def __get_values_from_number(self, number):
        
        current_turn = (number % GameState.current_turn_values) + 1
        self.__set_current_turn(current_turn)        
        number = int(number / GameState.current_turn_values) 
        
        current_round = number + 1
        self.__set_current_round(current_round)      
        
    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += (self.get_current_turn() - 1) * base
        base *= self.current_turn_values
        
        accumulated_value += (self.get_current_round() - 1) * base
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_current_turn(observation[0])
        self.__set_current_round(observation[1])   
            
    def get_as_observation(self):
        
        state = (self.get_current_turn(), 
                 self.get_current_round())
        
        return np.array(state)
                       
    def build_by_player(player):
        
        game = player.get_game()
        
        game_state = GameState()        
        game_state.__get_values_from_game(game)
        
        return game_state
                       
    def build_by_number(number):
        
        game_state = GameState()        
        game_state.__get_values_from_number(number)
        
        return game_state
    
    def build_by_observation(observation):
        
        game_state = GameState()
        game_state.__get_values_from_observation(observation)             
        
        return game_state
    
    def get_total_numbers():
        
        base = 1        
        base *= GameState.current_turn_values
        base *= GameState.current_round_values
        
        return base
    
    def get_low_values():
        return np.array([0] * 2)
    
    def get_high_values():
        
        high_values = [GameState.current_turn_values, 
                       GameState.current_round_values]
        
        return np.array(high_values)
    
    
    def get_number_of_observations():
        
        return 2
    
    def get_expected_distribution():
        
        distribution = ['Uniform', 'Uniform']
        
        return np.array(distribution)