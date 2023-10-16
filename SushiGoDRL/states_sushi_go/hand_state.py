# -*- coding: utf-8 -*-


import numpy as np

from states_sushi_go.state import State
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import MakiCard

class HandState(State):
    
    max_maki_in_hand_values = 4
    max_nigiri_in_hand_values = 4
    
    def __init__(self):
        
        self.__max_maki_in_hand = 0
        self.__max_nigiri_in_hand = 0
        
    def get_max_maki_in_hand(self):
        
        return self.__max_maki_in_hand
        
    def __set_max_maki_in_hand(self, max_maki_in_hand):
        
        self.__max_maki_in_hand = max_maki_in_hand
        
    def get_max_nigiri_in_hand(self):
        
        return self.__max_nigiri_in_hand
        
    def __set_max_nigiri_in_hand(self, max_nigiri_in_hand):
        
        self.__max_nigiri_in_hand = max_nigiri_in_hand
                        
    def __get_values_from_hand(self, hand):
        
        for card in hand.get_cards():
            
            if card.get_number() == NigiriCard.get_number():                
                nigiri_value = card.get_value()                
                if nigiri_value > self.get_max_nigiri_in_hand():
                    self.__set_max_nigiri_in_hand(nigiri_value)
            elif card.get_number() == MakiCard.get_number():
                maki_value = card.get_value()                
                if maki_value > self.get_max_maki_in_hand():
                    self.__set_max_maki_in_hand(maki_value)
                    
    def __get_values_from_number(self, number):
        
        self.__set_max_maki_in_hand(number % HandState.max_maki_in_hand_values)
        number = int(number / HandState.max_maki_in_hand_values) 
        
        self.__set_max_nigiri_in_hand(number)      
        
    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_max_maki_in_hand() * base
        base *= self.max_maki_in_hand_values
        
        accumulated_value += self.get_max_nigiri_in_hand() * base
        base *= self.max_nigiri_in_hand_values     
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_max_maki_in_hand(observation[0])
        self.__set_max_nigiri_in_hand(observation[1])   
            
    def get_as_observation(self):
        
        state = (self.get_max_maki_in_hand(), 
                 self.get_max_nigiri_in_hand())
        
        return np.array(state)
                       
    def build_by_player(player):
        
        hand = player.get_hand()
        
        hand_state = HandState()        
        hand_state.__get_values_from_hand(hand)
        
        return hand_state
                       
    def build_by_number(number):
        
        hand_state = HandState()        
        hand_state.__get_values_from_number(number)
        
        return hand_state
    
    def build_by_observation(observation):
        
        hand_state = HandState()
        hand_state.__get_values_from_observation(observation)             
        
        return hand_state
    
    def get_total_numbers():
        
        base = 1        
        base *= HandState.max_maki_in_hand_values
        base *= HandState.max_nigiri_in_hand_values
        
        return base
    
    def get_low_values():
        return np.array([0] * 2)
    
    def get_high_values():
        
        high_values = [HandState.max_maki_in_hand_values, 
                       HandState.max_nigiri_in_hand_values]
        
        return np.array(high_values)
    
    def get_number_of_observations():
        
        return 2
    
    def get_expected_distribution():
        
        distribution = ['Uniform', 'Uniform']
                
        return np.array(distribution)
    
class HandStateWithoutZeroValue(State):
    
    max_maki_in_hand_values = 3
    max_nigiri_in_hand_values = 3
    
    def __init__(self):
        
        self.__max_maki_in_hand = 0
        self.__max_nigiri_in_hand = 0
        
    def get_max_maki_in_hand(self):
        
        return self.__max_maki_in_hand
        
    def __set_max_maki_in_hand(self, max_maki_in_hand):
        
        self.__max_maki_in_hand = max_maki_in_hand
        
    def get_max_nigiri_in_hand(self):
        
        return self.__max_nigiri_in_hand
        
    def __set_max_nigiri_in_hand(self, max_nigiri_in_hand):
        
        self.__max_nigiri_in_hand = max_nigiri_in_hand
                        
    def __get_values_from_hand(self, hand):
        
        for card in hand.get_cards():
            
            if card.get_number() == NigiriCard.get_number():    
                
                nigiri_value = card.get_value()            
                
                if nigiri_value > self.get_max_nigiri_in_hand() + 1:                    
                    self.__set_max_nigiri_in_hand(nigiri_value - 1)
                    
            elif card.get_number() == MakiCard.get_number():
                maki_value = card.get_value()                
                if maki_value > self.get_max_maki_in_hand() + 1:
                    self.__set_max_maki_in_hand(maki_value - 1)
                    
    def __get_values_from_number(self, number):
        
        self.__set_max_maki_in_hand(number % HandStateWithoutZeroValue.max_maki_in_hand_values)
        number = int(number / HandStateWithoutZeroValue.max_maki_in_hand_values) 
        
        self.__set_max_nigiri_in_hand(number)      
        
    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_max_maki_in_hand() * base
        base *= self.max_maki_in_hand_values
        
        accumulated_value += self.get_max_nigiri_in_hand() * base
        base *= self.max_nigiri_in_hand_values     
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_max_maki_in_hand(observation[0])
        self.__set_max_nigiri_in_hand(observation[1])   
            
    def get_as_observation(self):
        
        state = (self.get_max_maki_in_hand(), 
                 self.get_max_nigiri_in_hand())
        
        return np.array(state)
                       
    def build_by_player(player):
        
        hand = player.get_hand()
        
        hand_state = HandStateWithoutZeroValue()        
        hand_state.__get_values_from_hand(hand)
        
        return hand_state
                       
    def build_by_number(number):
        
        hand_state = HandStateWithoutZeroValue()        
        hand_state.__get_values_from_number(number)
        
        return hand_state
    
    def build_by_observation(observation):
        
        hand_state = HandStateWithoutZeroValue()
        hand_state.__get_values_from_observation(observation)             
        
        return hand_state
    
    def get_total_numbers():
        
        base = 1        
        base *= HandStateWithoutZeroValue.max_maki_in_hand_values
        base *= HandStateWithoutZeroValue.max_nigiri_in_hand_values
        
        return base
    
    def get_low_values():
        return np.array([0] * 2)
    
    def get_high_values():
        
        high_values = [HandStateWithoutZeroValue.max_maki_in_hand_values, 
                       HandStateWithoutZeroValue.max_nigiri_in_hand_values]
        
        return np.array(high_values)
    
    def get_number_of_observations():
        
        return 2