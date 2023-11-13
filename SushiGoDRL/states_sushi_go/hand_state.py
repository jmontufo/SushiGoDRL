# -*- coding: utf-8 -*-


import numpy as np

from states_sushi_go.state import State

from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import SalmonNigiriCard
from model_sushi_go.card import SquidNigiriCard
from model_sushi_go.card import WasabiCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import DumplingCard
from model_sushi_go.card import PuddingCard

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
    
class CompleteHandState(State):
    
    chopsticks_values = 4
    egg_nigiri_values = 4
    salmon_nigiri_values = 4
    squid_nigiri_values = 4
    wasabi_values = 4
    one_maki_values = 4
    two_maki_values = 4
    three_maki_values = 4
    sashimi_values = 4
    tempura_values = 4
    dumpling_values = 4
    pudding_values = 4
    
    def __init__(self):
        
        self.__chopsticks = 0
        self.__egg_nigiri = 0
        self.__salmon_nigiri = 0
        self.__squid_nigiri = 0
        self.__wasabi = 0
        self.__one_maki = 0
        self.__two_maki = 0
        self.__three_maki = 0
        self.__sashimi = 0
        self.__tempura = 0
        self.__dumpling = 0
        self.__pudding = 0
                                
    def __get_values_from_hand(self, hand):
        
        for card in hand.get_cards():
            
            if type(card) == ChopsticksCard and self.__chopsticks < 3:                
                self.__chopsticks = self.__chopsticks + 1
            elif type(card) == EggNigiriCard and self.__egg_nigiri < 3:                
                self.__egg_nigiri = self.__egg_nigiri + 1
            elif type(card) == SalmonNigiriCard and self.__salmon_nigiri < 3:                
                self.__salmon_nigiri = self.__salmon_nigiri + 1
            elif type(card) == SquidNigiriCard and self.__squid_nigiri < 3:                
                self.__squid_nigiri = self.__squid_nigiri + 1
            elif type(card) == WasabiCard and self.__wasabi < 3:                
                self.__wasabi = self.__wasabi + 1
            elif type(card) == OneMakiCard and self.__one_maki < 3:                
                self.__one_maki = self.__one_maki + 1
            elif type(card) == TwoMakiCard and self.__two_maki < 3:                
                self.__two_maki = self.__two_maki + 1
            elif type(card) == ThreeMakiCard and self.__three_maki < 3:                
                self.__three_maki = self.__three_maki + 1
            elif type(card) == SashimiCard and self.__sashimi < 3:                
                self.__sashimi = self.__sashimi + 1
            elif type(card) == TempuraCard and self.__tempura < 3:                
                self.__tempura = self.__tempura + 1
            elif type(card) == DumplingCard and self.__dumpling < 3:                
                self.__dumpling = self.__dumpling + 1
            elif type(card) == PuddingCard and self.__pudding < 3:                
                self.__pudding = self.__pudding + 1
    
    def __build_when_unknown_hand(self): 
        
        self.__chopsticks = 1
        self.__egg_nigiri = 1
        self.__salmon_nigiri = 1
        self.__squid_nigiri = 1
        self.__wasabi = 1
        self.__one_maki = 1
        self.__two_maki = 1
        self.__three_maki = 1
        self.__sashimi = 1
        self.__tempura = 1
        self.__dumpling = 1
        self.__pudding = 1 
        
    def __get_values_from_observation(self, observation):
        
        self.__chopsticks = observation[0]
        self.__egg_nigiri = observation[1]
        self.__salmon_nigiri = observation[2]
        self.__squid_nigiri = observation[3]
        self.__wasabi = observation[4]
        self.__one_maki = observation[5]
        self.__two_maki = observation[6]
        self.__three_maki = observation[7]
        self.__sashimi = observation[8]
        self.__tempura = observation[9]
        self.__dumpling = observation[10]
        self.__pudding = observation[11] 
            
    def get_as_observation(self):
        
        state = (self.__chopsticks, 
                 self.__egg_nigiri,
                 self.__salmon_nigiri,
                 self.__squid_nigiri,
                 self.__wasabi,
                 self.__one_maki,
                 self.__two_maki,
                 self.__three_maki,
                 self.__sashimi,
                 self.__tempura,
                 self.__dumpling,
                 self.__pudding)
        
        return np.array(state)
                       
    def build_by_player(player):
        
        hand = player.get_hand()
        
        hand_state = CompleteHandState()        
        hand_state.__get_values_from_hand(hand)
        
        return hand_state
    
    def build_when_unknown_hand():
        
        hand_state = CompleteHandState()        
        hand_state.__build_when_unknown_hand()
        
        return hand_state
                          
    def build_by_observation(observation):
        
        hand_state = CompleteHandState()
        hand_state.__get_values_from_observation(observation)             
        
        return hand_state
    
    def get_low_values():
        return np.array([0] * 12)
    
    def get_high_values():
                
        high_values = [CompleteHandState.chopsticks_values, 
                       CompleteHandState.egg_nigiri_values, 
                       CompleteHandState.salmon_nigiri_values, 
                       CompleteHandState.squid_nigiri_values, 
                       CompleteHandState.wasabi_values, 
                       CompleteHandState.one_maki_values, 
                       CompleteHandState.two_maki_values, 
                       CompleteHandState.three_maki_values, 
                       CompleteHandState.sashimi_values, 
                       CompleteHandState.tempura_values, 
                       CompleteHandState.dumpling_values, 
                       CompleteHandState.pudding_values]
        
        return np.array(high_values)
    
    def get_number_of_observations():
        
        return 12
    
    def get_expected_distribution():
        
        distribution = ['Poisson'] * 12
                
        return np.array(distribution)