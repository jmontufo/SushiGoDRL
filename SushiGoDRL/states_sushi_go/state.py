# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class State(ABC):
   
    @abstractmethod
    def build_by_player(player):
        pass
    
    def build_by_number(number):
        pass

    def get_state_number(self):
        pass    
    
    def get_total_numbers():
        pass
    
    @abstractmethod
    def build_by_observation(observation):
        pass    

    @abstractmethod
    def get_as_observation(self):
        pass
    
    @abstractmethod
    def get_low_values():
        pass
    
    @abstractmethod
    def get_high_values():
        pass
    
    @abstractmethod
    def get_number_of_observations():
        pass
    
    @abstractmethod
    def get_expected_distribution():
        pass
            
    def trained_with_chopsticks_phase():
        return False
    
