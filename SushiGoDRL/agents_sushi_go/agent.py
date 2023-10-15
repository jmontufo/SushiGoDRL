# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, player = None, filename = None):
        
        self.__player = player
        self.filename = filename
    
    def get_player(self):
        
        return self.__player
    
    def set_player(self, player):
        
        self.__player = player
       
    @abstractmethod
    def choose_action(self, legal_actions):
        pass
    
    @abstractmethod
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    @abstractmethod
    def save_training(self):
        pass