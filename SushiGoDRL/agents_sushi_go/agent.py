# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, player = None, filename = None):
        
        self.__player = player
        self.filename = filename
        
        if player is not None:
            game = player.get_game()
            if game is not None:
                self.__chopsticks_phase_mode = game.is_chopsticks_phase_mode()
    
    def get_player(self):
        
        return self.__player
    
    def is_chopsticks_phase_mode(self):
        
        return  self.__chopsticks_phase_mode
    
    def set_player(self, player):
        
        self.__player = player
        if player is not None:
            game = player.get_game()
            if game is not None:
                self.__chopsticks_phase_mode = game.is_chopsticks_phase_mode()
                       
    @abstractmethod
    def choose_action(self, legal_actions):
        pass
    
    @abstractmethod
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    @abstractmethod
    def save_training(self):
        pass
    
    @abstractmethod
    def trained_with_chopsticks_phase(self):
        pass