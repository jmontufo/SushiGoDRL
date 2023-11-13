# -*- coding: utf-8 -*-

      
import gym

import numpy as np

class DynamicSpace(gym.Space):

    def __init__(self, max_space):
        
        self.n = max_space    
        self.available_actions = []
    
    def update_actions(self, actions):
        
        self.available_actions = actions
        return self.available_actions
    
    def sample(self):
        
        return np.random.choice(self.available_actions)
    
    def contains(self, x):
        
        return x in self.available_actions
    
    @property
    def shape(self):
        
        return len(self.available_actions)
    
    def __repr__(self):
        
        return "Dynamic(%d)" % self.n
    