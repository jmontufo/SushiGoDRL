# -*- coding: utf-8 -*-

      
import gym

import numpy as np
from gym_sushi_go.spaces.dynamic_space import DynamicSpace

class DynamicMultiSpace(gym.Space):

    def __init__(self, num_dynamic_spaces, max_space):
        
        self.n = max_space ** num_dynamic_spaces
        
        self.num_dynamic_spaces = num_dynamic_spaces
        self.max_space = max_space
        
        self.dynamic_spaces = []
        self.available_actions = []
    
        for i in range(self.num_dynamic_spaces):
            self.dynamic_spaces.append(DynamicSpace(max_space))
            self.available_actions.append(self.dynamic_spaces[i].available_actions)
            
    
    def update_actions(self, actions):
                
        self.available_actions = []
    
        for i in range(self.num_dynamic_spaces):
            self.dynamic_spaces[i].update_actions(actions[i])
            self.available_actions.append(self.dynamic_spaces[i].available_actions)
    
    def sample(self):
        
        sample = []
        for i in range(self.num_dynamic_spaces):
            sample.append(self.dynamic_spaces[i].sample())
            
        return sample
    
    def contains(self, x):
        
        contains_all = True
        for i in range(self.num_dynamic_spaces):
            contains_all = contains_all and self.dynamic_spaces[i].contains(x[i])
        
        return contains_all
    
    @property
    def shape(self):
        
        return [self.num_dynamic_spaces, self.max_space]
    
    def __repr__(self):
        
        return "DynamicMulti(%d)" % self.n
    