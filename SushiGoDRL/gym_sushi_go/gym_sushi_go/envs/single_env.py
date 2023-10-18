# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:49:53 2020

@author: Jose Montufo
"""

import numpy as np
import gym
from gym import spaces

from gym_sushi_go.spaces.dynamic_space import DynamicSpace

from model_sushi_go.game import SingleGame
from states_sushi_go.player_state import PlayerState

class SushiGoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, agents, state_type = PlayerState, setup = "Original",
                 reward_by_win = 0, chopsticks_phase_mode = False):
        
        super(SushiGoEnv, self).__init__()
        
        self.state_type = state_type
        
        lows = self.state_type.get_low_values()
        highs = self.state_type.get_high_values()
        
        if chopsticks_phase_mode:
            self.__action_space = DynamicSpace(37)
        else:
            self.__action_space = DynamicSpace(37)
            
        self.observation_space = spaces.Box(lows, highs, dtype=np.int32)
        
        self.setup = setup
        self.agents = agents
        self.reward_by_win = reward_by_win
        self.chopsticks_phase_mode = chopsticks_phase_mode
        
        self.game = SingleGame(self.setup, self.agents, reward_by_win, 
                               chopsticks_phase_mode)    
        
    
    def step(self, action):
          
        player = self.game.get_player(0)
        
        reward = self.game.play_action_number(action)
        
        observation = self.state_type.build_by_player(player).get_as_observation()        
        done = self.game.is_finished()
        
        return observation, reward, done, {};
      
    @property
    def action_space(self):
        
        if self.game is not None:            
            
            self.__action_space.update_actions(self.game.get_legal_actions_numbers()) 
            
        return self.__action_space
    
    def reset(self):
      
        self.game = SingleGame(self.setup, self.agents, self.reward_by_win,
                               self.chopsticks_phase_mode)     
        
        player = self.game.get_player(0)
        observation = self.state_type.build_by_player(player).get_as_observation()        
      
        return observation   
    
    def render(self, mode='human'):
        print(str(self.game))
    
    def close(self):
        return;
  


