# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:49:53 2020

@author: Jose Montufo
"""

import gym
from gym import spaces

import numpy as np

from gym_sushi_go.spaces.dynamic_multi_space import DynamicMultiSpace

from model_sushi_go.game import MultiplayerGame
from states_sushi_go.player_state import PlayerState

class SushiGoMultiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players, state_type = PlayerState, setup = "Original",
                 reward_by_win = 0, chopsticks_phase_mode = False):
        
        super(SushiGoMultiEnv, self).__init__()
        
        self.state_type = state_type
        
        lows = self.state_type.get_low_values() * num_players
        highs = self.state_type.get_high_values() * num_players
        
        if chopsticks_phase_mode:
            self.num_actions = 9
        else:
            self.num_actions = 37
            
                      
        self.__action_space = DynamicMultiSpace(num_players, self.num_actions)
        
        self.observation_space = spaces.Box(lows, highs, dtype=np.int32)
        
        self.setup = setup
        self.num_players = num_players
        self.reward_by_win = reward_by_win
        self.chopsticks_phase_mode = chopsticks_phase_mode
        
        self.game = MultiplayerGame(self.setup, num_players, reward_by_win, 
                                    chopsticks_phase_mode)
    
    def step(self, action):
               
        rewards = self.game.play_action_number(action)
        
        observations = self.__build_observations() 
            
        done = self.game.is_finished()
        
        info = {}
        if done:
            winners = self.game.declare_winner()
            scores =  self.game.report_scores()
            
            points_by_victory = []
            
            for player_number in range(0, self.num_players):
                if player_number in winners:
                    points_by_victory.append(1 / len(winners))
                else :
                    points_by_victory.append(0)
                    
                            
            info['points_by_victory'] = points_by_victory  
            info['score'] = scores             
            
        info['rival_legal_actions'] = [] 
        for player_number in range(0, self.num_players):
            info['rival_legal_actions'].append(self.game.get_rival_legal_actions_numbers(player_number))   
                    
        
        return observations, rewards, done, info;
      
    @property
    def action_space(self):
        
        if self.game is not None:            
            
            legal_actions_array = self.game.get_legal_actions_numbers(force_no_chopsticks_phase=[False]*self.num_players)
            
            self.__action_space.update_actions(legal_actions_array) 
                            
        return self.__action_space
    
    
    def reset(self):
      
        self.game = MultiplayerGame(self.setup, self.num_players, self.reward_by_win,
                               self.chopsticks_phase_mode)     
              
        return self.__build_observations()  
    
    def __build_observations(self):
        
        observations = []
        
        for player_number in range(0, self.num_players):
            
            player = self.game.get_player(player_number)        
            observations.append(self.state_type.build_by_player(player).get_as_observation()) 
            
        return observations
    
    def render(self, mode='human'):
        print(str(self.game))
    
    def close(self):
        return;
  


