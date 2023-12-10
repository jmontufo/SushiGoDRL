# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:37:55 2020

@author: Jose Montufo
"""

import gym
from gym_sushi_go.envs.single_env import SushiGoEnv
from gym_sushi_go.envs.multiplayer_env import SushiGoMultiEnv

# Register the environment
gym.register(
    id='sushi-go-v0',
    entry_point='gym_sushi_go.envs.single_env:SushiGoEnv'
)

gym.register(
    id='sushi-go-multi-v0',
    entry_point='gym_sushi_go.envs.multiplayer_env:SushiGoMultiEnv'
)