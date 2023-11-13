# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:43:28 2023

@author: jmont
"""
from states_sushi_go.player_state import PlayerSimplState
from states_sushi_go.game_state import GameState
from states_sushi_go.with_phase.player_state import PlayerSimplWithPhaseState
from states_sushi_go.with_phase.game_state import GameWithPhaseState
from states_sushi_go.with_phase.complete_state import CompleteMemoryState

from agent_builder.q_learning.q_learning import QL_Builder
from agent_builder.reinforce.reinforce_baseline import PG_Builder
from agent_builder.reinforce.reinforce_learned_baseline import PGRLB_Builder

from agents_sushi_go.random_agent import RandomAgent
from agents_sushi_go.card_lover_agent import SashimiLoverAgent
from agents_sushi_go.card_lover_agent import SashimiSuperLoverAgent
from agents_sushi_go.card_lover_agent import SashimiHaterAgent
from agents_sushi_go.card_lover_agent import TempuraLoverAgent
from agents_sushi_go.card_lover_agent import TempuraSuperLoverAgent
from agents_sushi_go.card_lover_agent import DumplingLoverAgent
from agents_sushi_go.card_lover_agent import DumplingSuperLoverAgent
from agents_sushi_go.card_lover_agent import MakiLoverAgent
from agents_sushi_go.card_lover_agent import MakiSuperLoverAgent
from agents_sushi_go.card_lover_agent import MakiHaterAgent
from agents_sushi_go.card_lover_agent import WasabiLoverAgent
from agents_sushi_go.card_lover_agent import WasabiLoverAtFirstAgent
from agents_sushi_go.card_lover_agent import NigiriLoverAgent
from agents_sushi_go.card_lover_agent import NigiriSuperLoverAgent
from agents_sushi_go.card_lover_agent import PuddingLoverAgent
from agents_sushi_go.card_lover_agent import PuddingSuperLoverAgent
from agents_sushi_go.card_lover_agent import PuddingHaterAgent
from agents_sushi_go.card_lover_agent import ChopstickLoverAgent
from agents_sushi_go.card_lover_agent import ChopstickHaterAgent
from agents_sushi_go.card_lover_agent import ChopstickLoverAtFirstAgent


versus_agents = [
                RandomAgent(),
                ChopstickLoverAgent(),
                ChopstickHaterAgent(),
                ChopstickLoverAtFirstAgent(),
                DumplingLoverAgent(),
                MakiLoverAgent(),
                MakiHaterAgent(),
                NigiriLoverAgent(),
                PuddingLoverAgent(),
                PuddingHaterAgent(),
                SashimiLoverAgent(),
                SashimiHaterAgent(),
                TempuraLoverAgent(),
                WasabiLoverAgent(),
                WasabiLoverAtFirstAgent(),
                # DumplingSuperLoverAgent(),
                # MakiSuperLoverAgent(),
                # NigiriSuperLoverAgent(),
                # PuddingSuperLoverAgent(),
                # SashimiSuperLoverAgent(),
                # TempuraSuperLoverAgent(),
                # QLearningAgentPhase1(),
                # MCTreeSearchAgentPhase1(),
                # DeepQLearningAgentPhase1(),
                # DoubleDeepQLearningAgentPhase1(),
                # QLearningAgentPhase2(),
                # MCTreeSearchAgentPhase2(),
                # DeepQLearningAgentPhase2(),
                # DoubleDeepQLearningAgentPhase2(),
                # QLearningAgentPhase3(),
                # MCTreeSearchAgentPhase3(),
                # DeepQLearningAgentPhase3(),
                # DoubleDeepQLearningAgentPhase3()
                ]

num_players = 2
# state_type = PlayerSimplWithPhaseState
total_episodes = 2000 
max_epsilon = 1             
min_epsilon = 0.05            
decay_rate = 0.0005       
# learning_rate = 0.3           
# discount = 1           
reference = "ReinforceWithBaseline"      
# reward_by_win = 0


state_type_list = [CompleteMemoryState]

learning_rate_list = [0.01]
         
# discount_list = [0.9, 0.95, 0.99] 
discount_list = [1]           

reward_by_win_list = [0]

for state_type in state_type_list:
    for learning_rate in learning_rate_list:
        for reward_by_win in reward_by_win_list:
            for discount in discount_list:
            
                agent_reference = reference + "-" + str(reward_by_win)
        
                # builder = QL_Builder(num_players, versus_agents, state_type, total_episodes, 
                #                  max_epsilon, min_epsilon, decay_rate, learning_rate, discount, 
                #                  agent_reference, reward_by_win = reward_by_win)
                # builder = PGRLB_Builder(num_players, versus_agents, state_type, total_episodes, 
                #                      learning_rate, discount, reference)
                builder = PG_Builder(num_players, versus_agents, state_type, total_episodes, 
                                                          learning_rate, discount, reference)

                builder.run()

