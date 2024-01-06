# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:43:28 2023

@author: jmont
"""


from gym_sushi_go import setup


from states_sushi_go.player_state import PlayerSimplState
from states_sushi_go.game_state import GameState
from states_sushi_go.with_phase.player_state import PlayerSimplWithPhaseState
from states_sushi_go.with_phase.game_state import GameWithPhaseState
from states_sushi_go.with_phase.complete_state import CompleteMemoryState
from states_sushi_go.complete_state_fixed import CompleteState
from states_sushi_go.with_phase.player_state import PlayerCompleteState

from agent_builder.q_learning.q_learning import QL_Builder
from agent_builder.deep_q_learning.deep_q_learning_torch import DQLT_Builder
from agent_builder.deep_q_learning.double_deep_q_learning_torch import DDQLT_Builder
from agent_builder.deep_q_learning.dualing_deep_q_learning import DualingDQLT_Builder
from agent_builder.deep_q_learning.maxmin_deep_q_learning import Maxmin_DQL_Builder
from agent_builder.reinforce.reinforce import PG_Reinforce_Builder
from agent_builder.reinforce.reinforce_learned_baseline import PGRLB_Builder
from agent_builder.reinforce.actor_critic import AC_Builder
from agent_builder.multiagent_learning.minimax_multi_deep_q_learning import MinimaxMultiDQL_Builder

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
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase2


from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

try:
    
    versus_agents = [
                    # RandomAgent(),
                    # ChopstickLoverAgent(),
                    # ChopstickHaterAgent(),
                    # ChopstickLoverAtFirstAgent(),
                    # DumplingLoverAgent(),
                    # MakiLoverAgent(),
                    # MakiHaterAgent(),
                    # NigiriLoverAgent(),
                    # PuddingLoverAgent(),
                    # PuddingHaterAgent(),
                    # SashimiLoverAgent(),
                    # SashimiHaterAgent(),
                    # TempuraLoverAgent(),
                    # WasabiLoverAgent(),
                    # WasabiLoverAtFirstAgent(),
                    # DumplingSuperLoverAgent(),
                    # MakiSuperLoverAgent(),
                    # NigiriSuperLoverAgent(),
                    # PuddingSuperLoverAgent(),
                    # SashimiSuperLoverAgent(),
                    # TempuraSuperLoverAgent(),
                    RandomAgent(),
                    ChopstickLoverAgent(),
                    DumplingLoverAgent(),
                    MakiLoverAgent(),
                    NigiriLoverAgent(),
                    PuddingLoverAgent(),
                    SashimiLoverAgent(),
                    TempuraLoverAgent(),
                    WasabiLoverAgent(),
                    DeepQLearningTorchAgentPhase1(),
                    DoubleDeepQLearningTorchAgentPhase1(),
                    DualingDeepQLearningTorchAgentPhase1(),
                    MaxminDQLearningTorchAgentPhase1(),
                    DeepQLearningTorchAgentPhase2(),
                    DoubleDeepQLearningTorchAgentPhase2(),
                    DualingDeepQLearningTorchAgentPhase2(),
                    MaxminDQLearningTorchAgentPhase2()
                    ]
    
    num_players = 2
    # state_type = PlayerSimplWithPhaseState
    total_episodes = 40000 
    max_epsilon = 1             
    min_epsilon = 0.05            
    decay_rate = 0.0005       
    # learning_rate = 0.3           
    # discount = 1           
    reference = "Phase3"      
    # reward_by_win = 0
    N = 2 
    
    # state_type_list = [CompleteState.get_type_from_num_players(num_players)]
    state_type_list = [CompleteMemoryState]
    
    # learning_rate_list = [0.005, 0.001, 0.0005]
    learning_rate_list = [0.00075]
             
    discount_list = [1]  
    # discount_list = [0.2]         
    
    reward_by_win_list = [-1]
    # reward_by_win_list = [0]
    
    for reward_by_win in reward_by_win_list:
        for state_type in state_type_list:
            for learning_rate in learning_rate_list:
                for discount in discount_list:
                
                    agent_reference = reference + "-" + str(reward_by_win)
            
                    # builder = QL_Builder(num_players, versus_agents, state_type, total_episodes, 
                    #                  max_epsilon, min_epsilon, decay_rate, learning_rate, discount, 
                    #                  agent_reference, reward_by_win = reward_by_win)
                    # builder = PGRLB_Builder(num_players, versus_agents, state_type, total_episodes, 
                    #                       learning_rate, discount, reference)
                    # builder = AC_Builder(num_players, versus_agents, state_type, total_episodes, 
                    #                                           learning_rate, discount, reference)
                    #builder = DQLT_Builder(num_players, versus_agents, state_type, total_episodes, 
                    #                        max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, None, reward_by_win)
                    builder = DualingDQLT_Builder(num_players, versus_agents, state_type, total_episodes, 
                                                  max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, None, reward_by_win)
                    #builder = DDQLT_Builder(num_players, versus_agents, state_type, total_episodes, 
                    #                         max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, None, reward_by_win)
                    # builder = PG_Reinforce_Builder(num_players, versus_agents, state_type, total_episodes, 
                    #                        learning_rate, discount, reference)
                    # builder = Maxmin_DQL_Builder(num_players, versus_agents, state_type, 
                    #                                total_episodes, max_epsilon, min_epsilon, 
                    #                                decay_rate, learning_rate, discount, N, 
                    #                                reference, None, reward_by_win)
                    # builder = MinimaxMultiDQL_Builder(state_type, total_episodes, max_epsilon, 
                    #                                   min_epsilon, decay_rate, learning_rate, discount,
                    #                                   reference, None)

                    builder.run()
finally:
     tracker.stop()