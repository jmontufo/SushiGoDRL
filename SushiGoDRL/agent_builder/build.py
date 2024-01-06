# -*- coding: utf-8 -*-

import pickle


from gym_sushi_go import setup
from agent_builder.utils import *


# from agent_builder.deep_q_learning.deep_q_learning import DQL_Builder
from agent_builder.deep_q_learning.deep_q_learning_torch import DQLT_Builder
from agent_builder.deep_q_learning.dualing_deep_q_learning import DualingDQLT_Builder
from agent_builder.deep_q_learning.maxmin_deep_q_learning import Maxmin_DQL_Builder
from agent_builder.reinforce.reinforce import PG_Reinforce_Builder
from agent_builder.reinforce.reinforce_baseline import PG_Builder
from agent_builder.reinforce.reinforce_learned_baseline import PGRLB_Builder
from agent_builder.reinforce.actor_critic import AC_Builder
from agent_builder.multiagent_learning.minimax_deep_q_learning import MinimaxDQL_Builder
from agent_builder.multiagent_learning.minimax_multi_deep_q_learning import MinimaxMultiDQL_Builder
# from agent_builder.double_deep_q_learning.double_deep_q_learning import DDQL_Builder

from states_sushi_go.complete_state_fixed import CompleteState
from states_sushi_go.with_phase.complete_state import CompleteMemoryState
from states_sushi_go.with_phase.player_state import PlayerCompleteState
from states_sushi_go.with_phase.game_state import GameWithPhaseState
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
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase1
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase2
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase3
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase3
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase3

num_players = 2

versus_agents = [
                RandomAgent()
                # ,
                # ChopstickLoverAgent(),
                # ChopstickHaterAgent(),
                # ChopstickLoverAtFirstAgent(),
                # DumplingLoverAgent(),
                # DumplingSuperLoverAgent(),
                # MakiLoverAgent(),
                # MakiSuperLoverAgent(),
                # MakiHaterAgent(),
                # NigiriLoverAgent(),
                # NigiriSuperLoverAgent(),
                # PuddingLoverAgent(),
                # PuddingSuperLoverAgent(),
                # PuddingHaterAgent(),
                # SashimiLoverAgent(),
                # SashimiSuperLoverAgent(),
                # SashimiHaterAgent(),
                # TempuraLoverAgent(),
                # TempuraSuperLoverAgent(),
                # WasabiLoverAgent(),
                # WasabiLoverAtFirstAgent()
                # ,
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

# state_type = CompleteState.get_type_from_num_players(num_players)

state_type = CompleteMemoryState
total_episodes = 5000
learning_rate = 0.0001    
discount = 0.95
max_epsilon = 1             
min_epsilon = 0.05            
decay_rate = 0.0005   
reward_by_win = -1
N = 4
        
reference = "Phase1"     

# previous_filename = "Deep_Q_Learning_torch_2p_CompleteMemoryState_lr0.001_10000_Phase1-2" 
previous_filename = None

# builder = DualingDQLT_Builder(num_players, versus_agents, state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, previous_filename, reward_by_win)
# builder = MinimaxDQL_Builder(num_players, versus_agents, state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, previous_filename, reward_by_win)
# builder = MinimaxMultiDQL_Builder(state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, previous_filename)
# builder = Maxmin_DQL_Builder(num_players, versus_agents, state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, discount, N, reference, previous_filename, reward_by_win)
       
# builder = PG_Builder(num_players, versus_agents, state_type, total_episodes, learning_rate, discount, reference, previous_filename)
#builder = PGRLB_Builder(num_players, versus_agents, state_type, total_episodes, learning_rate, discount, reference, previous_filename)
# builder = DQLT_Builder(num_players, versus_agents, state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, discount, reference, previous_filename, reward_by_win)

builder = AC_Builder(num_players, versus_agents, state_type, total_episodes, 
                                           learning_rate, discount, reference, 
                                           None, reward_by_win)
builder.run()


# Q_input = open("Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Prueba3.pkl", 'rb')
# state_transf_data = pickle.load(Q_input)
# Q_input.close()

# builder.set_params(state_transf_data)
# builder.save_params()