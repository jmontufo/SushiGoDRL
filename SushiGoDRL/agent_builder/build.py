# -*- coding: utf-8 -*-

import pickle


from agent_builder.utils import *


from agent_builder.deep_q_learning.deep_q_learning import DQL_Builder
from agent_builder.reinforce.reinforce import PG_Reinforce_Builder
from agent_builder.reinforce.policy_gradients import PG_Builder
# from agent_builder.double_deep_q_learning.double_deep_q_learning import DDQL_Builder

from states_sushi_go.complete_state_fixed import CompleteState
from states_sushi_go.with_phase.complete_state import CompleteMemoryState
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
from agents_sushi_go.q_learning_agent import QLearningAgentPhase1
from agents_sushi_go.q_learning_agent import QLearningAgentPhase2
from agents_sushi_go.q_learning_agent import QLearningAgentPhase3
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3
from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase3
from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase3

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
total_episodes = 20000
learning_rate = 0.005     
discount = 0.9
max_epsilon = 1		             
min_epsilon = 0.05            
decay_rate = 0.0005   
        
reference = "Phase1"     

# previous_filename = "Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_5000_Phase4-1" 
previous_filename = None

# builder = DQL_Builder(num_players, versus_agents, state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, reference, previous_filename)
    
builder = PG_Builder(num_players, versus_agents, state_type, total_episodes, learning_rate, discount, reference, previous_filename)
# builder = DDQL_Builder(num_players, versus_agents, state_type, total_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, reference, previous_filename)

builder.run()


# Q_input = open("Deep_Q_Learning_2p_TwoPlayersCompleteState_lr1_20000_Prueba3.pkl", 'rb')
# state_transf_data = pickle.load(Q_input)
# Q_input.close()

# builder.set_params(state_transf_data)
# builder.save_params()