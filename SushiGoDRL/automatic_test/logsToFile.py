# -*- coding: utf-8 -*-
import numpy as np
import os.path
from os import path

from model_sushi_go.game import SingleGame

from test_logs import AgentTypeLogSet

# from agents_sushi_go.random_agent import RandomAgent
# from agents_sushi_go.card_lover_agent import SashimiLoverAgent
# from agents_sushi_go.card_lover_agent import SashimiSuperLoverAgent
# from agents_sushi_go.card_lover_agent import SashimiHaterAgent
# from agents_sushi_go.card_lover_agent import TempuraLoverAgent
# from agents_sushi_go.card_lover_agent import TempuraSuperLoverAgent
# from agents_sushi_go.card_lover_agent import DumplingLoverAgent
# from agents_sushi_go.card_lover_agent import DumplingSuperLoverAgent
# from agents_sushi_go.card_lover_agent import MakiLoverAgent
# from agents_sushi_go.card_lover_agent import MakiSuperLoverAgent
# from agents_sushi_go.card_lover_agent import MakiHaterAgent
# from agents_sushi_go.card_lover_agent import WasabiLoverAgent
# from agents_sushi_go.card_lover_agent import WasabiLoverAtFirstAgent
# from agents_sushi_go.card_lover_agent import NigiriLoverAgent
# from agents_sushi_go.card_lover_agent import NigiriSuperLoverAgent
# from agents_sushi_go.card_lover_agent import PuddingLoverAgent
# from agents_sushi_go.card_lover_agent import PuddingSuperLoverAgent
# from agents_sushi_go.card_lover_agent import PuddingHaterAgent
# from agents_sushi_go.card_lover_agent import ChopstickLoverAgent
# from agents_sushi_go.card_lover_agent import ChopstickHaterAgent
# from agents_sushi_go.card_lover_agent import ChopstickLoverAtFirstAgent


# from agents_sushi_go.q_learning_agent import QLearningAgentPhase1
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase2
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase3
# from agents_sushi_go.q_learning_agent import QLearningAgentPhase4
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase4
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase3
# from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPhase4
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase1
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase2
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase3
# from agents_sushi_go.deep_q_learning_agent_v2 import DoubleDeepQLearningAgentPhase4



logName = "LogDoubleDeepQLearningTorchAgentPhase2"

agentTypeLogSet = AgentTypeLogSet.load(logName)

for agentType in agentTypeLogSet.log_by_agent:
    
    agent_type_log = agentTypeLogSet.log_by_agent[agentType]
    
    win_perc = agent_type_log.victories_by_player[0]* 100 / agent_type_log.num_of_games
    rewards_mean = agent_type_log.points_by_player[0] / agent_type_log.num_of_games
    
    rival_win_perc = agent_type_log.victories_by_player[1]* 100 / agent_type_log.num_of_games
    rival_rewards_mean = agent_type_log.points_by_player[1] / agent_type_log.num_of_games
    
    f = open(logName + ".txt", "a")
    
    f.write(agentType.__name__ + "\t")
    f.write("{:.2f}".format(win_perc) + "\t")
    f.write("{:.2f}".format(rewards_mean) + "\t")
    f.write("{:.2f}".format(rival_win_perc) + "\t")
    f.write("{:.2f}".format(rival_rewards_mean) + "\n")

    f.close()
    

        
