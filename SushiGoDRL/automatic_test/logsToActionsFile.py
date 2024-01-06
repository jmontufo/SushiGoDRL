# -*- coding: utf-8 -*-
import numpy as np
import os.path
from os import path

from model_sushi_go.game import SingleGame

from test_logs import AgentTypeLogSet

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


from agents_sushi_go.deep_q_learning.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_agent_v2 import DeepQLearningAgentPhase4
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase3
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase3
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase3
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase3
from agents_sushi_go.multiagent_learning.minimax_multi_deep_q_learning_agent import MinimaxQAgent


agents = [#MinimaxQAgent()
    # QLearningAgentPhase1(),
    #             MCTreeSearchAgentPhase1(),
    #             DeepQLearningAgentPhase1(),
    #             DoubleDeepQLearningAgentPhase1(),
    #             QLearningAgentPhase2(),
    #             MCTreeSearchAgentPhase2(),
    #             DeepQLearningAgentPhase2(),
    #             DoubleDeepQLearningAgentPhase2(),
    #             QLearningAgentPhase3(),
    #             MCTreeSearchAgentPhase3() ,
    #             DeepQLearningAgentPhase3(),
    #             DoubleDeepQLearningAgentPhase3(),
    #             QLearningAgentPhase4(),
    #             MCTreeSearchAgentPhase4(),
    #             DeepQLearningAgentPhase4(),
    #             DoubleDeepQLearningAgentPhase4()
    DoubleDeepQLearningTorchAgentPhase2()            
    ]

f = open("ActionsLog.txt", "a")

for agent in agents:

    
    f.write(type(agent).__name__ + "\t")    

    logName = "Log" + type(agent).__name__
    
    agentTypeLogSet = AgentTypeLogSet.load(logName)
    
    action_used = np.array([0]*9)
    action_n = np.array([0]*9)
    
    for agentType in agentTypeLogSet.log_by_agent:
        
        agent_type_log = agentTypeLogSet.log_by_agent[agentType]
        
        action_used +=   np.array(agent_type_log.action_used)
        action_n +=   np.array(agent_type_log.action_n)    
        
    
    action_percent = (action_used / action_n) * 100
    
    for i in action_percent:
        f.write("{:.2f}".format(i) + "\t")
         
    f.write("\n")   
    
    
f.close()


        
