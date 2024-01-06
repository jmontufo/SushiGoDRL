# -*- coding: utf-8 -*-
import numpy as np
import os.path
from os import path

from model_sushi_go.game import SingleGame

from test_logs import AgentTypeLogSet


type_name = "MinimaxQAgentSashimiPlus"
fileName =  type_name + "-ActionsByTurn"

f = open(fileName + ".txt", "a")

logName = "Log" + type_name

agentTypeLogSet = AgentTypeLogSet.load(logName)

action_used_by_turn = []
action_n_by_turn = []

for i in range(10):
    action_used_by_turn.append(np.array([0]*9))
    action_n_by_turn.append(np.array([0]*9))

for agentType in agentTypeLogSet.log_by_agent:
    
    agent_type_log = agentTypeLogSet.log_by_agent[agentType]
    
    for i in range(10):
    
        action_used_by_turn[i] +=  np.array(agent_type_log.action_used_by_turn[i*2])
        action_used_by_turn[i] +=  np.array(agent_type_log.action_used_by_turn[i*2 + 1])
        action_n_by_turn[i] +=  np.array(agent_type_log.action_n_by_turn[i*2])
        action_n_by_turn[i] +=  np.array(agent_type_log.action_n_by_turn[i*2 + 1])
        
action_percent = []

for i in range(10):    
    
    action_used = action_used_by_turn[i]
    action_n = action_n_by_turn[i]
     
    percents = []
    for i in range(len(action_n)):
        if action_n[i] == 0:
            percents.append(0)
        else:
            percents.append((action_used[i] / action_n[i]) * 100)
    
    action_percent.append(percents)

for j in range(10):
    for i in range(9):
        f.write("{:.2f}".format(action_percent[j][i]) + "\t")
    f.write("\n")
     
f.write("\n")   

    
f.close()


        
