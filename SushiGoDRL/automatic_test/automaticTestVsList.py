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
from agents_sushi_go.multiagent_learning.minimax_multi_deep_q_learning_agent import MinimaxQAgentSashimiPlus

agent_to_test = MinimaxQAgentSashimiPlus()
games_number = 200
    
versus_agents = [
                # RandomAgent()
                #ChopstickLoverAgent(),
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
                # WasabiLoverAgent()
                # WasabiLoverAtFirstAgent(),
                # DeepQLearningAgentPhase1(),
                # DeepQLearningAgentPhase4(),
                DeepQLearningTorchAgentPhase1(),
                DeepQLearningTorchAgentPhase2(),
                DeepQLearningTorchAgentPhase3(),
                DoubleDeepQLearningTorchAgentPhase1(),
                DoubleDeepQLearningTorchAgentPhase2(),
                DoubleDeepQLearningTorchAgentPhase3(),
                DualingDeepQLearningTorchAgentPhase1(),
                DualingDeepQLearningTorchAgentPhase2(),
                DualingDeepQLearningTorchAgentPhase3(),
                MaxminDQLearningTorchAgentPhase1(),
                MaxminDQLearningTorchAgentPhase2(),
                MaxminDQLearningTorchAgentPhase3(),
                MinimaxQAgent()
                ]

logName = "Log" + type(agent_to_test).__name__
if path.exists(logName + ".pkl"):
    agentTypeLogSet = AgentTypeLogSet.load(logName)
else:
    agentTypeLogSet = AgentTypeLogSet()

for versus_agent in versus_agents:

    agents = []
    agents.append(versus_agent)
    
    agentTypeLog = agentTypeLogSet.getLogForAgent(type(versus_agent))
    
    for i in range(0,games_number):
        
        new_game = SingleGame("Original", agents, chopsticks_phase_mode = True)
    
        agent_to_test.set_player(new_game.get_player(0))
    
        print ("Loop " + str(i) + ":\n")
        rival_new_legal_actions = new_game.get_rival_legal_actions_numbers()
        
        second_card = None
        while not new_game.is_finished():            
                        
            force_no_chopsticks_phase = not agent_to_test.trained_with_chopsticks_phase()        
            legal_actions = new_game.get_legal_actions(force_no_chopsticks_phase)
            
            if not agent_to_test.trained_with_chopsticks_phase():
                if new_game.is_in_chopsticks_phase():
                    action = second_card
                   
                else:
                    
                    double_action = agent_to_test.choose_action(legal_actions, rival_new_legal_actions)
            
                    cards = double_action.get_pair_of_cards()                
                    second_card = cards[1]
                    action = cards[0]
            else:
                action = agent_to_test.choose_action(legal_actions, rival_new_legal_actions)
                    
            action_number = 8
            if action is not None:
                action_number = action.get_number()   
            
                            
            turn = (new_game.get_turn() - 1) * 2 + new_game.get_phase() - 1
                        
            for legal_action in legal_actions:
                
                if agent_to_test.trained_with_chopsticks_phase():
                    legal_action_number = 8
                    if legal_action is not None:
                        legal_action_number = legal_action.get_number()
                        
                    agentTypeLog.action_n[legal_action_number] += 1
                    agentTypeLog.action_n_by_turn[turn][legal_action_number] += 1 
                else: 
                    legal_action_number = legal_action.get_action_number()
                    if legal_action_number < 8:
                        agentTypeLog.action_n[legal_action_number] += 1
                        agentTypeLog.action_n_by_turn[turn][legal_action_number] += 1 
                        
            if not agent_to_test.trained_with_chopsticks_phase() and new_game.is_in_chopsticks_phase():
               agentTypeLog.action_n[8] += 1
               agentTypeLog.action_n_by_turn[turn][8] += 1  
            
                    
            agentTypeLog.action_used[action_number] += 1
            agentTypeLog.action_used_by_turn[turn][action_number] += 1
           
            reward = new_game.play_action(action)
            new_legal_actions = new_game.get_legal_actions()
            rival_new_legal_actions = new_game.get_rival_legal_actions_numbers()
            
        winners_list = new_game.declare_winner()
        scores_list = new_game.report_scores()
        print (winners_list)
        print (scores_list)
        
        for i in range(0,len(scores_list)):
            if i in winners_list:
                agentTypeLog.victories_by_player[i] += 1/len(winners_list)
            agentTypeLog.points_by_player[i] += scores_list[i]
        
        agentTypeLog.num_of_games += 1
       
        
    win_perc = agentTypeLog.victories_by_player[0] * 100 / agentTypeLog.num_of_games
    rewards_mean = agentTypeLog.points_by_player[0] / agentTypeLog.num_of_games
    
    f = open("Test results " + type(agent_to_test).__name__ + ".txt", "a")
    
    f.write(type(versus_agent).__name__ + "\t")
    f.write("{:.2f}".format(win_perc) + "\t")
    f.write("{:.2f}".format(rewards_mean) + "\n")

    f.close()
       
    
    print ("Wins per player: ")
    print (agentTypeLog.victories_by_player)
    print ("Total points per player: " )
    print (agentTypeLog.points_by_player)
    
    print ("Actions used: " )
    
agentTypeLogSet.save(logName)    
    
    # print (action_used)
    
    # f = open("Action used " + type(agent_to_test).__name__ + "-" + type(versus_agent).__name__  + ".txt", "a")
    
    # for i in action_used:
    #     f.write(str(i) + "\t")
        
    # f.write("\n")

    # for i in action_n:
    #     f.write(str(i) + "\t")
        
    # f.write("\n")

    # f.close()
        
        
    # print ("% action used: " )    
    
    # print((np.array(action_used) / np.array(action_n)) * 100)

    
    # print ("Actions used by turn: " )
    
    # for turn in action_used_by_turn:
    #     print (turn)
        
        
    # print ("% action used by turn: " )    
    
    # for i in range(10):   
    #     print((np.array(action_used_by_turn[i]) / np.array(action_n_by_turn[i])) * 100)
        
        
    # f = open("Action used by turn " + type(agent_to_test).__name__ + "-" + type(versus_agent).__name__  + ".txt", "a")
    
    # for t in range(10):                  
    #     for i in action_used_by_turn[t]:
    #         f.write(str(i) + "\t")
            
    #     f.write("\n")   
            
    # for t in range(10):   
    #     for i in action_n_by_turn[t]:
    #         f.write(str(i) + "\t")
            
    #     f.write("\n")
        
    # f.close()
        
