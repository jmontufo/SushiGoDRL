# -*- coding: utf-8 -*-
from model_sushi_go.game import SingleGame

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
# # from agents_sushi_go.deep_q_learning_agent_v2 import DeepQLearningAgentPrueba3
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
# from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3
from agents_sushi_go.deep_q_learning.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DoubleDeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.deep_q_learning_torch_agent import  DualingDeepQLearningTorchAgentPhase2
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase1
from agents_sushi_go.deep_q_learning.maxmin_deep_q_learning_agent import MaxminDQLearningTorchAgentPhase2
from agents_sushi_go.multiagent_learning.minimax_multi_deep_q_learning_agent import MinimaxQAgent


agent_to_test = DeepQLearningAgentPhase1()

agents = []
# agents.append(DeepQLearningTorchAgentPhase2())
# agents.append(DoubleDeepQLearningTorchAgentPhase2())
# agents.append(DualingDeepQLearningTorchAgentPhase2())
agents.append(RandomAgent())

games_number = 500
victories_by_player = [0,0,0,0,0]
points_by_player = [0,0,0,0,0]

action_used = [0]*9

action_used_by_turn = []

for i in range(0,20):
    action_used_by_turn.append([0]*9)

for i in range(0, games_number):
    
    new_game = SingleGame("Original", agents, chopsticks_phase_mode = True)

    agent_to_test.set_player(new_game.get_player(0))

    print ("Loop " + str(i) + ":\n")
    
    rival_new_legal_actions = new_game.get_rival_legal_actions_numbers()
    
    second_card = None
    while not new_game.is_finished():
                
        force_no_chopsticks_phase = not agent_to_test.trained_with_chopsticks_phase()
        legal_actions_player0 = new_game.get_legal_actions(force_no_chopsticks_phase)
        
                        
        if not agent_to_test.trained_with_chopsticks_phase():
        
            if new_game.is_in_chopsticks_phase():
                action = second_card
               
            else:
                double_action = agent_to_test.choose_action(legal_actions_player0, rival_new_legal_actions)
        
                cards = double_action.get_pair_of_cards()                
                second_card = cards[1]
                action = cards[0]
        else:
            action = agent_to_test.choose_action(legal_actions_player0, rival_new_legal_actions)
        
        action_number = 8
        if action is not None:
            action_number = action.get_number()
        
        turn = (new_game.get_turn() - 1) * 2 + new_game.get_phase() - 1
        
        action_used[action_number] += 1
        action_used_by_turn[turn][action_number] += 1
       
        reward = new_game.play_action(action)
        
        done = new_game.is_finished()
        new_legal_actions = new_game.get_legal_actions()
        rival_new_legal_actions = new_game.get_rival_legal_actions_numbers()
        
        # agent_to_test.learn_from_previous_action(reward, done, new_legal_actions)
        
    # agent_to_test.save_training()
        
    winners_list = new_game.declare_winner()
    scores_list = new_game.report_scores()
    print (winners_list)
    print (scores_list)
    
    for i in range(0,len(scores_list)):
        if i in winners_list:
            victories_by_player[i] += 1
        points_by_player[i] += scores_list[i]
    
print ("Wins per player: ")
print (victories_by_player)
print ("Total points per player: " )
print (points_by_player)

print ("Actions used: " )

for turn in action_used_by_turn:
    print (turn)
