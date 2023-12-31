# -*- coding: utf-8 -*-

from model_sushi_go.game import MultiplayerGame

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
from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase1
from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase2
from agents_sushi_go.deep_q_learning_agent import DeepQLearningAgentPhase3
from agents_sushi_go.deep_q_learning_agent import DoubleDeepQLearningAgentPhase1
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase1
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase2
from agents_sushi_go.mc_tree_search_agent import  MCTreeSearchAgentPhase3

num_players = 2

games_number = 300
victories_by_player = [0] * num_players
points_by_player = [0] * num_players

agents = [DeepQLearningAgentPhase1(), RandomAgent()]
    
for i in range(0,games_number):
    
    new_game = MultiplayerGame("Original", num_players)

    for j in range(num_players):
        agents[j].set_player(new_game.get_player(j))

    print ("Loop " + str(i) + ":\n")
    while not new_game.is_finished():
              
        legal_actions_player = new_game.get_legal_actions()
        
        cards = []
        for k in range(0, num_players):
            var  = agents[k].choose_action(legal_actions_player[k])            
            cards.append(var.get_pair_of_cards())   
               
        reward = new_game.play_cards(cards) 
        
    winners_list = new_game.declare_winner()
    scores_list = new_game.report_scores()
    print (winners_list)
    print (scores_list)
    
    for m in range(0,num_players):
        if m in winners_list:
            victories_by_player[m] += 1
        points_by_player[m] += scores_list[m]
    
print ("Wins per player: ")
print (victories_by_player)
print ("Total points per player: " )
print (points_by_player)
