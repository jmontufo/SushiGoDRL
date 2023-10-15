# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:52:10 2023

@author: jmont
"""

from model_sushi_go.game import SingleGame
from agents_sushi_go.random_agent import RandomAgent


agents = [RandomAgent()]


game = SingleGame("Original", agents)

while not game.is_finished():
      
      player = game.get_single_player()
      player_hand = player.get_hand()
      player_table = player.get_table()
      number_of_cards = player_hand.get_num_of_cards()
      
      print("")
      print("Player table: ")
      print("")
      
      for card in player_table.get_cards():
          print("\t" + str(card))
      
      print("")        
      print("Your hand: ")
      print("")        
      
      for card_index, card in enumerate(player_hand.get_cards()):
          print("\t" + str(card_index + 1) + " : " + str(card))
              
      max_input = number_of_cards
      
      if player.is_chopsticks_move_available():
          max_input = number_of_cards + 1
          print("\t" + str(number_of_cards + 1) + " : Use Chopsticks!")
          
      
      first_action_number = 0
      second_action_number = None
      
      if first_action_number == number_of_cards + 1:
          
          first_action_number = 0
          second_action_number = 0
        
      cards = []
      
      cards.append(type(player_hand.get_cards()[first_action_number - 1]))
      
      if second_action_number is not None:
          cards.append(type(player_hand.get_cards()[second_action_number - 1])) 
        
      game.play_cards(cards)
      
print("")
print("Scores:")
print("")

for i,score in enumerate(game.report_scores()):        
    print("Player " + str(i+1) + " score: " + str(score))
 
print("")   
if 0 in game.declare_winner():    
    print ("YOU WIN!!!!")
else:
    print ("Your lose. Try again!")