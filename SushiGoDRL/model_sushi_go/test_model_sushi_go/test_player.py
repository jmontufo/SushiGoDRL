# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:14:13 2023

@author: jmont
"""

from model_sushi_go.player import Player, TestPlayer
from model_sushi_go.table import Table
from model_sushi_go.game import SingleGame
from model_sushi_go.game import MultiplayerGame
from model_sushi_go.hand import Hand
from model_sushi_go.card import MakiCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import WasabiCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import PuddingCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import SquidNigiriCard
from agents_sushi_go.random_agent import RandomAgent

def test_init():
    game = SingleGame()
    player = Player(game, 1)
    
    assert player.get_game() == game
    assert player.get_position() == 1
    assert player.get_hand().get_num_of_cards() == 10    
    assert player.get_awaiting_hand() == None
    assert player.get_table().get_cards() == []
    assert player.get_current_score() == 0
    assert player.get_last_action_reward() == 0
    
def test_get_game():
    
    game = SingleGame()
    player = game.get_player(0)
        
    assert player.get_game() == game
    
def test_get_set_hand():
    
    game = SingleGame()
    player = game.get_player(0)
    
    hand_cards = [OneMakiCard(), 
                  TwoMakiCard(), 
                  ThreeMakiCard()]    
    hand = Hand(hand_cards)
    
    TestPlayer.set_hand(player, hand)
    assert player.get_hand() == hand  

def test_get_set_awaiting_hand():
    
    game = SingleGame()
    player = game.get_player(0)
    
    awaiting_hand_cards = [OneMakiCard(), 
                           TwoMakiCard(), 
                           ThreeMakiCard()]    
    awaiting_hand = Hand(awaiting_hand_cards)
    
    TestPlayer.set_awaiting_hand(player, awaiting_hand)
    assert player.get_awaiting_hand() == awaiting_hand   
    
def test_get_position():
      
    game = MultiplayerGame("Original", 4)
    player0 = game.get_player(0)
    player1 = game.get_player(1)
    player2 = game.get_player(2)
    player3 = game.get_player(3)
          
    assert player0.get_position() == 0  
    assert player1.get_position() == 1  
    assert player2.get_position() == 2  
    assert player3.get_position() == 3  
      
    game = SingleGame("Original", [RandomAgent(), 
                                   RandomAgent(), 
                                   RandomAgent()])
    player0 = game.get_player(0)
    player1 = game.get_player(1)
    player2 = game.get_player(2)
    player3 = game.get_player(3)
          
    assert player0.get_position() == 0  
    assert player1.get_position() == 1  
    assert player2.get_position() == 2  
    assert player3.get_position() == 3  
    
def test_add_partial_score():
    
    game = SingleGame()
    player = game.get_player(0)
    
    player.add_partial_score(3)
    
    assert player.get_current_score() == 3
    assert player.get_last_action_reward() == 3
    
    player.add_partial_score(2)
    
    assert player.get_current_score() == 5
    assert player.get_last_action_reward() == 5
    
   
def test_initialize_player_hand():    
    
    game = SingleGame()
    player = game.get_player(0)
    
    hand_cards = [OneMakiCard(), 
                  TwoMakiCard(), 
                  ThreeMakiCard()]    
    hand = Hand(hand_cards)
    
    TestPlayer.set_hand(player, hand)
    player.initialize_player_hand()
    assert player.get_hand().get_num_of_cards() == 10 
    
def test_is_chopsticks_move_available():    
    game = SingleGame()
    player = game.get_player(0)
    table = player.get_table()
    hand = Hand()
    
    table.add_card(ChopsticksCard())
    
    assert player.is_chopsticks_move_available() == True
    
    table.return_cards_except_pudding()
    table.add_card(WasabiCard())
    
    assert player.is_chopsticks_move_available() == False
    
    table.add_card(ChopsticksCard())
    TestPlayer.set_hand(player, hand)
    
    assert player.is_chopsticks_move_available() == False
    
    hand.add_card(ChopsticksCard())
    
    assert player.is_chopsticks_move_available() == False
    
    hand.add_card(ChopsticksCard())
    
    assert player.is_chopsticks_move_available() == True
  
def test_get_table():
    game = SingleGame()
    player = game.get_player(0)
    table = Table(player)
    
    TestPlayer.set_table(player, table)
    assert player.get_table() == table
    
      
def test_get_maki_rolls():
    game = SingleGame()
    player = game.get_player(0)
    table = player.get_table()
    
    assert player.get_maki_rolls() == 0
    
    table.add_card(ChopsticksCard())
    assert player.get_maki_rolls() == 0
        
    table.add_card(TwoMakiCard())
    assert player.get_maki_rolls() == 2
        
    table.add_card(OneMakiCard())
    assert player.get_maki_rolls() == 3
    
def test_get_puddings():
    
    game = SingleGame()
    player = game.get_player(0)
    table = player.get_table()
    
    assert player.get_puddings() == 0
    
    table.add_card(TwoMakiCard())
    assert player.get_puddings() == 0
        
    table.add_card(PuddingCard())
    assert player.get_puddings() == 1
        
def test_get_hand_cards():
    
    game = SingleGame()
    player = game.get_player(0)
    cards = [TwoMakiCard(), PuddingCard()]    
    hand = Hand(cards)    
    TestPlayer.set_hand(player, hand)
    
    assert player.get_hand_cards() == cards
    
def test_play_a_card():
    game = SingleGame()
    
    player = game.get_player(0)
    playedCard = SquidNigiriCard()
    otherCard = PuddingCard()
    cards = [playedCard, otherCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
    tempuraCard = TempuraCard()
    player.get_table().add_card(tempuraCard)
    
    other_player = game.get_player(1)      
    
    player.play_a_turn(NigiriCard)
    assert other_player.get_awaiting_hand().get_cards() == [otherCard]
    assert player.get_current_score() == 3
    assert player.get_last_action_reward() == 3
    assert player.get_table().get_cards() == [tempuraCard, playedCard]
    
    other_player.take_cards_from_other_player()
    nextPlayedCard = TempuraCard()
    cards = [nextPlayedCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
        
    player.play_a_turn(TempuraCard)
    assert other_player.get_awaiting_hand().get_cards() == []
    assert player.get_current_score() == 8
    assert player.get_last_action_reward() == 5
    assert player.get_table().get_cards() == [tempuraCard, 
                                              playedCard, 
                                              nextPlayedCard]
    
def test_play_a_card_with_chopsticks():
     game = SingleGame()
     
     player = game.get_player(0)
     playedCard = WasabiCard()
     playedCard2 = SquidNigiriCard()
     cards = [playedCard, playedCard2]    
     hand = Hand(cards)
     TestPlayer.set_hand(player, hand)
     chopsticksCard = ChopsticksCard()
     player.get_table().add_card(chopsticksCard)
     
     other_player = game.get_player(1)      
     
     player.play_a_turn(WasabiCard, NigiriCard)
     assert other_player.get_awaiting_hand().get_cards() == [chopsticksCard]
     assert player.get_current_score() == 9
     assert player.get_last_action_reward() == 9
     assert player.get_table().get_cards() == [playedCard, playedCard2]    
    
def test_take_cards_from_other_player():
    
    game = SingleGame()    
    player = game.get_player(0)
    
    awaiting_hand_cards = [OneMakiCard(), 
                           TwoMakiCard(), 
                           ThreeMakiCard()]    
    awaiting_hand = Hand(awaiting_hand_cards)
    
    TestPlayer.set_awaiting_hand(player, awaiting_hand)
    player.take_cards_from_other_player()
    
    assert player.get_hand() == awaiting_hand
    assert player.get_awaiting_hand() == None
    
def test_give_back_cards():
    
    game = SingleGame()    
    player = game.get_player(0)
    puddingCard = PuddingCard()
    player.get_table().add_card(OneMakiCard())
    player.get_table().add_card(puddingCard)
    player.get_table().add_card(ThreeMakiCard())
    
    player.give_back_cards()
    
    assert player.get_table().get_cards() == [puddingCard]
    
    