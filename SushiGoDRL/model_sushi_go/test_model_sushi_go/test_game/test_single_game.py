# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:52:32 2023

@author: jmont
"""

from model_sushi_go.game import SingleGame
from model_sushi_go.game import TestSingleGame
from agents_sushi_go.random_agent import RandomAgent
from agents_sushi_go.card_lover_agent import SashimiLoverAgent
from agents_sushi_go.card_lover_agent import MakiLoverAgent
import model_sushi_go.test_model_sushi_go.test_deck as t_d
import model_sushi_go.test_model_sushi_go.test_action.test_action_manager as t_a_m
from model_sushi_go.card import SquidNigiriCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import PuddingCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.hand import Hand
from model_sushi_go.player import TestPlayer
from model_sushi_go.action import Action, ChopsticksAction

def test_init():
    
    agents =  [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]
    game = SingleGame("Original", agents)
    
    assert game.get_num_players() == 5
    assert game.get_setup() == "Original"
    
    deck = game.get_deck()
    
    cards_by_type = t_d.get_cards_in_deck_by_type(deck)
    
    position = 0
    for player in game.get_players():
        
        assert player.get_position() == position
        position = position + 1
        
        hand = player.get_hand().get_cards()
        cards_by_type = t_d.add_cards_in_deck_by_type(cards_by_type, hand) 
     
    t_d.assert_deck_has_the_original_game_cards(cards_by_type)
    
    assert not game.is_party_version()
    assert game.get_cards_by_round() == 7
    assert game.get_round() == 1
    assert game.get_turn() == 1
    assert game.get_log() == []
    
    game_agents = TestSingleGame.get_agents(game)
    assert game_agents == agents
    
    for agent_index, agent in enumerate(agents):
        
        player_number = agent_index + 1
        assert agent.get_player() == game.get_player(player_number)

def test_get_single_player():
    # The single player is the one without automatic agent
    
    agents =  [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]
    game = SingleGame("Original", agents)
    
    single_player = game.get_single_player()
    
    for agent in agents:        
        assert agent.get_player() != single_player
        
def test_get_legal_actions():    
    agents =  [RandomAgent()]
    game = SingleGame("Original", agents)
    player = game.get_player(0)
    
    firstCard = SquidNigiriCard()
    secondCard = PuddingCard()
    cards = [firstCard, secondCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
        
    table = player.get_table()
    table.add_card(ChopsticksCard())
    
    expected_actions = [Action(NigiriCard), 
                        Action(PuddingCard), 
                        ChopsticksAction(NigiriCard, PuddingCard)]
    actions = game.get_legal_actions()
    
    assert t_a_m.same_unordered_list_of_actions(actions, expected_actions)
    
def test_get_legal_actions_numbers():    
    agents =  [RandomAgent()]
    game = SingleGame("Original", agents)
    player = game.get_player(0)
    
    firstCard = SquidNigiriCard()
    secondCard = PuddingCard()
    cards = [firstCard, secondCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
    
    actions = game.get_legal_actions_numbers()
    
    assert actions ==  [1, 7] or actions == [7, 1]
    
def test_play_cards():
    
    agent = SashimiLoverAgent()
    game = SingleGame("Original", [agent])
    player = game.get_player(0)
    
    firstCard = SquidNigiriCard()
    secondCard = PuddingCard()
    cards = [firstCard, secondCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
        
    other_player = agent.get_player()
    other_card = ChopsticksCard()
    other_hand = Hand([SashimiCard(), other_card])
    TestPlayer.set_hand(other_player, other_hand)
    
    assert game.play_cards([SquidNigiriCard, None]) == 3
    assert player.get_hand_cards() == [other_card]
    assert other_player.get_hand_cards() == [secondCard]
    assert game.get_turn() == 2
    assert game.get_round() == 1
 
def test_play_cards_end_of_round():   
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2])
    player = game.get_player(0)
    
    firstCard = SashimiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
    
    puddingInTable = PuddingCard()
    player.get_table().add_card(SashimiCard())
    player.get_table().add_card(SashimiCard())
    player.get_table().add_card(puddingInTable)
    
    TestPlayer.set_score(player, 22)    
        
    other_player1 = agent1.get_player()
    other_card1 = ThreeMakiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
    
    other_player2 = agent2.get_player()
    other_card2 = TwoMakiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)

    TestSingleGame.set_turn(game, 9)
    
    assert game.play_action(Action(SashimiCard)) == 10
    assert other_player1.get_last_action_reward() == 6
    assert other_player2.get_last_action_reward() == 3
    assert len(player.get_hand_cards()) == 9
    assert len(other_player1.get_hand_cards()) == 9
    assert len(other_player2.get_hand_cards()) == 9
    assert player.get_table().get_cards() == [puddingInTable]
    assert other_player1.get_table().get_cards() == []
    assert other_player2.get_table().get_cards() == []
    assert game.get_turn() == 1
    assert game.get_round() == 2
    
def test_play_cards_end_of_round_distributed_maki():   
    
    # First place distributed, no rest
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    agent3 = MakiLoverAgent()
    agent4 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2, agent3, agent4])
    player = game.get_player(0)
    
    firstCard = TwoMakiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
            
    other_player1 = agent1.get_player()
    other_card1 = TwoMakiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
    
    other_player2 = agent2.get_player()
    other_card2 = TwoMakiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)    
    
    other_player3 = agent3.get_player()
    other_card3 = TwoMakiCard()
    other_hand3 = Hand([other_card3])
    TestPlayer.set_hand(other_player3, other_hand3)
    
    other_player4 = agent4.get_player()
    other_card4 = TwoMakiCard()
    other_hand4 = Hand([other_card4])
    TestPlayer.set_hand(other_player4, other_hand4)

    TestSingleGame.set_turn(game, 7)
    
    assert game.play_action(Action(MakiCard)) == 1
    assert other_player1.get_last_action_reward() == 1
    assert other_player2.get_last_action_reward() == 1
    assert other_player3.get_last_action_reward() == 1
    assert other_player4.get_last_action_reward() == 1
    assert len(player.get_hand_cards()) == 7
    assert game.get_turn() == 1
    assert game.get_round() == 2
    
    # Second place distributed, no rest
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    agent3 = MakiLoverAgent()
    agent4 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2, agent3, agent4])
    player = game.get_player(0)
    
    firstCard = ThreeMakiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
            
    other_player1 = agent1.get_player()
    other_card1 = TwoMakiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
    
    other_player2 = agent2.get_player()
    other_card2 = TwoMakiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)    
    
    other_player3 = agent3.get_player()
    other_card3 = OneMakiCard()
    other_hand3 = Hand([other_card3])
    TestPlayer.set_hand(other_player3, other_hand3)
    
    other_player4 = agent4.get_player()
    other_card4 = OneMakiCard()
    other_hand4 = Hand([other_card4])
    TestPlayer.set_hand(other_player4, other_hand4)

    TestSingleGame.set_turn(game, 7)
    
    assert game.play_action(Action(MakiCard)) == 6
    assert other_player1.get_last_action_reward() == 1
    assert other_player2.get_last_action_reward() == 1
    assert other_player3.get_last_action_reward() == 0
    assert other_player4.get_last_action_reward() == 0
    assert len(player.get_hand_cards()) == 7
    assert game.get_turn() == 1
    assert game.get_round() == 2
    
    # First place distributed, nothing for second
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    agent3 = MakiLoverAgent()
    agent4 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2, agent3, agent4])
    player = game.get_player(0)
    
    firstCard = ThreeMakiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
            
    other_player1 = agent1.get_player()
    other_card1 = ThreeMakiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
    
    other_player2 = agent2.get_player()
    other_card2 = TwoMakiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)    
    
    other_player3 = agent3.get_player()
    other_card3 = OneMakiCard()
    other_hand3 = Hand([other_card3])
    TestPlayer.set_hand(other_player3, other_hand3)
    
    other_player4 = agent4.get_player()
    other_card4 = OneMakiCard()
    other_hand4 = Hand([other_card4])
    TestPlayer.set_hand(other_player4, other_hand4)

    TestSingleGame.set_turn(game, 7)
    
    assert game.play_action(Action(MakiCard)) == 3
    assert other_player1.get_last_action_reward() == 3
    assert other_player2.get_last_action_reward() == 0
    assert other_player3.get_last_action_reward() == 0
    assert other_player4.get_last_action_reward() == 0
    assert len(player.get_hand_cards()) == 7
    assert game.get_turn() == 1
    assert game.get_round() == 2
    
 
def test_play_cards_end_of_game():   
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2])
    player = game.get_player(0)
    
    firstCard = SashimiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
    
    player.get_table().add_card(SashimiCard())
    player.get_table().add_card(SashimiCard())
    
    TestPlayer.set_score(player, 22)    
        
    other_player1 = agent1.get_player()
    other_card1 = ThreeMakiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
    
    other_player2 = agent2.get_player()
    other_card2 = TwoMakiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)

    TestSingleGame.set_turn(game, 9)
    TestSingleGame.set_round(game, 3)
    
    assert game.play_action_number(4) == 10
    assert other_player1.get_last_action_reward() == 6
    assert other_player2.get_last_action_reward() == 3
    assert len(player.get_hand_cards()) == 9
    assert len(other_player1.get_hand_cards()) == 9
    assert len(other_player2.get_hand_cards()) == 9
    assert player.get_table().get_cards() == []
    assert other_player1.get_table().get_cards() == []
    assert other_player2.get_table().get_cards() == []
    assert game.get_turn() == 1
    assert game.get_round() == 4
    assert game.is_finished() == True
    
def test_play_cards_end_of_game_distribute_pudding():   
    
    # When two players, only positive points for pudding
    agent1 = MakiLoverAgent()
    game = SingleGame("Original", [agent1])
    player = game.get_player(0)
    
    firstCard = SashimiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
    
    puddingInTable = PuddingCard()
    player.get_table().add_card(SashimiCard())
    player.get_table().add_card(SashimiCard())
    player.get_table().add_card(puddingInTable)
    
    TestPlayer.set_score(player, 22)    
        
    other_player1 = agent1.get_player()
    other_card1 = ThreeMakiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
    

    TestSingleGame.set_turn(game, 10)
    TestSingleGame.set_round(game, 3)
    
    # 10 for Sahsimi + 6 for Pudding + 3 for second place Maki 
    assert game.play_action_number(4) == 19 
    assert other_player1.get_last_action_reward() == 6
    assert len(player.get_hand_cards()) == 10
    assert len(other_player1.get_hand_cards()) == 10
    assert player.get_table().get_cards() == [puddingInTable]
    assert game.get_turn() == 1
    assert game.get_round() == 4
    assert game.is_finished() == True
    
    
    # Distribute positive and negative, no rest
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    agent3 = MakiLoverAgent()
    agent4 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2, agent3, agent4])
    player = game.get_player(0)
    
    firstCard = SashimiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
        
    other_player1 = agent1.get_player()
    other_card1 = SashimiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
        
    other_player2 = agent2.get_player()
    other_card2 = SashimiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)
        
    other_player3 = agent3.get_player()
    other_card3 = PuddingCard()
    other_hand3 = Hand([other_card3])
    TestPlayer.set_hand(other_player3, other_hand3)
        
    other_player4 = agent4.get_player()
    other_card4 = PuddingCard()
    other_hand4 = Hand([other_card4])
    TestPlayer.set_hand(other_player4, other_hand4)

    TestSingleGame.set_turn(game, 7)
    TestSingleGame.set_round(game, 3)
    
    assert game.play_action_number(4) == -1 
    assert other_player1.get_last_action_reward() == -1
    assert other_player2.get_last_action_reward() == -1
    assert other_player3.get_last_action_reward() == 4
    assert other_player4.get_last_action_reward() == 4
    assert game.get_turn() == 1
    assert game.get_round() == 4
    assert game.is_finished() == True
    
def test_declare_winner():   
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    agent3 = MakiLoverAgent()
    agent4 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2, agent3, agent4])
    player = game.get_player(0)
    
    firstCard = SashimiCard()
    cards = [firstCard]    
    hand = Hand(cards)
    TestPlayer.set_hand(player, hand)
        
    other_player1 = agent1.get_player()
    other_card1 = SashimiCard()
    other_hand1 = Hand([other_card1])
    TestPlayer.set_hand(other_player1, other_hand1)
        
    other_player2 = agent2.get_player()
    other_card2 = SashimiCard()
    other_hand2 = Hand([other_card2])
    TestPlayer.set_hand(other_player2, other_hand2)
        
    other_player3 = agent3.get_player()
    other_card3 = PuddingCard()
    other_hand3 = Hand([other_card3])
    TestPlayer.set_hand(other_player3, other_hand3)
        
    other_player4 = agent4.get_player()
    other_card4 = PuddingCard()
    other_hand4 = Hand([other_card4])
    TestPlayer.set_hand(other_player4, other_hand4)

    TestSingleGame.set_turn(game, 7)
    TestSingleGame.set_round(game, 3)
    
    
    # One winner by score

    TestPlayer.set_score(player, 22)        
    TestPlayer.set_score(other_player1, 12)        
    TestPlayer.set_score(other_player2, 12)    
    TestPlayer.set_score(other_player3, 12)    
    TestPlayer.set_score(other_player4, 12)    
    
    assert game.play_action_number(4) == -1 
    assert game.declare_winner() == [0]
    
    # Tie by score, tie in puddings
    TestPlayer.set_score(player, 22)        
    TestPlayer.set_score(other_player1, 22)        
    TestPlayer.set_score(other_player2, 12)    
    TestPlayer.set_score(other_player3, 12)    
    TestPlayer.set_score(other_player4, 12)   
    
    assert game.declare_winner() == [0, 1]
    
    # Tie by score, winners by puddings
    TestPlayer.set_score(player, 12)        
    TestPlayer.set_score(other_player1, 22)        
    TestPlayer.set_score(other_player2, 22)    
    TestPlayer.set_score(other_player3, 22)    
    TestPlayer.set_score(other_player4, 22)   
    
    assert game.declare_winner() == [3, 4]
    
def test_report_scores():   
    
    agent1 = MakiLoverAgent()
    agent2 = MakiLoverAgent()
    agent3 = MakiLoverAgent()
    agent4 = MakiLoverAgent()
    game = SingleGame("Original", [agent1, agent2, agent3, agent4])
    player = game.get_player(0)
    other_player1 = agent1.get_player()
    other_player2 = agent2.get_player()
    other_player3 = agent3.get_player()
    other_player4 = agent4.get_player()
    
    # One winner by score

    TestPlayer.set_score(player, 22)        
    TestPlayer.set_score(other_player1, 32)        
    TestPlayer.set_score(other_player2, 12)    
    TestPlayer.set_score(other_player3, 42)    
    TestPlayer.set_score(other_player4, 52)    
    
    assert game.report_scores() == [22,32,12,42,52]