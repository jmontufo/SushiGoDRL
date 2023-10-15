# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:22:47 2023

@author: jmont
"""


from model_sushi_go.table import Table
from model_sushi_go.player import Player
from model_sushi_go.game import SingleGame
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import PuddingCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import WasabiCard

def test_init():   
    game = SingleGame()
    player = Player(game, 1)
    table = Table(player)
    
    assert table.get_cards() == []
    assert table.get_maki_rolls() == 0
    assert table.get_wasabis() == 0
    assert table.get_chopsticks() == 0
    assert table.get_puddings() == 0

def test_pick_chopsticks_card():
    game = SingleGame()
    player = Player(game, 1)
    table = Table(player)
    
    otherCard = SashimiCard()
    chopsticksCard = ChopsticksCard()
    
    table.add_card(otherCard)
    table.add_card(chopsticksCard)
        
    pickedCard = table.pick_chopsticks_card()
    
    assert table.get_chopsticks() == 0
    assert pickedCard == chopsticksCard
    
    for card in table.get_cards():
        assert card != pickedCard
        
    otherChopsticksCard = ChopsticksCard()
    table.add_card(otherChopsticksCard)
    table.add_card(chopsticksCard)    
    
    pickedCard = table.pick_chopsticks_card()
        
    assert table.get_chopsticks() == 1
    assert pickedCard == otherChopsticksCard
    
    for card in table.get_cards():
        assert card != pickedCard
        
def test_return_cards_except_pudding():
    
    game = SingleGame()
    player = Player(game, 1)
    table = Table(player)
    
    card1 = SashimiCard()
    card2 = PuddingCard()
    card3 = ChopsticksCard()
    card4 = PuddingCard()
    
    table.add_card(card1)
    table.add_card(card2)
    table.add_card(card3)
    table.add_card(card4)
    
    returnedCards = table.return_cards_except_pudding()
    
    assert returnedCards == [card1, card3]
    assert table.get_cards() == [card2, card4]
    assert table.get_maki_rolls() == 0
    assert table.get_wasabis() == 0
    assert table.get_chopsticks() == 0
    assert table.get_puddings() == 2
    
    table = Table(player)
    
    card1 = SashimiCard()
    card2 = OneMakiCard()
    card3 = ChopsticksCard()
    card4 = WasabiCard()
    
    table.add_card(card1)
    table.add_card(card2)
    table.add_card(card3)
    table.add_card(card4)
    
    returnedCards = table.return_cards_except_pudding()
    
    assert returnedCards == [card1, card2, card3, card4]
    assert table.get_cards() == []
    assert table.get_maki_rolls() == 0
    assert table.get_wasabis() == 0
    assert table.get_chopsticks() == 0
    assert table.get_puddings() == 0
    
    table = Table(player)
    
    card1 = PuddingCard()
    card2 = PuddingCard()
    card3 = PuddingCard()
    card4 = PuddingCard()
    
    table.add_card(card1)
    table.add_card(card2)
    table.add_card(card3)
    table.add_card(card4)
    
    returnedCards = table.return_cards_except_pudding()
    
    assert returnedCards == []
    assert table.get_cards() == [card1, card2, card3, card4]
    assert table.get_maki_rolls() == 0
    assert table.get_wasabis() == 0
    assert table.get_chopsticks() == 0
    assert table.get_puddings() == 4
    
def test_find_card_by_type():
    
    game = SingleGame()
    player = Player(game, 1)
    table = Table(player)
    
    card1 = OneMakiCard()
    card2 = TwoMakiCard()
    card3 = ChopsticksCard()
    card4 = PuddingCard()
    
    table.add_card(card1)
    table.add_card(card2)
    table.add_card(card3)
    table.add_card(card4)
    
    assert table.find_card_by_type(TwoMakiCard) == card2
    assert table.find_card_by_type(MakiCard) == card1
    assert table.find_card_by_type(PuddingCard) == card4
    
    def filter_one_maki(maki_card):
        return maki_card.get_value() > 1
        
    assert table.find_card_by_type(MakiCard, filter_one_maki) == card2
    
def test_num_of_cards_of_type():
    
    game = SingleGame()
    player = Player(game, 1)
    table = Table(player)
    
    card1 = OneMakiCard()
    card2 = TwoMakiCard()
    card3 = ChopsticksCard()
    card4 = OneMakiCard()
    
    table.add_card(card1)
    table.add_card(card2)
    table.add_card(card3)
    table.add_card(card4)
    
    assert table.num_of_cards_of_type(TwoMakiCard) == 1
    assert table.num_of_cards_of_type(MakiCard) == 3
    assert table.num_of_cards_of_type(ChopsticksCard) == 1
    assert table.num_of_cards_of_type(PuddingCard) == 0
    
def test_add_card():
    
    game = SingleGame()
    player = Player(game, 1)
    table = Table(player)
    
    card1 = WasabiCard()
    card2 = WasabiCard()
    card3 = ChopsticksCard()
    card4 = TwoMakiCard()
    
    table.add_card(card1)
    table.add_card(card2)
    table.add_card(card3)
    table.add_card(card4)
    
    assert table.get_maki_rolls() == 2
    assert table.get_wasabis() == 2
    assert table.get_chopsticks() == 1
    assert table.get_puddings() == 0
    
    assert card1.get_attached_nigiri() == None
    assert card2.get_attached_nigiri() == None
    
    card5 = EggNigiriCard()        
    table.add_card(card5)
    
    assert card1.get_attached_nigiri() == card5
    assert card2.get_attached_nigiri() == None
    assert card5.get_attached_wasabi() == card1
    assert table.get_wasabis() == 1
        
    card6 = EggNigiriCard() 
    card7 = EggNigiriCard()        
    table.add_card(card6)      
    table.add_card(card7)
    
    assert card1.get_attached_nigiri() == card5
    assert card2.get_attached_nigiri() == card6
    assert card5.get_attached_wasabi() == card1
    assert card6.get_attached_wasabi() == card2
    assert card7.get_attached_wasabi() == None
        
    card8 = OneMakiCard() 
    card9 = PuddingCard()        
    table.add_card(card8)      
    table.add_card(card9)
    
    assert table.get_maki_rolls() == 3
    assert table.get_wasabis() == 0
    assert table.get_chopsticks() == 1
    assert table.get_puddings() == 1
    