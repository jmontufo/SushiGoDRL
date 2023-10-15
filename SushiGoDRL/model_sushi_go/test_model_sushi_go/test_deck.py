# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:52:32 2023

@author: jmont
"""

from model_sushi_go.deck import Deck
from model_sushi_go.game import SingleGame
from model_sushi_go.card import CARD_TYPES
from model_sushi_go.card import WasabiCard
from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import SalmonNigiriCard
from model_sushi_go.card import SquidNigiriCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import DumplingCard
from model_sushi_go.card import PuddingCard

def get_cards_in_deck_by_type(deck):    
    num_cards_by_type = {}
    for card_type in CARD_TYPES.get_card_types():
        num_cards_by_type[card_type] = 0
    
    for card in deck.get_deck_cards():
        num_cards_by_type[type(card)] += 1
    
    return num_cards_by_type

def add_cards_in_deck_by_type(num_cards_by_type, cards):    
    for card in cards:
        num_cards_by_type[type(card)] += 1
        
    return num_cards_by_type

def assert_deck_has_the_original_game_cards(cards_by_type):    
    assert cards_by_type[ChopsticksCard] == 4
    assert cards_by_type[EggNigiriCard] == 5
    assert cards_by_type[SalmonNigiriCard] == 10
    assert cards_by_type[SquidNigiriCard] == 5
    assert cards_by_type[WasabiCard] == 6
    assert cards_by_type[OneMakiCard] == 6
    assert cards_by_type[TwoMakiCard] == 12
    assert cards_by_type[ThreeMakiCard] == 8
    assert cards_by_type[SashimiCard] == 14
    assert cards_by_type[TempuraCard] == 14
    assert cards_by_type[DumplingCard] == 14
    assert cards_by_type[PuddingCard] == 10
    
def assert_deck_has_the_party_original_game_cards(cards_by_type):    
    
    assert cards_by_type[ChopsticksCard] == 3
    assert cards_by_type[EggNigiriCard] == 4
    assert cards_by_type[SalmonNigiriCard] == 5
    assert cards_by_type[SquidNigiriCard] == 3
    assert cards_by_type[WasabiCard] == 3
    assert cards_by_type[OneMakiCard] == 4
    assert cards_by_type[TwoMakiCard] == 5
    assert cards_by_type[ThreeMakiCard] == 3
    assert cards_by_type[SashimiCard] == 8
    assert cards_by_type[TempuraCard] == 8
    assert cards_by_type[DumplingCard] == 8
    assert cards_by_type[PuddingCard] == 5        

def test_init(): 
    # Original game
    game = SingleGame("Original")
    deck = Deck(game)
    
    num_cards_by_type = get_cards_in_deck_by_type(deck)
    assert_deck_has_the_original_game_cards(num_cards_by_type)
    
    # Party game with original sets
    game = SingleGame("Party-original")
    deck = Deck(game)
    
    num_cards_by_type = get_cards_in_deck_by_type(deck)
    assert_deck_has_the_party_original_game_cards(num_cards_by_type)

def test_get_game():        
    game = SingleGame("Original")
    deck = Deck(game)
    
    assert deck.get_game() == game
    
def test_get_set_deck_cards():    
    game = SingleGame("Original")
    deck = Deck(game)
    
    cards_list = [OneMakiCard(), TempuraCard()]
    
    deck.set_deck_cards(cards_list)
    assert deck.get_deck_cards() == cards_list
    
def test_deal_hand():
    game = SingleGame("Original")
    deck = Deck(game)
    
    print(len(deck.get_deck_cards()))
    hand_dealt = deck.deal_hand()
    
    assert len(hand_dealt) == 10
    
    num_cards_by_type = get_cards_in_deck_by_type(deck)
    num_cards_by_type = add_cards_in_deck_by_type(num_cards_by_type, hand_dealt) 
    
    assert_deck_has_the_original_game_cards(num_cards_by_type)

# TO DO: Only if party is going to be used. 
# def test_add_desert():
# def test_take_back_cards():