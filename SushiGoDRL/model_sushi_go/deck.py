# -*- coding: utf-8 -*-

import numpy as np

from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import SalmonNigiriCard
from model_sushi_go.card import SquidNigiriCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import DumplingCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import WasabiCard
from model_sushi_go.card import PuddingCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard

class Deck(object):
    
    def __init__(self, game):
                
        self.__game = game
        self.__deck_cards = []
        
        if game.get_setup() == "Party-original":
            self.__init_deck_cards_party_original()            
        elif game.get_setup() == "Original":
            self.__init_deck_cards_original()                            
    
    def __init_deck_cards_original(self):
                
        for i in range(1,109):
            
            if i<=5:
                self.__deck_cards.append(EggNigiriCard())
            elif i <= 15:
                self.__deck_cards.append(SalmonNigiriCard())
            elif i <= 20:
                self.__deck_cards.append(SquidNigiriCard())
            elif i <= 34:
                self.__deck_cards.append(TempuraCard())
            elif i <= 48:
                self.__deck_cards.append(SashimiCard())
            elif i <= 62:
                self.__deck_cards.append(DumplingCard())
            elif i <= 66:
                self.__deck_cards.append(ChopsticksCard())
            elif i <= 72:
                self.__deck_cards.append(WasabiCard())
            elif i <= 82:
                self.__deck_cards.append(PuddingCard())
            elif i <= 88:
                self.__deck_cards.append(OneMakiCard())
            elif i <= 100:
                self.__deck_cards.append(TwoMakiCard())
            elif i <= 108:
                self.__deck_cards.append(ThreeMakiCard())
        
    # Create initial deck of cards
    def __init_deck_cards_party_original(self):
                
        for i in range(1,60):
            
            if i<=4:
                self.__deck_cards.append(EggNigiriCard())
            elif i <= 9:
                self.__deck_cards.append(SalmonNigiriCard())
            elif i <= 12:
                self.__deck_cards.append(SquidNigiriCard())
            elif i <= 20:
                self.__deck_cards.append(TempuraCard())
            elif i <= 28:
                self.__deck_cards.append(SashimiCard())
            elif i <= 36:
                self.__deck_cards.append(DumplingCard())
            elif i <= 39:
                self.__deck_cards.append(ChopsticksCard())
            elif i <= 42:
                self.__deck_cards.append(WasabiCard())
            elif i <= 47:
                self.__deck_cards.append(PuddingCard())
            elif i <= 51:
                self.__deck_cards.append(OneMakiCard())
            elif i <= 56:
                self.__deck_cards.append(TwoMakiCard())
            elif i < 60:
                self.__deck_cards.append(ThreeMakiCard())
   
    def get_game(self):
        
        return self.__game
   
    def get_deck_cards(self):
        
        return self.__deck_cards
    
    def set_deck_cards(self, cards):
        
        self.__deck_cards = cards  
                    
    # Deals cards when called
    def deal_hand(self):
        
        game = self.get_game()
        cards = self.get_deck_cards()
        cards_by_round = game.get_cards_by_round()
        
        np.random.shuffle(cards)
            
        hand_dealt = cards[0:cards_by_round]
        self.set_deck_cards(cards[cards_by_round:])
        
        return hand_dealt
            
    def add_dessert(self):
        
        game = self.get_game()  
        
        if game.is_party_version():
                      
            dessert_cards = []
        
            if game.get_round() == 2:
                for i in range(0,3):
                    dessert_cards.append(PuddingCard())
                    
            if game.get_round() == 3:
                for i in range(0,2):
                    dessert_cards.append(PuddingCard())
        
            self.take_back_cards(dessert_cards)
                    
    def take_back_cards(self, cards):
        
        game = self.get_game()
         
        if game.is_party_version():
            for card in cards:
                self.__deck_cards.append(card)
                
   
        
   
        
        