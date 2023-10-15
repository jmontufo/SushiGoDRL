# -*- coding: utf-8 -*-

from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import WasabiCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import PuddingCard

class Table(object):

    def __init__(self, player):

        self.__player = player
        self.__cards = []
        # Number of maki rolls
        self.__maki_rolls = 0
        # Num of Wasabis in play
        self.__wasabis = 0
        # Num of Chopsticks in play
        self.__chopsticks = 0
        # Num of Puddings in play
        self.__puddings = 0
            
    def get_cards(self):
        
        return self.__cards
    
    def __set_cards(self, cards):
        
        self.__cards = cards
    
    def get_maki_rolls(self):
        
        return self.__maki_rolls    
    
    def __increase_maki_rolls(self, value):
        
        self.__maki_rolls += value
    
    def __reset_maki_rolls(self):
        
        self.__maki_rolls = 0
    
    def get_wasabis(self):
        
        return self.__wasabis
    
    def __increase_wasabis(self):
        
        self.__wasabis += 1
    
    def __decrease_wasabis(self):
        
        self.__wasabis -= 1
    
    def __reset_wasabis(self):
        
        self.__wasabis = 0
    
    def get_chopsticks(self):
        
        return self.__chopsticks
    
    def __increase_chopsticks(self):
        
        self.__chopsticks += 1
    
    def __decrease_chopsticks(self):
        
        self.__chopsticks -= 1
    
    def __reset_chopsticks(self):
        
        self.__chopsticks = 0
    
    def get_puddings(self):
        
        return self.__puddings
    
    def __increase_puddings(self):
        
        self.__puddings += 1
        
    def __str__(self):

        cards = self.get_cards()
        
        to_string = "Table:\n"
           
        if len(cards) == 0:            
            to_string += "\tEmpty table\n"
        else:
            for card in cards:
                to_string += str(card) + "\n"                
            
            to_string += "maki_rolls:" + str(self.get_maki_rolls()) + "\n"
            to_string += "wasabi:" + str(self.get_wasabis()) + "\n"
            to_string += "chopstick:" + str(self.get_chopsticks()) + "\n"
                            
        return to_string
    
    def pick_chopsticks_card(self):
                
        cards = self.get_cards()
        
        chopsticks_card = self.find_card_by_type(ChopsticksCard)
        assert chopsticks_card is not None
        
        cards.remove(chopsticks_card)
        self.__decrease_chopsticks()
        
        return chopsticks_card
    
    def return_cards_except_pudding(self):
                
        cards = self.get_cards()
               
        cards_returned = []
        pudding_cards = []
        
        for card in cards:
            if not isinstance(card, PuddingCard):
                cards_returned.append(card)
                card.return_to_deck()                
            else:
                pudding_cards.append(card)
                
        self.__set_cards(pudding_cards)    
        
        self.__reset_maki_rolls()
        self.__reset_wasabis()
        self.__reset_chopsticks()
        
        return cards_returned
       
    
    def find_card_by_type(self, card_type, filter_method = None):
        
        cards = self.get_cards()
        
        for card in cards:
            if isinstance(card, card_type): 
                if filter_method != None: 
                    if filter_method(card):
                        return card
                else:
                    return card
                
        return None
    
    def num_of_cards_of_type(self, card_type):
        
        num_of_cards = 0
        cards = self.get_cards()
        
        for card in cards:
            if isinstance(card, card_type):
                num_of_cards += 1
        
        return num_of_cards
    
    def add_card(self, card):
        
        self.__cards.append(card)
        
        if isinstance(card, WasabiCard):
            self.__increase_wasabis()
        if isinstance(card, ChopsticksCard):
            self.__increase_chopsticks()
        if isinstance(card, MakiCard):
            self.__increase_maki_rolls(card.get_value())
        if isinstance(card, PuddingCard):
            self.__increase_puddings()
            
        if isinstance(card, NigiriCard) and self.get_wasabis() > 0:
            self.__attach_nigiri_to_wasabi(card)
            
    def __attach_nigiri_to_wasabi(self, nigiri_card):
        
        wasabi_card = self.find_card_by_type(WasabiCard, 
                                             self.__wasabi_is_not_attached)
        assert wasabi_card is not None
        
        wasabi_card.set_attached_nigiri(nigiri_card)
        nigiri_card.set_attached_wasabi(wasabi_card)
        
        self.__decrease_wasabis()
        
    def __wasabi_is_not_attached(self, wasabi_card):
         
        return not wasabi_card.has_attached_nigiri()
        
        
        
        
        
        
        
        
        
        
           