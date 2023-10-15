# -*- coding: utf-8 -*-

from model_sushi_go.card import WasabiCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import DumplingCard
from model_sushi_go.card import PuddingCard


class ActionManager(object):
    
    def get_legal_actions(cards, is_chopsticks_move_available):
                      
        legal_actions = set()
        
        for card in cards:
            
            action = Action(card.get_type_for_action())            
            legal_actions.add(action)
            
        if is_chopsticks_move_available:
            
            chopsticks_legal_actions = ActionManager.__get_chopsticks_legal_actions(cards)
            legal_actions.update(chopsticks_legal_actions)
                        
        return list(legal_actions)
    
    def __get_chopsticks_legal_actions(cards):
        
        legal_actions = set()
        
        for card in cards:
            for other_card in cards:
                    
                is_not_the_same_card = card is not other_card
                # to avoid repeated actions, consider only ordered types
                cards_ordered = card.get_number() <= other_card.get_number()                    
                is_not_chopstick = not isinstance(card, ChopsticksCard)
                card_is_wasabi = isinstance(card, WasabiCard)
                other_card_is_nigiri = isinstance(other_card, NigiriCard)
                
                if is_not_the_same_card and cards_ordered and is_not_chopstick:
                    
                    card_type = card.get_type_for_action()
                    other_card_type = other_card.get_type_for_action()
                    
                    action = ChopsticksAction(card_type, other_card_type)
                    
                    legal_actions.add(action)
                    
                elif card_is_wasabi and other_card_is_nigiri:
                    
                    action = WasabiBeforeNigiriAction()
                    legal_actions.add(action)
        
        return legal_actions
    
    def action_from_number(number):
        
        if number == 36:
            
            return WasabiBeforeNigiriAction()
        
        elif number > 7:
            
            number_to_pair = [[NigiriCard,NigiriCard], # 8 : Nigiri, Nigiri
                              [NigiriCard,WasabiCard], # 9 : Nigiri, Wasabi
                              [NigiriCard,MakiCard], # 10 : Nigiri, Maki
                              [NigiriCard,SashimiCard], # 11 : Nigiri, Sashimi
                              [NigiriCard,TempuraCard], # 12 : Nigiri, Tempura
                              [NigiriCard,DumplingCard], # 13 : Nigiri, Gyoza
                              [NigiriCard,PuddingCard], # 14 : Nigiri, Pudding
                              [WasabiCard,WasabiCard], # 15 : Wasabi, Wasabi
                              [WasabiCard,MakiCard], # 16 : Wasabi, Maki
                              [WasabiCard,SashimiCard], # 17 : Wasabi, Sashimi
                              [WasabiCard,TempuraCard], # 18 : Wasabi, Tempura
                              [WasabiCard,DumplingCard], # 19 : Wasabi, Gyoza
                              [WasabiCard,PuddingCard], # 20 : Wasabi, Pudding
                              [MakiCard,MakiCard], # 21 : Maki, Maki
                              [MakiCard,SashimiCard], # 22 : Maki, Sashimi
                              [MakiCard,TempuraCard], # 23 : Maki, Tempura
                              [MakiCard,DumplingCard], # 24 : Maki, Gyoza
                              [MakiCard,PuddingCard], # 25 : Maki, Pudding
                              [SashimiCard,SashimiCard], # 26 : Sashimi, Sashimi
                              [SashimiCard,TempuraCard], # 27 : Sashimi, Tempura
                              [SashimiCard,DumplingCard], # 28 : Sashimi, Gyoza
                              [SashimiCard,PuddingCard], # 29 : Sashimi, Pudding
                              [TempuraCard,TempuraCard], # 30 : Tempura, Tempura
                              [TempuraCard,DumplingCard], # 31 : Tempura, Gyoza
                              [TempuraCard,PuddingCard], # 32 : Tempura, Pudding
                              [DumplingCard,DumplingCard], # 33 : Gyoza, Gyoza
                              [DumplingCard,PuddingCard], # 34 : Gyoza, Pudding
                              [PuddingCard,PuddingCard]] # 35 : Pudding, Pudding
            
            pair = number_to_pair[number - 8]
            
            return ChopsticksAction(pair[0], pair[1])
        
        else:
            
            number_to_type = [ChopsticksCard, 
                              NigiriCard, 
                              WasabiCard,
                              MakiCard, 
                              SashimiCard, 
                              TempuraCard, 
                              DumplingCard, 
                              PuddingCard] 
            
            return Action(number_to_type[number])
 
    def numbers_from_legal_actions(legal_actions):
        
        array_of_numbers = []
        
        for action in legal_actions:
            array_of_numbers.append(action.get_action_number())
            
        return array_of_numbers    
            

class Action(object):

    def __init__(self, card_type):
        
        self.__first_card = card_type        
        self.__second_card = None
        
    def get_first_card(self):
        
        return self.__first_card
    
    def set_first_card(self, first_card):
        
        self.__first_card = first_card
    
    def get_second_card(self):
        
        return self.__second_card
    
    def set_second_card(self, second_card):
        
        self.__second_card = second_card
        
    def get_pair_of_cards(self):
        
        first_card_type = self.get_first_card()
        second_card_type = self.get_second_card()
        
        cards = []
        cards.append(first_card_type)
        cards.append(second_card_type)
        
        return cards
    
    def is_simple_action_of_type(self, card_type):
        
        return self.get_first_card() == card_type and self.get_second_card() is None
    
    def is_any_action_of_type(self, card_type):
        
        return self.get_first_card() == card_type or self.get_second_card() == card_type
    
    def is_chopsticks_action_with_type(self, card_type):
        
        return self.is_any_action_with_type(card_type) and self.get_second_card() is not None
    
    def is_chopsticks_action_with_type_twice(self, card_type):
        
         return self.get_first_card() == card_type and self.get_second_card() == card_type
        
    def __hash__(self):   
        
        return hash((self.get_first_card(), self.get_second_card()))

    def __eq__(self, other):
        
        if not isinstance(other, type(self)): 
            return NotImplemented
        
        this_first_card = self.get_first_card()
        other_first_card = other.get_first_card()
        
        first_cards_equal = this_first_card == other_first_card
                
        this_second_card = self.get_second_card()
        other_second_card = other.get_second_card()
        
        same_second_cards = this_second_card == other_second_card
        second_cards_are_none = this_second_card is None and other_second_card is None
        second_cards_equal = same_second_cards or second_cards_are_none       
        
        return first_cards_equal and second_cards_equal
   
    def get_action_number(self):
        
        first_card_type = self.get_first_card()
        
        return first_card_type.get_number()
    
    def __str__(self):
                
        first_card_type = self.get_first_card()
        second_card_type = self.get_second_card()
        
        to_string = str(self.get_action_number()) 
        to_string += " - " + first_card_type.get_name()
        
        if second_card_type is not None:
             to_string += " - " + second_card_type .get_name()
             
        return to_string
    
class ChopsticksAction(Action):
    
    def __init__(self, first_card_type, second_card_type):
      
        assert first_card_type.get_number() <= second_card_type.get_number()
        assert first_card_type.get_number() != 0
               
        self.set_first_card(first_card_type)
        self.set_second_card(second_card_type)
                
    def get_action_number(self):
                
        first_card_type = self.get_first_card()
        second_card_type = self.get_second_card()
        
        second_card_number = second_card_type.get_number()
        
        if first_card_type is NigiriCard:
            return 7 + second_card_number
        elif first_card_type is WasabiCard:            
            return 13 + second_card_number
        elif first_card_type is MakiCard:            
            return 18 + second_card_number
        elif first_card_type is SashimiCard:            
            return 22 + second_card_number
        elif first_card_type is TempuraCard:            
            return 25 + second_card_number
        elif first_card_type is DumplingCard:            
            return 27 + second_card_number
        elif first_card_type is PuddingCard:            
            return 28 + second_card_number            
        
class WasabiBeforeNigiriAction(ChopsticksAction):
    
    def __init__(self):
              
        self.set_first_card(WasabiCard)
        self.set_second_card(NigiriCard)
    
    def get_action_number(self):
        
        return 36