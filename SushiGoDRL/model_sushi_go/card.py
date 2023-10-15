# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class Card(ABC):
    
    @abstractmethod
    def score(self, order):
        # order(int): order of the type of cards on the set being evalutated.
        # It's useful when the number of cards of the same type determines the
        # score. For example, the first tempura doesn't scores, but the 2nd 
        # scores 5
        
        pass
   
    @abstractmethod
    def get_type_for_action(self = None):
        
        pass    
    
    @abstractmethod
    def get_number(self = None):
        
        pass        
    
    @abstractmethod
    def get_name(self = None):
        
        pass       
    
    def return_to_deck(self):
        
        return    
    
    def get_max_value_for_type_of_card():
        
        return 0
    
    def get_value(self = None):
        
        return 0
        
    def __str__(self):
        
        return self.get_name() + " - " + str(self.get_number())

class NigiriCard(Card):
    
    __number = 1    
    __name = "Nigiri"
    __max_value_for_type_of_card = 3
    __attached_wasabi = None
    
    def get_type_for_action(self = None):
        
        return NigiriCard
        
    def get_number(self = None):
        
        return NigiriCard.__number
    
    def get_name(self = None):
        
        return NigiriCard.__name
    
    def get_max_value_for_type_of_card():
        
        return NigiriCard.__max_value_for_type_of_card
            
    def score(self, order = 0):
        
        score = self.get_marginal_score()
        
        if self.has_attached_wasabi():
            score *= 3
            
        return score 
        
    @abstractmethod
    def get_marginal_score(self = None):
        
        pass    
    
    def has_attached_wasabi(self):
        
        return self.__attached_wasabi is not None
           
    def get_attached_wasabi(self):
        
        return self.__attached_wasabi
    
    def set_attached_wasabi(self, wasabi_card):
        
        self.__attached_wasabi = wasabi_card
        
    def return_to_deck(self):
        
        self.set_attached_wasabi(None)   
    
class EggNigiriCard(NigiriCard):
    
    __name = "Egg Nigiri"
    __marginal_score = 1
    __value = 1
        
    def get_name(self = None):
        
        return EggNigiriCard.__name
    
    def get_marginal_score(self = None):
        
        return EggNigiriCard.__marginal_score
            
    def get_value(self = None):
        
        return EggNigiriCard.__value
        
class SalmonNigiriCard(NigiriCard):
    
    __name = "Salmon Nigiri"
    __marginal_score = 2
    __value = 2
       
    def get_name(self = None):
        
        return SalmonNigiriCard.__name
    
    def get_marginal_score(self = None):
        
        return SalmonNigiriCard.__marginal_score
            
    def get_value(self = None):
        
        return SalmonNigiriCard.__value  
    
class SquidNigiriCard(NigiriCard):
    
    __name = "Squid Nigiri"
    __marginal_score = 3
    __value = 3
    
    def get_name(self = None):
        
        return SquidNigiriCard.__name
    
    def get_marginal_score(self = None):
        
        return SquidNigiriCard.__marginal_score
    
    def get_value(self = None):
        
        return SquidNigiriCard.__value       

# Super class for Tempura an Sashimi, cards that share same way of scoring  
class SetCard(Card):
    
    __name = "Set Card (Tempura or Sashimi)"
        
    def get_name(self = None):
        
        return SetCard.__name    
    
    def score(self, order):
        if order % self.get_complete_set() == 0:
            return self.get_complete_set_bonus()
        else:
            return 0
    
    @abstractmethod
    def get_complete_set(self = None):
        
        pass    
    
    @abstractmethod
    def get_complete_set_bonus(self = None):
        
        pass    

class SashimiCard(SetCard):
        
    __number = 4  
    __name = "Sashimi"   
    __complete_set = 3   
    __set_bonus = 10 
            
    def get_number(self = None):
        
        return SashimiCard.__number
    
    def get_name(self = None):
        
        return SashimiCard.__name
    
    def get_type_for_action(self = None):
        
        return SashimiCard

    def get_complete_set(self = None):
        
        return SashimiCard.__complete_set

    def get_complete_set_bonus(self = None):
        
        return SashimiCard.__set_bonus    
      
class TempuraCard(SetCard):
    
    __number = 5
    __name = "Tempura"
    __complete_set = 2   
    __set_bonus = 5
        
    def get_number(self = None):
        
        return TempuraCard.__number
    
    def get_name(self = None):
        
        return TempuraCard.__name
        
    def get_type_for_action(self = None):
        
        return TempuraCard

    def get_complete_set(self = None):
        
        return TempuraCard.__complete_set

    def get_complete_set_bonus(self = None):
        
        return TempuraCard.__set_bonus    

class DumplingCard(Card):   
    
    __number = 6
    __name = "Dumpling"
    __marginal_set_bonus = [0,1,2,3,4,5]  
        
    def get_number(self = None):
        
        return DumplingCard.__number
    
    def get_name(self = None):
        
        return DumplingCard.__name
        
    def get_type_for_action(self = None):
        
        return DumplingCard
    
    def score(self, order):
        
        marginal_set_bonus = DumplingCard.__marginal_set_bonus
        
        if order < len(marginal_set_bonus):
            return marginal_set_bonus[order]
        else:
            return 0

class ChopsticksCard(Card):   
    
    __number = 0
    __name = "Chopsticks"  
        
    def get_number(self = None):
        
        return ChopsticksCard.__number
    
    def get_name(self = None):
        
        return ChopsticksCard.__name
        
    def get_type_for_action(self = None):
        
        return ChopsticksCard
            
    def score(self, order = 0):
        
        return 0
        
class WasabiCard(Card):    
    
    __number = 2
    __name = "Wasabi"    
    __attached_nigiri = None
        
    def get_number(self = None):
        
        return WasabiCard.__number
    
    def get_name(self = None):
        
        return WasabiCard.__name
        
    def get_type_for_action(self = None):
        
        return WasabiCard
    
    def score(self, order = 0):
        
        return 0 
    
    def has_attached_nigiri(self):
        
        return self.__attached_nigiri != None
    
    def get_attached_nigiri(self):
        
        return self.__attached_nigiri
    
    def set_attached_nigiri(self, nigiri_card):
        
        self.__attached_nigiri = nigiri_card
        
    def return_to_deck(self):
        
        self.set_attached_nigiri(None)   
        
class PuddingCard(Card):    
    
    __number = 7
    __name = "Pudding"  
        
    def get_number(self = None):
        
        return PuddingCard.__number
    
    def get_name(self = None):
        
        return PuddingCard.__name
    
    def get_type_for_action(self = None):
        
        return PuddingCard
        
    def score(self, order = 0):
        
        return 0

class MakiCard(Card):    
    
    __number = 3    
    __name = "Maki Roll"
    
    def get_number(self = None):
        
        return MakiCard.__number
    
    def get_name(self = None):
        
        return MakiCard.__name
    
    def get_type_for_action(self = None):
        
        return MakiCard    
        
    def get_max_value_for_type_of_card():
        
        return 3
    
    def score(self, order = 0):
        
        return 0
    
class OneMakiCard(MakiCard): 
    
    __name = "One Maki Roll"
    __value = 1
    
    def get_name(self = None):
        
        return OneMakiCard.__name
    
    def get_value(self = None):
        
        return OneMakiCard.__value  
    
class TwoMakiCard(MakiCard):  
    
    __name = "Two Maki Rolls"
    __value = 2
    
    def get_name(self = None):
        
        return TwoMakiCard.__name
            
    def get_value(self = None):
        
        return TwoMakiCard.__value  
    
class ThreeMakiCard(MakiCard): 
    
    __name = "Three Maki Rolls"
    __value = 3
        
    def get_name(self = None):
        
        return ThreeMakiCard.__name
        
    def get_value(self = None):
        
        return ThreeMakiCard.__value
 
class CARD_TYPES(ABC):

    __NUM_TYPES_OF_CARD = 8
    
    __card_types_by_number= {
        0: ChopsticksCard,
        1: NigiriCard,
        2: WasabiCard,
        3: MakiCard,
        4: SashimiCard,
        5: TempuraCard,
        6: DumplingCard,
        7: PuddingCard }
    
    __card_types = [
        ChopsticksCard,
        EggNigiriCard,
        SalmonNigiriCard,
        SquidNigiriCard,
        WasabiCard,
        OneMakiCard,
        TwoMakiCard,
        ThreeMakiCard,
        SashimiCard,
        TempuraCard,
        DumplingCard,
        PuddingCard ]
    
    def get_num_types_of_card():
    
        return CARD_TYPES.__NUM_TYPES_OF_CARD    
    
    def get_card_types_by_number():
    
        return CARD_TYPES.__card_types_by_number
    
    def get_card_types():
    
        return CARD_TYPES.__card_types
           
    
    
    
    
    
    
    
    
    
    