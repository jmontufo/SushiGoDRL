# -*- coding: utf-8 -*-


class Hand(object):

    def __init__(self, initial_cards = None):
    
        if initial_cards is None:
            self.__cards = []
        else:
            self.__cards = initial_cards
         
    def get_cards(self):
        
        return self.__cards
    
    def get_num_of_cards(self):
        
        return len(self.__cards)  
    
    def add_card(self, card):
        
        self.__cards.append(card)
        
    def remove_card(self, card):
                
        self.__cards.remove(card)        
    
    def choose_card(self, card_type):

        card_with_max_value = None
        max_value_found = -1
        max_value_possible = card_type.get_max_value_for_type_of_card()
        
        cards = self.get_cards()
        
        for card in cards:         
            
            if isinstance(card, card_type):                
                card_value = card.get_value()                
                if card_value > max_value_found:
                    max_value_found = card_value
                    card_with_max_value = card
                    if max_value_found is max_value_possible:
                        break
                
        return card_with_max_value
        
    def __str__(self):

        to_string = "Hand:\n"
        
        cards = self.get_cards()        
            
        for card in cards:
            to_string += str(card) + "\n"
            
        if len(cards) == 0:
            to_string += "Empty hand\n"
                            
        return to_string
        