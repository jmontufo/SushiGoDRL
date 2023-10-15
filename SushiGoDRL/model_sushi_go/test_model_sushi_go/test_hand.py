# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:40:01 2023

@author: jmont
"""

import pytest

from model_sushi_go.hand import Hand
from model_sushi_go.card import MakiCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import SashimiCard

def test_init():    
    handWithoutCards = Hand()
    assert handWithoutCards.get_cards() == []

    initialCards = [OneMakiCard(), 
                    TwoMakiCard()]    
    handWithInitialCards = Hand(initialCards)    
    assert handWithInitialCards.get_cards() == initialCards

def test_get_cards():    
    initialCards = [OneMakiCard(), 
                    TwoMakiCard(), 
                    ThreeMakiCard()]    
    handWithInitialCards = Hand(initialCards)    
    assert handWithInitialCards.get_cards() == initialCards
    
def test_get_num_of_cards():
    handWithoutCards = Hand()
    assert handWithoutCards.get_num_of_cards() == 0
    
    initialCards = [TwoMakiCard(), 
                    TwoMakiCard(), 
                    ThreeMakiCard()]    
    handWithInitialCards = Hand(initialCards)    
    assert handWithInitialCards.get_num_of_cards() == 3
    
def test_add_card():
     handWithoutInitialCards = Hand()
     firstCard = TwoMakiCard()
     secondCard = OneMakiCard()
     handWithoutInitialCards.add_card(firstCard)     
     assert handWithoutInitialCards.get_cards() == [firstCard]
     
     handWithoutInitialCards.add_card(secondCard)     
     assert handWithoutInitialCards.get_cards() == [firstCard, 
                                                    secondCard]
     
     firstCard = TwoMakiCard()
     secondCard = TwoMakiCard()
     thirdCard = ThreeMakiCard()
     fourthCard = OneMakiCard()
     initialCards = [firstCard, 
                     secondCard, 
                     thirdCard]    
     handWithInitialCards = Hand(initialCards)  
     
     handWithInitialCards.add_card(fourthCard)    
     assert handWithInitialCards.get_cards() == [firstCard, 
                                                 secondCard, 
                                                 thirdCard,
                                                 fourthCard]    
    
def test_remove_card():
     handWithoutInitialCards = Hand()
     firstCard = OneMakiCard()
     secondCard = OneMakiCard() 
     
     # If the card instance is not in the hand, raise error
     with pytest.raises(ValueError):
         handWithoutInitialCards.remove_card(firstCard)    
     
     # Although the card is the same type, the instance is not the same. 
     # Therefore, it raises an error. The instance in the hand is not removed.
     handWithoutInitialCards.add_card(firstCard)          
     with pytest.raises(ValueError):
         handWithoutInitialCards.remove_card(secondCard)   
     assert handWithoutInitialCards.get_cards() == [firstCard]
     
     handWithoutInitialCards.remove_card(firstCard)    
     assert handWithoutInitialCards.get_cards() == []
     
     firstCard = TwoMakiCard()
     secondCard = OneMakiCard()
     thirdCard = ThreeMakiCard()
     fourthCard = OneMakiCard()
     initialCards = [firstCard, 
                     secondCard, 
                     thirdCard]    
     handWithInitialCards = Hand(initialCards)  
     
     handWithInitialCards.remove_card(firstCard)     
     assert handWithInitialCards.get_cards() == [secondCard, 
                                                 thirdCard] 
     handWithInitialCards.add_card(fourthCard)         
     handWithInitialCards.remove_card(secondCard) 
     assert handWithInitialCards.get_cards() == [thirdCard, 
                                                 fourthCard]   

     with pytest.raises(ValueError):
         handWithInitialCards.remove_card(secondCard)        
 
def test_choose_card():      
    handWithoutInitialCards = Hand()
    
    assert handWithoutInitialCards.choose_card(MakiCard) == None
    
    firstCard = TwoMakiCard()
    secondCard = OneMakiCard()
    thirdCard = ThreeMakiCard()
    fourthCard = ThreeMakiCard()
    fifthCard = TempuraCard()
    initialCards = [firstCard, 
                    secondCard, 
                    thirdCard,
                    fourthCard,
                    fifthCard]    
    handWithInitialCards = Hand(initialCards) 
    
    # Choose always the card of the type indicated with the higher value
    assert handWithInitialCards.choose_card(MakiCard) == thirdCard
    assert handWithInitialCards.choose_card(OneMakiCard) == secondCard
    assert handWithInitialCards.choose_card(TwoMakiCard) == firstCard
    assert handWithInitialCards.choose_card(ThreeMakiCard) == thirdCard
    assert handWithInitialCards.choose_card(TempuraCard) == fifthCard
    assert handWithInitialCards.choose_card(SashimiCard) == None
  
def test_to_string():
    handWithoutInitialCards = Hand()      
    assert str(handWithoutInitialCards) == """Hand:
Empty hand
"""
    
    firstCard = TwoMakiCard()
    secondCard = OneMakiCard()
    thirdCard = ThreeMakiCard()
    initialCards = [firstCard, 
                    secondCard, 
                    thirdCard]    
    handWithInitialCards = Hand(initialCards)  
    
    assert str(handWithInitialCards) == """Hand:
Two Maki Rolls - 3
One Maki Roll - 3
Three Maki Rolls - 3
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    