# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import NigiriCard
from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import WasabiCard

def test_score():
    
    eggNigiriCard = EggNigiriCard()
    assert eggNigiriCard.score() == 1
   
    # When the sashimi has wasabi attached, score is multiplied by 3     
    wasabiCard = WasabiCard()
    eggNigiriCard.set_attached_wasabi(wasabiCard)
    assert eggNigiriCard.score() == 3
    
def test_get_type_for_action():
    assert EggNigiriCard().get_type_for_action() == NigiriCard    

def test_get_number():
    assert EggNigiriCard().get_number() == 1       

def test_get_name():
    assert EggNigiriCard().get_name() == "Egg Nigiri"
    
def test_return_to_deck():  
    wasabiCard = WasabiCard()
    eggNigiriCard = EggNigiriCard()
    
    # When the nigiri is returned to the deck, unattach the wasabi
    eggNigiriCard.set_attached_wasabi(wasabiCard)    
    eggNigiriCard.return_to_deck()    
    assert eggNigiriCard.get_attached_wasabi() == None

def test_get_max_value_for_type_of_card():
    assert EggNigiriCard.get_max_value_for_type_of_card() == 3 
    
def test_get_value():
    assert EggNigiriCard().get_value() == 1  

def test_to_string():
    assert str(EggNigiriCard()) == "Egg Nigiri - 1"  

def test_get_marginal_score():
    assert EggNigiriCard().get_marginal_score() == 1
    
def test_has_attached_wasabi():
    
    eggNigiriCard = EggNigiriCard()
    assert eggNigiriCard.has_attached_wasabi() == False
        
    wasabiCard = WasabiCard()
    eggNigiriCard.set_attached_wasabi(wasabiCard)    
    assert eggNigiriCard.has_attached_wasabi() == True
    
def test_get_attached_wasabi():
    
    eggNigiriCard = EggNigiriCard()
    assert eggNigiriCard.get_attached_wasabi() == None
        
    wasabiCard = WasabiCard()
    eggNigiriCard.set_attached_wasabi(wasabiCard)
    
    assert eggNigiriCard.get_attached_wasabi() == wasabiCard
   
def test_set_attached_wasabi():
     
     eggNigiriCard = EggNigiriCard()
     assert eggNigiriCard.get_attached_wasabi() == None
     assert eggNigiriCard.has_attached_wasabi() == False
         
     wasabiCard = WasabiCard()
     eggNigiriCard.set_attached_wasabi(wasabiCard)
     
     assert eggNigiriCard.get_attached_wasabi() == wasabiCard
     assert eggNigiriCard.has_attached_wasabi() == True
               
     
  
