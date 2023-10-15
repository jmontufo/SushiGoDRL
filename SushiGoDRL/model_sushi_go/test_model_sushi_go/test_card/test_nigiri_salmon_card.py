# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import NigiriCard
from model_sushi_go.card import SalmonNigiriCard
from model_sushi_go.card import WasabiCard

def test_score():
    
    salmonNigiriCard = SalmonNigiriCard()
    assert salmonNigiriCard.score() == 2
      
    # When the sashimi has wasabi attached, score is multiplied by 3  
    wasabiCard = WasabiCard()
    salmonNigiriCard.set_attached_wasabi(wasabiCard)
    
    assert salmonNigiriCard.score() == 6
    
def test_get_type_for_action():
    assert SalmonNigiriCard().get_type_for_action() == NigiriCard    

def test_get_number():
    assert SalmonNigiriCard().get_number() == 1       

def test_get_name():
    assert SalmonNigiriCard().get_name() == "Salmon Nigiri"
    
def test_return_to_deck():  
    wasabiCard = WasabiCard()
    salmonNigiriCard = SalmonNigiriCard()
    
    # When the nigiri is returned to the deck, unattach the wasabi
    salmonNigiriCard.set_attached_wasabi(wasabiCard)    
    salmonNigiriCard.return_to_deck()    
    assert salmonNigiriCard.get_attached_wasabi() == None

def test_get_max_value_for_type_of_card():
    assert SalmonNigiriCard.get_max_value_for_type_of_card() == 3 
    
def test_get_value():
    assert SalmonNigiriCard().get_value() == 2  

def test_to_string():
    assert str(SalmonNigiriCard()) == "Salmon Nigiri - 1"  

def test_get_marginal_score():
    assert SalmonNigiriCard().get_marginal_score() == 2
    
def test_has_attached_wasabi():
    
    salmonNigiriCard = SalmonNigiriCard()
    assert salmonNigiriCard.has_attached_wasabi() == False
        
    wasabiCard = WasabiCard()
    salmonNigiriCard.set_attached_wasabi(wasabiCard)    
    assert salmonNigiriCard.has_attached_wasabi() == True
    
def test_get_attached_wasabi():
    
    salmonNigiriCard = SalmonNigiriCard()
    assert salmonNigiriCard.get_attached_wasabi() == None
        
    wasabiCard = WasabiCard()
    salmonNigiriCard.set_attached_wasabi(wasabiCard)
    
    assert salmonNigiriCard.get_attached_wasabi() == wasabiCard
   
def test_set_attached_wasabi():
     
     salmonNigiriCard = SalmonNigiriCard()
     assert salmonNigiriCard.get_attached_wasabi() == None
     assert salmonNigiriCard.has_attached_wasabi() == False
         
     wasabiCard = WasabiCard()
     salmonNigiriCard.set_attached_wasabi(wasabiCard)
     
     assert salmonNigiriCard.get_attached_wasabi() == wasabiCard
     assert salmonNigiriCard.has_attached_wasabi() == True
               
     
  
