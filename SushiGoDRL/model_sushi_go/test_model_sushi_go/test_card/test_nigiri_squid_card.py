# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import NigiriCard
from model_sushi_go.card import SquidNigiriCard
from model_sushi_go.card import WasabiCard

def test_score():
    
    squidNigiriCard = SquidNigiriCard()
    assert squidNigiriCard.score() == 3
        
    # When the sashimi has wasabi attached, score is multiplied by 3
    wasabiCard = WasabiCard()
    squidNigiriCard.set_attached_wasabi(wasabiCard)
    
    assert squidNigiriCard.score() == 9
    
def test_get_type_for_action():
    assert SquidNigiriCard().get_type_for_action() == NigiriCard    

def test_get_number():
    assert SquidNigiriCard().get_number() == 1       

def test_get_name():
    assert SquidNigiriCard().get_name() == "Squid Nigiri"
    
def test_return_to_deck():  
    wasabiCard = WasabiCard()
    squidNigiriCard = SquidNigiriCard()
    
    # When the nigiri is returned to the deck, unattach the wasabi
    squidNigiriCard.set_attached_wasabi(wasabiCard)
    squidNigiriCard.return_to_deck()    
    assert squidNigiriCard.get_attached_wasabi() == None

def test_get_max_value_for_type_of_card():
    assert SquidNigiriCard.get_max_value_for_type_of_card() == 3 
    
def test_get_value():
    assert SquidNigiriCard().get_value() == 3  

def test_to_string():
    assert str(SquidNigiriCard()) == "Squid Nigiri - 1"  

def test_get_marginal_score():
    assert SquidNigiriCard().get_marginal_score() == 3
    
def test_has_attached_wasabi():
    
    squidNigiriCard = SquidNigiriCard()
    assert squidNigiriCard.has_attached_wasabi() == False
        
    wasabiCard = WasabiCard()
    squidNigiriCard.set_attached_wasabi(wasabiCard)    
    assert squidNigiriCard.has_attached_wasabi() == True
    
def test_get_attached_wasabi():
    
    squidNigiriCard = SquidNigiriCard()
    assert squidNigiriCard.get_attached_wasabi() == None
        
    wasabiCard = WasabiCard()
    squidNigiriCard.set_attached_wasabi(wasabiCard)
    
    assert squidNigiriCard.get_attached_wasabi() == wasabiCard
   
def test_set_attached_wasabi():
     
     squidNigiriCard = SquidNigiriCard()
     assert squidNigiriCard.get_attached_wasabi() == None
     assert squidNigiriCard.has_attached_wasabi() == False
         
     wasabiCard = WasabiCard()
     squidNigiriCard.set_attached_wasabi(wasabiCard)
     
     assert squidNigiriCard.get_attached_wasabi() == wasabiCard
     assert squidNigiriCard.has_attached_wasabi() == True
               
     
  
