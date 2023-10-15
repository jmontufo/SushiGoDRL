# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import WasabiCard

def test_score():    
    assert WasabiCard().score() == 0        
    
def test_get_type_for_action():
    assert WasabiCard().get_type_for_action() == WasabiCard    

def test_get_number():
    assert WasabiCard().get_number() == 2       

def test_get_name():
    assert WasabiCard().get_name() == "Wasabi"
    
def test_return_to_deck():  
    wasabiCard = WasabiCard()    
    eggNigiriCard = EggNigiriCard()
    
    wasabiCard.set_attached_nigiri(eggNigiriCard)
    assert wasabiCard.get_attached_nigiri() == eggNigiriCard
    
    wasabiCard.return_to_deck()    
    assert wasabiCard.get_attached_nigiri() == None

def test_get_max_value_for_type_of_card():
    assert WasabiCard.get_max_value_for_type_of_card() == 0 
    
def test_get_value():
    assert WasabiCard().get_value() == 0  

def test_to_string():
    assert str(WasabiCard()) == "Wasabi - 2"  
    
def test_has_attached_nigiri():
    wasabiCard = WasabiCard()    
    assert wasabiCard.has_attached_nigiri() == False
        
    eggNigiriCard = EggNigiriCard()
    wasabiCard.set_attached_nigiri(eggNigiriCard)
    
    assert wasabiCard.has_attached_nigiri() == True
    
def test_get_and_set_attached_nigiri():    
    wasabiCard = WasabiCard()    
    assert wasabiCard.get_attached_nigiri() == None
        
    eggNigiriCard = EggNigiriCard()
    wasabiCard.set_attached_nigiri(eggNigiriCard)
    
    assert wasabiCard.get_attached_nigiri() == eggNigiriCard