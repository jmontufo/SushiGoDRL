# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import SashimiCard

def test_score():
    
    # Only the card that completes a set scores
    sashimiCard = SashimiCard()
    assert sashimiCard.score(1) == 0
    assert sashimiCard.score(2) == 0
    assert sashimiCard.score(3) == 10
    assert sashimiCard.score(4) == 0
    assert sashimiCard.score(5) == 0
    assert sashimiCard.score(6) == 10
    assert sashimiCard.score(13) == 0
    assert sashimiCard.score(14) == 0
    assert sashimiCard.score(15) == 10
    
def test_get_type_for_action():
    assert SashimiCard().get_type_for_action() == SashimiCard    

def test_get_number():
    assert SashimiCard().get_number() == 4       

def test_get_name():
    assert SashimiCard().get_name() == "Sashimi"
    
def test_get_max_value_for_type_of_card():
    assert SashimiCard.get_max_value_for_type_of_card() == 0 
    
def test_get_value():
    assert SashimiCard().get_value() == 0  

def test_to_string():
    assert str(SashimiCard()) == "Sashimi - 4"  

def test_get_complete_set():
    assert SashimiCard().get_complete_set() == 3

def test_get_complete_set_bonus():
    assert SashimiCard().get_complete_set_bonus() == 10
     
  
