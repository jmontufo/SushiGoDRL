# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import TempuraCard

def test_score():
    
    # Only the card that completes a set scores
    tempuraCard = TempuraCard()
    assert tempuraCard.score(1) == 0
    assert tempuraCard.score(2) == 5
    assert tempuraCard.score(3) == 0
    assert tempuraCard.score(4) == 5
    assert tempuraCard.score(13) == 0
    assert tempuraCard.score(14) == 5
    
def test_get_type_for_action():
    assert TempuraCard().get_type_for_action() == TempuraCard    

def test_get_number():
    assert TempuraCard().get_number() == 5       

def test_get_name():
    assert TempuraCard().get_name() == "Tempura"
    
def test_get_max_value_for_type_of_card():
    assert TempuraCard.get_max_value_for_type_of_card() == 0 
    
def test_get_value():
    assert TempuraCard().get_value() == 0  

def test_to_string():
    assert str(TempuraCard()) == "Tempura - 5"  

def test_get_complete_set():
    assert TempuraCard().get_complete_set() == 2

def test_get_complete_set_bonus():
    assert TempuraCard().get_complete_set_bonus() == 5
     
  
