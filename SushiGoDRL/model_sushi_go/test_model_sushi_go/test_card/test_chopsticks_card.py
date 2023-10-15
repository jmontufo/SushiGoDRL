# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import ChopsticksCard

def test_score():
    
    chopsticksCard = ChopsticksCard()
    assert chopsticksCard.score() == 0
    
def test_get_type_for_action():
    assert ChopsticksCard().get_type_for_action() == ChopsticksCard    

def test_get_number():
    assert ChopsticksCard().get_number() == 0       

def test_get_name():
    assert ChopsticksCard().get_name() == "Chopsticks"
    
def test_get_max_value_for_type_of_card():
    assert ChopsticksCard.get_max_value_for_type_of_card() == 0 
    
def test_get_value():
    assert ChopsticksCard().get_value() == 0  

def test_to_string():
    assert str(ChopsticksCard()) == "Chopsticks - 0"

