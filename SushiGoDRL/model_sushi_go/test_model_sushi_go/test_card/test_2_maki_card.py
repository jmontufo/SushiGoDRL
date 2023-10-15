# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import MakiCard
from model_sushi_go.card import TwoMakiCard

def test_score():
    
    twoMakiCard = TwoMakiCard()
    assert twoMakiCard.score() == 0
    
def test_get_type_for_action():
    assert TwoMakiCard().get_type_for_action() == MakiCard    

def test_get_number():
    assert TwoMakiCard().get_number() == 3      

def test_get_name():
    assert TwoMakiCard().get_name() == "Two Maki Rolls"

def test_get_max_value_for_type_of_card():
    assert TwoMakiCard.get_max_value_for_type_of_card() == 3 
    
def test_get_value():
    assert TwoMakiCard().get_value() == 2  

def test_to_string():
    assert str(TwoMakiCard()) == "Two Maki Rolls - 3"  





