# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import MakiCard
from model_sushi_go.card import ThreeMakiCard

def test_score():
    
    threeMakiCard = ThreeMakiCard()
    assert threeMakiCard.score() == 0
    
def test_get_type_for_action():
    assert ThreeMakiCard().get_type_for_action() == MakiCard    

def test_get_number():
    assert ThreeMakiCard().get_number() == 3      

def test_get_name():
    assert ThreeMakiCard().get_name() == "Three Maki Rolls"

def test_get_max_value_for_type_of_card():
    assert ThreeMakiCard.get_max_value_for_type_of_card() == 3 
    
def test_get_value():
    assert ThreeMakiCard().get_value() == 3  

def test_to_string():
    assert str(ThreeMakiCard()) == "Three Maki Rolls - 3"  





