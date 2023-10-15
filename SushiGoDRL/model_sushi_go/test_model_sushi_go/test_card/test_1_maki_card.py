# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import MakiCard
from model_sushi_go.card import OneMakiCard

def test_score():
    
    oneMakiCard = OneMakiCard()
    assert oneMakiCard.score() == 0
    
def test_get_type_for_action():
    assert OneMakiCard().get_type_for_action() == MakiCard    

def test_get_number():
    assert OneMakiCard().get_number() == 3      

def test_get_name():
    assert OneMakiCard().get_name() == "One Maki Roll"

def test_get_max_value_for_type_of_card():
    assert OneMakiCard.get_max_value_for_type_of_card() == 3 
    
def test_get_value():
    assert OneMakiCard().get_value() == 1  

def test_to_string():
    assert str(OneMakiCard()) == "One Maki Roll - 3"  





