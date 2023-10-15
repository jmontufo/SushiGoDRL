# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import PuddingCard

def test_score():
    
    puddingCard = PuddingCard()
    assert puddingCard.score() == 0
    
def test_get_type_for_action():
    assert PuddingCard().get_type_for_action() == PuddingCard    

def test_get_number():
    assert PuddingCard().get_number() == 7       

def test_get_name():
    assert PuddingCard().get_name() == "Pudding"
    
def test_get_max_value_for_type_of_card():
    assert PuddingCard.get_max_value_for_type_of_card() == 0 
    
def test_get_value():
    assert PuddingCard().get_value() == 0  

def test_to_string():
    assert str(PuddingCard()) == "Pudding - 7"

