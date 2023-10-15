# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:12:56 2023

@author: jmont
"""

from model_sushi_go.card import DumplingCard

def test_score():
    
    dumplingCard = DumplingCard()
    assert dumplingCard.score(1) == 1
    assert dumplingCard.score(2) == 2
    assert dumplingCard.score(3) == 3
    assert dumplingCard.score(4) == 4
    assert dumplingCard.score(5) == 5
    assert dumplingCard.score(6) == 0
    assert dumplingCard.score(13) == 0
    assert dumplingCard.score(14) == 0
    assert dumplingCard.score(15) == 0
    
def test_get_type_for_action():
    assert DumplingCard().get_type_for_action() == DumplingCard    

def test_get_number():
    assert DumplingCard().get_number() == 6       

def test_get_name():
    assert DumplingCard().get_name() == "Dumpling"
    
def test_get_max_value_for_type_of_card():
    assert DumplingCard.get_max_value_for_type_of_card() == 0 
    
def test_get_value():
    assert DumplingCard().get_value() == 0  

def test_to_string():
    assert str(DumplingCard()) == "Dumpling - 6"

