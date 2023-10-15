# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:32:04 2023

@author: jmont
"""

from model_sushi_go.action import ActionManager
from model_sushi_go.action import Action
from model_sushi_go.action import ChopsticksAction
from model_sushi_go.action import WasabiBeforeNigiriAction
from model_sushi_go.card import WasabiCard
from model_sushi_go.card import EggNigiriCard
from model_sushi_go.card import SalmonNigiriCard
from model_sushi_go.card import SquidNigiriCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import OneMakiCard
from model_sushi_go.card import TwoMakiCard
from model_sushi_go.card import ThreeMakiCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import TempuraCard
from model_sushi_go.card import DumplingCard
from model_sushi_go.card import PuddingCard

def same_unordered_list_of_actions(actions, expected_actions):
    for action in actions:
        found = False
        for expected_action in expected_actions:
            if action == expected_action:
                found = True
        if not found:
            return False
        expected_actions.remove(action)
    return expected_actions == []

def test_get_legal_actions():        

    cards = [OneMakiCard(), WasabiCard()]
    chopsticks_available = False
    
    actions = ActionManager.get_legal_actions(cards, chopsticks_available)
    
    expected_actions = [Action(MakiCard), Action(WasabiCard)]
    
    assert same_unordered_list_of_actions(actions, expected_actions)
    
    cards = [OneMakiCard(), TempuraCard()]
    chopsticks_available = True
    
    actions = ActionManager.get_legal_actions(cards, chopsticks_available)
    
    expected_actions = [Action(MakiCard), 
                        Action(TempuraCard), 
                        ChopsticksAction(MakiCard, TempuraCard)]
   
    assert same_unordered_list_of_actions(actions, expected_actions)
    
    cards = [SquidNigiriCard(), WasabiCard(), TempuraCard()]
    chopsticks_available = True
    
    actions = ActionManager.get_legal_actions(cards, chopsticks_available)
    
    expected_actions = [Action(NigiriCard), 
                        Action(WasabiCard), 
                        Action(TempuraCard), 
                        ChopsticksAction(NigiriCard, WasabiCard), 
                        ChopsticksAction(NigiriCard, TempuraCard), 
                        ChopsticksAction(WasabiCard, TempuraCard), 
                        WasabiBeforeNigiriAction()]
   
    assert same_unordered_list_of_actions(actions, expected_actions)
    
def test_numbers_from_legal_actions():
    
    actions = [Action(MakiCard), Action(WasabiCard)]
    assert ActionManager.numbers_from_legal_actions(actions) == [3, 2]
    
    actions = [Action(MakiCard), 
               Action(TempuraCard), 
               ChopsticksAction(MakiCard, TempuraCard)]
    assert ActionManager.numbers_from_legal_actions(actions) == [3, 5, 23]
    
    actions = [Action(NigiriCard), 
               Action(WasabiCard), 
               Action(TempuraCard), 
               ChopsticksAction(NigiriCard, WasabiCard), 
               ChopsticksAction(NigiriCard, TempuraCard), 
               ChopsticksAction(WasabiCard, TempuraCard), 
               WasabiBeforeNigiriAction()]
    expected_nums = [1, 2, 5, 9, 12, 18, 36]
    assert ActionManager.numbers_from_legal_actions(actions) == expected_nums
    