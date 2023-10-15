# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:08:12 2023

@author: jmont
"""
# from model_sushi_go.card import EggNigiriCard
# from model_sushi_go.card import WasabiCard

# wasabiCard = WasabiCard()    
# eggNigiriCard = EggNigiriCard()

# wasabiCard.set_attached_nigiri(eggNigiriCard)
# aa =  wasabiCard.get_attached_nigiri()

# wasabiCard.return_to_deck()    
# assert wasabiCard.get_attached_nigiri() == None


import numpy as np

cards = [324,342]
cards_by_round = 7
 
np.random.shuffle(cards)
     
hand_dealt = cards[0:cards_by_round]
deck_cards = (cards[cards_by_round:])
 
print(hand_dealt)