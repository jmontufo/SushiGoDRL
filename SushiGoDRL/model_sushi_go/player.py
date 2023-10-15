# -*- coding: utf-8 -*-
from model_sushi_go.hand import Hand
from model_sushi_go.table import Table

class Player(object):
    
    def __init__(self, game, position):
        
        # Game keeps track of overall board state and deck
        self.__game = game
        # Position in circle (0 next to 1 and deck.num_players-1)
        self.__position = position
        # Hand of the player
        self.__hand = None
        self.initialize_player_hand()
        # Cards passed by previous player
        self.__awaiting_hand = None
        # Cards in play
        self.__table = Table(self)
        # Score of player
        self.__score = 0
        # Scored reward
        self.__reward = 0
                
    def get_game(self):
        
        return self.__game
                    
    def get_hand(self):
        
        return self.__hand
    
    def __set_hand(self, hand):
        
        self.__hand = hand
    
    def get_awaiting_hand(self):
        
        return self.__awaiting_hand
    
    def __set_awaiting_hand(self, awaiting_hand):
        
        self.__awaiting_hand = awaiting_hand
    
    def __get_deck(self):
        
        game = self.get_game()
        return game.get_deck()
    
    def get_table(self):
        
        return self.__table
        
    def __set_table(self, table):
            
        self.__table = table
    
    def get_position(self):
        
        return self.__position
    
    def get_current_score(self):
        
        return self.__score       
    
    def get_last_action_reward(self):
        
        return self.__reward
    
    def __set_reward(self, reward):
        
        self.__reward = reward        
        
    def get_maki_rolls(self):
        
        table = self.get_table()
        return table.get_maki_rolls()   
        
    def get_puddings(self):
        
        table = self.get_table()
        return table.get_puddings()        
        
    def __str__(self):
        
        to_string = "Player " + str(self.get_position()) + ":\n"
        to_string += str(self.get_hand())         
        to_string += str(self.get_table())                             
        to_string += "Current score:" + str(self.get_current_score()) + "\n"
        to_string += "Last reward:" + str(self.get_last_action_reward()) + "\n"
        
        return to_string
    
    def initialize_player_hand(self):
        
        deck = self.__get_deck()
        initial_cards = deck.deal_hand()
        self.__set_hand(Hand(initial_cards))
    
    def is_chopsticks_move_available(self):
        
        table = self.get_table()
        hand = self.get_hand()
        
        return table.get_chopsticks() > 0 and hand.get_num_of_cards() > 1    
    
    def get_hand_cards(self):
        
        hand = self.get_hand()
        return hand.get_cards()
    
    def play_a_turn(self, first_card_type, second_card_type = None):
        
        assert first_card_type is not None
           
        chopsticks_action = second_card_type is not None
        
        original_score = self.get_current_score()     
            
        self.__play_a_card(first_card_type) 
                
        if chopsticks_action:
            
            self.__play_a_card(second_card_type) 
            self.__return_chopsticks_to_hand()           
                                        
        turn_reward = self.get_current_score() - original_score
        self.__set_reward(turn_reward)        
        
        self.__hand_cards_to_next_player()        
    
    def __play_a_card(self, played_card_type):
        
        assert played_card_type is not None
        
        hand = self.get_hand() 
        
        played_card = hand.choose_card(played_card_type)          
        assert played_card is not None               
        
        self.__move_card_from_hand_to_table(played_card)   

        self.__add_card_score(played_card)
        
        return    
   
    def __move_card_from_hand_to_table(self, card):
        
        hand = self.get_hand()
        table = self.get_table()

        hand.remove_card(card) 
        table.add_card(card)    
    
    def __add_card_score(self, played_card):
        
        table = self.get_table()        
        played_card_type = type(played_card)
        
        order = table.num_of_cards_of_type(played_card_type)
        
        card_score = played_card.score(order)
                   
        self.__score += card_score           
        
    def __return_chopsticks_to_hand(self):
       
        table = self.get_table()
        hand = self.get_hand()
        
        chopsticks_card = table.pick_chopsticks_card()        
        assert chopsticks_card is not None         
        
        hand.add_card(chopsticks_card)       
    
    def __hand_cards_to_next_player(self):
        
        game = self.get_game()
        
        next_player_position = (self.get_position() + 1) % game.get_num_players()
        next_player = game.get_player(next_player_position)
        
        assert next_player.get_awaiting_hand() is None
        
        next_player.__set_awaiting_hand(self.get_hand())
        
    # Used for the points obtained at the last turn by makis/puddings.
    # The reward of the last turn includes those point.
    def add_partial_score(self, new_score):

        self.__score += new_score
        self.__reward += new_score 
        
    def take_cards_from_other_player(self):
        
        awaiting_hand = self.get_awaiting_hand()
        
        assert awaiting_hand is not None
        
        self.__set_hand(awaiting_hand)            
        self.__set_awaiting_hand(None)      
      
    def give_back_cards(self):
        
        deck = self.__get_deck()
        table = self.get_table()
        
        returned_cards = table.return_cards_except_pudding()
        deck.take_back_cards(returned_cards)        
 
class TestPlayer(object):       
     
    def set_hand(player, hand):
        player._Player__set_hand(hand)
        
    def set_awaiting_hand(player, hand):
        player._Player__set_awaiting_hand(hand)
        
    def set_table(player, table):
        player._Player__set_table(table)
        
    def set_score(player, score):
        player._Player__score = score
        
   