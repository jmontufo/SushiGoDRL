
from abc import ABC, abstractmethod

from model_sushi_go.deck import Deck
from model_sushi_go.card import CARD_TYPES
from model_sushi_go.player import Player
from model_sushi_go.action import ActionManager
from agents_sushi_go.random_agent import RandomAgent
       
class Game(ABC):
    
    def __init__(self, setup, num_players, reward_by_win, 
                 chopsticks_phase_mode = False):
                        
        self.__setup = setup   
        self.__chopsticks_phase_mode = chopsticks_phase_mode         
        self.__num_players = num_players  
        self.__reward_by_win = reward_by_win
        
        self.__init_deck()          
        self.__init_cards_by_round()                            
        self.__init_players()
        
        self.__round = 1
        self.__turn = 1
        self.__phase = 1
        
        self.__log = []           
    
    def __init_deck(self):

        self.__deck = Deck(self)
        
        if self.get_setup() == "Party-original":
            self.__party = True        
        elif self.get_setup() == "Original":
            self.__party = False

    def __init_cards_by_round(self):
        
        num_players = self.get_num_players()
        self.__cards_by_round = 9     
        
        if not self.is_party_version():     
            
            cards_by_num_players = [0,0,10,9,8,7];
            self.__cards_by_round = cards_by_num_players[num_players]
        
    def __init_players(self):
        
        self.__players = []
        
        for player_number in range(0, self.get_num_players()):
            
            player = Player(self, player_number)
            self.__players.append(player)
            
    def get_setup(self):
        
        return self.__setup

    def is_chopsticks_phase_mode(self):
    
        return self.__chopsticks_phase_mode
    
    def get_num_players(self):
        
        return self.__num_players
    
    def get_reward_by_win(self):
        
        return self.__reward_by_win
    
    def get_deck(self):
        
        return self.__deck
        
    def is_party_version(self):
        
        return self.__party
        
    def get_players(self):
        
        return self.__players
    
    def get_player(self, player_number):
        
        return self.__players[player_number]
             
    def get_cards_by_round(self):
        
        return self.__cards_by_round
    
    def get_round(self):
        
        return self.__round
    
    def __increase_round(self):
        
        self.__round += 1
        
    def __round_has_finished(self):
        
        return self.get_turn() == self.get_cards_by_round() + 1
    
    def get_turn(self):
        
        return self.__turn
    
    def __increase_turn(self):
        
        self.__turn += 1
        
    def __reset_turn(self):
        
        self.__turn = 1
    
    def get_phase(self):
        
        return self.__phase
    
    def __increase_phase(self):
        
        self.__phase += 1
        
    def is_in_chopsticks_phase(self):
        
        return self.__phase == 2
    
    def __reset_phase(self):
        
        self.__phase = 1
    
    def __turn_has_finished(self):
        
        return self.__phase == 3 or not self.is_chopsticks_phase_mode()
        
    def is_finished(self):
        
        return self.get_round() >= 4
    
    def get_log(self):
        
        return self.__log
    
    def get_log_string(self):
        
        to_string = ""
        
        for line in self.__log:
            to_string += line + "\n"
            
        return to_string
            
    def __add_legal_actions_to_log(self, player_number, legal_actions):
        
        log = self.get_log()
        
        log.append("Legal actions for player " + str(player_number))
                 
        actions_str = ""
        for action in legal_actions:
            actions_str += str(action) + " - "
        
        log.append(actions_str[:-3])

    def __add_pair_of_cards_to_log(self, cards):
                
        log = self.get_log()
        
        str_cards = str(cards[0])
        
        if cards[1] is not None:
            str_cards += " - " + str(cards[1])
            
        log.append(str_cards)
        
    def __str__(self):
        
        to_string = ""
        
        for player in self.get_players():
            to_string += str(player) + "\n"
            
        return to_string    
    
    def print_log(self, filename):        
        f = open(filename, "a")
        f.write(self.get_log_string())
        f.close()
    
    @abstractmethod
    def get_legal_actions(self):
        pass
    
    @abstractmethod
    def get_legal_actions_numbers(self):
        pass
                
    def __get_legal_actions_of_player(self, player_number):
        
        player = self.get_player(player_number)
        
        chopsticks_available = player.is_chopsticks_move_available()
        
        if self.is_chopsticks_phase_mode():
            
            if chopsticks_available or not self.is_in_chopsticks_phase():
                legal_actions = player.get_hand().get_legal_actions()
            else:
                legal_actions = []
            if self.is_in_chopsticks_phase():
                legal_actions.append(None)
        else:
            player_cards = player.get_hand_cards()
            legal_actions = ActionManager.get_legal_actions(player_cards, chopsticks_available)
        
        self.__add_legal_actions_to_log(player_number, legal_actions)        
                
        return legal_actions
    
    def __get_legal_actions_numbers(self, legal_actions):
        if self.is_chopsticks_phase_mode():
            legal_actions_numbers = []
            for card_type in legal_actions:
                if card_type is None:
                    legal_actions_numbers.append(CARD_TYPES.get_num_types_of_card())  
                else:
                    legal_actions_numbers.append(card_type.get_number())                
        else:
            legal_actions_numbers = ActionManager.numbers_from_legal_actions(legal_actions)
            
        return legal_actions_numbers
    
    @abstractmethod
    def play_cards(self, cards):
        pass
    
    @abstractmethod
    def play_action(self, action):
        pass
    
    @abstractmethod
    def play_action_number(self, action):
        pass
    
    def __play_player_cards(self, player_number, cards):
        
        assert not self.is_finished()
        assert cards is not None
        assert len(cards) > 0 and len(cards) < 3
        
        log = self.get_log()
        
        if len(cards) == 1:
            cards.append(None)
        
        player = self.get_player(player_number)
        
        log.append("Turn of player " + str(player_number))
        log.append(str(player))   
        log.append("Cards played: ")                
        self.__add_pair_of_cards_to_log(cards)
                
        player.play_a_turn(cards[0], cards[1], 
                           self.is_chopsticks_phase_mode(),
                           self.is_in_chopsticks_phase())
        
        log.append ("Before action:")            
        log.append(str(player))   
     
    def __finish_turn(self):         
        
        if self.is_chopsticks_phase_mode():
            self.__increase_phase()
            
        if self.__turn_has_finished():
        
            self.__reset_phase()
            self.__increase_turn();
            
            for player in self.get_players():
                player.take_cards_from_other_player()
                
            if self.__round_has_finished():         
                
                 self.__clear_board_state()
                 
                 self.__reset_turn()
                 self.__increase_round()
                 
                 if self.is_finished():
                     self.__score_pudding()
                     self.__score_victory()
                         
    def __clear_board_state(self):
     
        deck = self.get_deck()   
     
        self.__score_maki_rolls()                                        
        deck.add_dessert()
        
        for player in self.get_players():
            player.give_back_cards()
            
        ## first all cards are returned, next are dealt again
        for player in self.get_players():
            player.initialize_player_hand()
        
    def __score_maki_rolls(self):
        
        players_rolls = []
        
        for player in self.get_players():
            
            player_maki_rolls = player.get_maki_rolls()                            
            players_rolls.append(player_maki_rolls)            
        
        scored = self.__distribute_points(players_rolls, max, 6, update = -1)
                
        if scored == 1:            
            self.__distribute_points(players_rolls, max, 3)     
            
    def __score_pudding(self):
                
        players_puddings = self.__get_players_puddings()
                
        self.__distribute_points(players_puddings, max, 6)
              
        if self.get_num_players() > 2:            
            self.__distribute_points(players_puddings, min, -6)
                        
    def __score_victory(self):
                
        winners = self.declare_winner()
             
        winner_reward = self.get_reward_by_win() / len(winners)
        for winner_position in winners:
            self.get_player(winner_position).add_reward(winner_reward)
            
    def __get_players_puddings(self):
        
        players_puddings = []
        
        for player in self.get_players():
            players_puddings.append(player.get_puddings()) 
            
        return players_puddings
    
    def __get_players_puddings_as_tiebreakers(self, max_score_players):
        
        players_puddings = []
        
        for player in self.get_players():
            if player.get_position() in max_score_players:
                players_puddings.append(player.get_puddings()) 
            else:
                players_puddings.append(-1) 
            
        return players_puddings
                  
    def declare_winner(self):
        
        players_scores = self.report_scores()
            
        max_score_players = Game.__find_all_in_list(players_scores, max)
                
        if len(max_score_players) > 1:
            
            tiebreaker_list = self.__get_players_puddings_as_tiebreakers(max_score_players)
                
            max_score_players = Game.__find_all_in_list(tiebreaker_list, max)
       
        return(max_score_players)
        
    def report_scores(self):
        
        scores_list = []
        
        for player in self.get_players():
            scores_list.append(player.get_current_score())
            
        return(scores_list)
    
    def __distribute_points(self, values, function, points, update = None):
        
        players_numbers = Game.__find_all_in_list(values, function)
                    
        for player_number in players_numbers:
            
            player = self.get_player(player_number)
            
            if self.is_party_version():
                score = points
            else:
                score = int(points / len(players_numbers))                    
                
            player.add_partial_score(score)
            
            if update is not None:
                values[player_number] = update
        
        return len(players_numbers)    
    
    def __find_all_in_list(values, function):
            
        value_to_find = function(values)
        
        if value_to_find == -1:
            return []
        
        index_list = []
        
        for index, value in enumerate(values):
            if value == value_to_find:
                index_list.append(index)
                
        return(index_list)                       
    

class SingleGame(Game):             

    def __init__(self, setup = "Original", 
                 agents  = [RandomAgent()],
                 reward_by_win = 0,
                 chopsticks_phase_mode = False):
      
        num_players = len(agents) + 1

        super(SingleGame, self).__init__(setup, num_players, 
                                         reward_by_win, chopsticks_phase_mode)       
        
        self.__init_agents(agents)  

    def __init_agents(self, agents):
        
        self.__agents = agents
        
        for agent_index, agent in enumerate(agents):
            
            player_number = agent_index + 1
            agent.set_player(self.get_player(player_number))
            
    def __get_agents(self):
        
        return self.__agents
    
    def get_single_player(self):
        
        return self.get_player(0)
            
    def get_legal_actions(self):
                
        return self._Game__get_legal_actions_of_player(0)
    
    def get_legal_actions_numbers(self):
        
        return self._Game__get_legal_actions_numbers(self.get_legal_actions())
    
    def play_cards(self, cards):
       
        try:
            
            self._Game__play_player_cards(0, cards)                     
            
            for agent_index, agent in enumerate(self.__get_agents()):
                    
                player_number = agent_index + 1
                player = self.get_player(player_number)
                
                self.get_log().append ("Turn of player " + str(player_number))            
                self.get_log().append(str(player))
                
                legal_actions = self._Game__get_legal_actions_of_player(player_number)  
                                       
                action = agent.choose_action(legal_actions)
                
                self.get_log().append("Action chosen:")            
                self.get_log().append(str(action))
                if self.is_chopsticks_phase_mode():
                    cards = [action]
                else:
                    cards = action.get_pair_of_cards()
                
                self._Game__play_player_cards(player_number, cards)                   
                 
                self.get_log().append ("After action:")
                    
                self.get_log().append(str(player))
            
            self._Game__finish_turn()                                                       
            
            return self.get_player(0).get_last_action_reward()
        
        except Exception as inst:
            
            for line in self.get_log():
                print(line)
            
            raise inst
    
    def play_action(self, action):
        if self.is_chopsticks_phase_mode():
            cards = [action]
        else:
            cards = action.get_pair_of_cards()
        
        return self.play_cards(cards)
    
    def play_action_number(self, number):        
        if self.is_chopsticks_phase_mode():
            if number < CARD_TYPES.get_num_types_of_card():
                action = CARD_TYPES.get_card_types_by_number()[number]
            else:
                action = None
        else:
            action = ActionManager.action_from_number(number)
        
        return self.play_action(action)
        
class MultiplayerGame(Game):
    
    def __init__(self, setup = "Original", num_players = 2, reward_by_win = 0):
        
        super(MultiplayerGame, self).__init__(setup, num_players, reward_by_win)       
            
    
    def get_legal_actions(self):
        
        legal_actions_of_players = []
        
        for player_number in range(0, self.get_num_players()):
        
            legal_actions =  self._Game__get_legal_actions_of_player(player_number)  
            legal_actions_of_players.append(legal_actions)
            
        return legal_actions_of_players
    
    
    def get_legal_actions_numbers(self):
        
        legal_actions_by_player = self.get_legal_actions()
        
        legal_actions_numbers_of_players = []
        
        for legal_actions in legal_actions_by_player:
        
            legal_actions_numbers = self._Game__get_legal_actions_numbers(legal_actions)            
            legal_actions_numbers_of_players.append(legal_actions_numbers)
            
        return legal_actions_numbers_of_players
              
                       
    def play_cards(self, cards):
               
        try:
            
            for player_number in range(0, self.get_num_players()):
                
                players_cards = cards[player_number]           
                self._Game__play_player_cards(player_number, players_cards)          
            
            self._Game__finish_turn()                             
               
            rewards = []       
            
            for player_number in range(0, self.get_num_players()):                
                rewards.append(self.get_player(player_number).get_last_action_reward())
                
            return rewards
                     
        except Exception as inst:
            
            for line in self.get_log():
                print(line)
            
            raise inst        
     
    def play_action(self, action):
        
        cards_by_player = []
        
        for player_number in range(0, self.get_num_players()):
            player_action = action[player_number]
            cards = player_action.get_pair_of_cards()
            cards_by_player.append(cards)
        
        return self.play_cards(cards_by_player)
    
    
    def play_action_number(self, number):
        
        actions_by_player = []
        
        for player_number in range(0, self.get_num_players()):
            player_action_number = number[player_number]
            action = ActionManager.action_from_number(player_action_number)
            actions_by_player.append(action)
                    
        return self.play_action(actions_by_player)

class TestSingleGame(object):       
     
    def get_agents(game):
        return game._SingleGame__get_agents()
        
    def set_phase(game, phase):
        game._Game__phase = phase
        
    def set_turn(game, turn):
        game._Game__turn = turn
        
    def set_round(game, round):
        game._Game__round = round
        
        