# -*- coding: utf-8 -*-

import pickle

from model_sushi_go.game import SingleGame
from model_sushi_go.game import MultiplayerGame

from agents_sushi_go.random_agent import RandomAgent
from agents_sushi_go.card_lover_agent import SashimiLoverAgent
from agents_sushi_go.card_lover_agent import NigiriLoverAgent
from agents_sushi_go.deep_q_learning.deep_q_learning_agent_v2 import DeepQLearningAgentPhase1
from agents_sushi_go.multiagent_learning.minimax_multi_deep_q_learning_agent import MinimaxQAgent

class GameOptions(object):
    
    def __init__(self):
        
        self.setup = "Original"
        self.mode = SingleGame
        self.mode_name = "Single"
        self.agents = []
        
        self.agents.append(RandomAgent)
        self.agents.append(RandomAgent)
        self.agents.append(RandomAgent)
        
        self.num_players = 4
        
    def save(self):
        
        output = open("ui_options.pkl", "wb")
        pickle.dump(self,output)
        output.close()


def get_input_number(initial_text, error_text, max_value, min_value = 1, invalid_value = None):
    
    action_number = None
    text = initial_text
    
    while action_number is None or action_number < min_value or action_number > max_value or action_number == invalid_value:
        
        try:
            action_number = int(input(text))    
        except ValueError:
            action_number = None
        
        text = error_text
        
    return action_number


def modify_setup():
    print ()
    print ("Setup:")
    print ("Current value: " + game_options.setup)
    print ()
    
    print ("\t1: Original")
    print ("\t2: Party-original")
    print ("\t3: Back")
    
    selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", 4)
    
    if selected_number == 1:
        game_options.setup = "Original"
    elif selected_number == 2:
        game_options.setup = "Party-original"
    if selected_number != 3:
        game_options.save()
    

def modify_mode():
    print ()
    print ("Game mode:")
    print ("Current value: " + game_options.setup)
    print ()
    
    print ("\t1: Single mode")
    print ("\t2: Multiplayer mode")
    print ("\t3: Back")
    
    selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", 4)
    
    if selected_number == 1:
        game_options.mode = SingleGame
        game_options.mode_name = "Single Mode"
    elif selected_number == 2:
        game_options.mode = MultiplayerGame
        game_options.mode_name = "Multiplayer Mode"
    if selected_number != 3:
        game_options.save()
    
def modify_agent(agent_number):
    
    agent_index = agent_number - 1
    
    print ()
    print ("Modify agent " + str(agent_number) + ":")
    print ("Current agent: " + str(game_options.agents[agent_index].__name__))    
    print ()

    for i, agent_type in enumerate(available_agents):
        print ("\t" + str(i + 1) + ": " + str(agent_type.__name__))
     
    count = len(available_agents) + 1
    if len(game_options.agents) > 1:   
        print ("\t" + str(len(available_agents) + 1) + ": [Remove]")
        count += 1
        
    print ("\t" + str(count) + ": Back")
        
    selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", len(available_agents) + 1)
            
    if selected_number < len(available_agents) + 1:
        game_options.agents[agent_index] = available_agents[selected_number - 1]
        game_options.save()
    elif selected_number == count - 1:
        game_options.agents.remove(game_options.agents[agent_index])
        game_options.save()
    
def add_agent():
    
    print ()
    print ("Add agent :")
    print ()

    for i, agent_type in enumerate(available_agents):
        print ("\t" + str(i + 1) + ": " + str(agent_type.__name__))
            
    print ("\t" + str(len(available_agents) + 1) + ": Back")
        
    selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", len(available_agents) + 1)
            
    if selected_number < len(available_agents) + 1:
        game_options.agents.append(available_agents[selected_number - 1])
        game_options.save()
    
def modify_agents():
    
    back = False
    
    while not back:
              
        print ()
        print ("Agents:")
        print ()
    
        count = 0
        add_agent_available = False
    
        for i,agent in enumerate(game_options.agents):
            
            print ("\t" + str(i + 1) + ": " + str(agent.__name__))
            count += 1
            
        if count < 4:
            print ("\t" + str(count + 1) + ": [Add agent]")
            count += 1
            add_agent_available = True
            
        print ("\t" + str(count + 1) + ": Back")
        
        selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", count + 1)
        
        if selected_number < len(game_options.agents) + 1:
            modify_agent(selected_number)
        elif add_agent_available and selected_number == count:
            add_agent()
        elif selected_number == count + 1:
            back = True            
    

def modify_num_players():
   
    print ()
    print ("Number of players:")
    print ("Current value: " + str(game_options.num_players))
    print ()
        
    selected_number = get_input_number("Enter number of players (2-5): ", "Please enter a correct value: ", 5, 2)
    
    game_options.num_players = selected_number
    
    game_options.save()

def modify_options():
    
    back = False
    
    while not back:
        print ()
        print ("Options:")
        print ()
        
        print ("\t1: Setup: " + game_options.setup)
        print ("\t2: Game mode: "  + game_options.mode_name)
        if game_options.mode == SingleGame:
            num_agents = len(game_options.agents)
            print ("\t3: Agents: "  + str(num_agents) + " (" + str(num_agents + 1) +" players)")
        else:            
            print ("\t3: Num players: " + str(game_options.num_players))
        print ("\t4: Back")
        
        selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", 4)
        
        if selected_number == 1:
            modify_setup()
        elif selected_number == 2:
            modify_mode()
        elif selected_number == 3:
            if game_options.mode == SingleGame:
                modify_agents()
            else:
                modify_num_players()
        else:
            back = True
        
    return

def play_single_game():
    
    agents = []
    
    for agent_type in game_options.agents:
        agents.append(agent_type())
    
    game = SingleGame(game_options.setup, agents, chopsticks_phase_mode = True)
    second_card = None
    while not game.is_finished():
        if not game.is_in_chopsticks_phase():
            
            for i,score in enumerate(game.report_scores()):        
                print("Player " + str(i+1) + " score: " + str(score))
                
            player = game.get_single_player()
            player_hand = player.get_hand()
            player_table = player.get_table()
            number_of_cards = player_hand.get_num_of_cards()
            
            for player_agent in range(len(game_options.agents)):
                player_num = player_agent + 1
                rival = game.get_player(player_num)
                rival_table = rival.get_table()
            
                print("")
                print("Player " + str(player_num) + " table: ")
                print("")
                
                for card in rival_table.get_cards():
                    print("\t" + str(card))
            
            print("")
            print("Player table: ")
            print("")
            
            for card in player_table.get_cards():
                print("\t" + str(card))
            
            print("")        
            print("Your hand: ")
            print("")        
            
            for card_index, card in enumerate(player_hand.get_cards()):
                print("\t" + str(card_index + 1) + " : " + str(card))
                    
            max_input = number_of_cards
        
            if player.is_chopsticks_move_available():
                max_input = number_of_cards + 1
                print("\t" + str(number_of_cards + 1) + " : Use Chopsticks!")
            
        
            first_action_number = get_input_number("Pick a card from your hand: ", "Please enter a correct card position: ", max_input)
           
            second_action_number = None
        
            if first_action_number == number_of_cards + 1:
            
                first_action_number = get_input_number("What is your first card?: ", "Please enter a correct card position: ", number_of_cards)
                second_action_number = get_input_number("What is your second card?: ", "Please enter a correct card position different from the first: ", number_of_cards, 1, first_action_number)
          
            cards = []
            
            cards.append(type(player_hand.get_cards()[first_action_number - 1]))
            
            if second_action_number is not None:
                second_card = type(player_hand.get_cards()[second_action_number - 1])
            
            game.play_cards(cards)
        else:
            game.play_cards([second_card, None])
            second_card = None
        
    print("")
    print("Scores:")
    print("")
    
    for i,score in enumerate(game.report_scores()):        
        print("Player " + str(i+1) + " score: " + str(score))
     
    print("")   
    if 0 in game.declare_winner():    
        print ("YOU WIN!!!!")
    else:
        print ("Your lose. Try again!")
        

def play():
    
    if game_options.mode == SingleGame:
          
        play_single_game()              
    else:        
        
        print ("Multiplayer games are not available yet")
        return
    
        
    input("End of the game!! Press any button to continue")

try:
    input_options = open("ui_options.pkl", "rb")
    game_options = pickle.load(input_options)
    input_options.close()
except:
    game_options = GameOptions()   



available_agents = [RandomAgent, 
                    SashimiLoverAgent, 
                    NigiriLoverAgent, 
                    DeepQLearningAgentPhase1,
                    MinimaxQAgent
                    ]

end_game = False

print ()
print ("Welcome to Sushi Go!")
print ()

while not end_game:
    
    print ("What to do?:")
    
    print ("\t1: Play!")
    print ("\t2: Modify options")
    print ("\t3: Exit")
    
    selected_number = get_input_number("Enter your selection: ", "Please enter a correct selection: ", 3)
        
    if selected_number == 1:
        play()
    elif selected_number == 2:
        modify_options()
    else:
        end_game = True

print("Bye!")
    

    