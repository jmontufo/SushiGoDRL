import random

from agents_sushi_go.agent import Agent

class RandomAgent(Agent):

    def __init__(self, player = None):
        
        super(RandomAgent, self).__init__(player)
         
    def choose_action(self, legalActions, rival_legal_actions = None):
                
        return random.choice(legalActions)
    
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    def save_training(self):
        pass
    
    def trained_with_chopsticks_phase(self):
        return True
    
    