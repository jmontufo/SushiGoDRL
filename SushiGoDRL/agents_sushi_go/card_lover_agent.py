import random

from agents_sushi_go.agent import Agent

from model_sushi_go.card import TempuraCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import MakiCard
from model_sushi_go.card import DumplingCard
from model_sushi_go.card import ChopsticksCard
from model_sushi_go.card import PuddingCard
from model_sushi_go.card import NigiriCard
from model_sushi_go.card import WasabiCard

class LoverAgent(Agent):
        
    def __init__(self, player = None):
        
        super(LoverAgent, self).__init__(player)
         
    
    def find_action(self, legal_actions, card_type):
                 
        if super(LoverAgent, self).is_chopsticks_phase_mode():
            if card_type in legal_actions:
                return card_type
        else:
            for action in legal_actions:
                if action.is_any_action_of_type(card_type):
                    return action
                    
        return random.choice(legal_actions)
    
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    def save_training(self):
        pass
    
    def trained_with_chopsticks_phase(self):
        return True
    
        
class SuperLoverAgent(Agent):
        
    def __init__(self, player = None):
        
        super(SuperLoverAgent, self).__init__(player)
         
    def find_action(self, legal_actions, card_type):
            
        # Super Lover agents make no sense in chopsticks phase mode
        assert not super(SuperLoverAgent, self).is_chopsticks_phase_mode()
        
        actions_with_type = []
        
        for action in legal_actions:
            if action.is_chopsticks_action_with_type_twice(card_type):
                return action
            elif action.is_any_action_of_type(card_type):
                actions_with_type.append(action)
            
        if len(actions_with_type) > 0:
            return random.choice(actions_with_type)
        else:    
            return random.choice(legal_actions)
    
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    def save_training(self):
        pass
    
    def trained_with_chopsticks_phase(self):
        return False
    
        
class HaterAgent(Agent):
        
    def __init__(self, player = None):
        
        super(HaterAgent, self).__init__(player)
         
    
    def find_action(self, legal_actions, card_type):
        
        assert len(legal_actions) > 0         
        actions_without_type = []
        
        if super(HaterAgent, self).is_chopsticks_phase_mode():
            if card_type in legal_actions and len(legal_actions) > 1:
                actions_without_type = legal_actions
                actions_without_type.remove(card_type)
        else:
            for action in legal_actions:
                if not action.is_any_action_of_type(card_type):
                    actions_without_type.append(action)
                    
        if len(actions_without_type) > 0:
            return random.choice(actions_without_type)
        else:    
            return random.choice(legal_actions)
    
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    def save_training(self):
        pass
            
    def trained_with_chopsticks_phase(self):
        return True
    

class SashimiLoverAgent(LoverAgent):  
        
    def __init__(self, player = None):
        
        super(SashimiLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, SashimiCard)
                        

class SashimiSuperLoverAgent(SuperLoverAgent):
        
    def __init__(self, player = None):
        
        super(SashimiSuperLoverAgent, self).__init__(player)

    def choose_action(self, legal_actions):
                
        return self.find_action(legal_actions, SashimiCard)
                        

class SashimiHaterAgent(HaterAgent):
        
    def __init__(self, player = None):
        
        super(SashimiHaterAgent, self).__init__(player)

    def choose_action(self, legal_actions):
                
        return self.find_action(legal_actions, SashimiCard)
        
class DumplingLoverAgent(LoverAgent):
        
    def __init__(self, player = None):
        
        super(DumplingLoverAgent, self).__init__(player)

    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, DumplingCard)
        
class DumplingSuperLoverAgent(SuperLoverAgent):
        
    def __init__(self, player = None):
        
        super(DumplingSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, DumplingCard)       
        
class TempuraLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(TempuraLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, TempuraCard)
        
class TempuraSuperLoverAgent(SuperLoverAgent):

    def __init__(self, player = None):
        
        super(TempuraSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, TempuraCard)    
               
class MakiLoverAgent(LoverAgent):
    
    def __init__(self, player = None):
        
        super(MakiLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, MakiCard)
        
class MakiSuperLoverAgent(SuperLoverAgent):

    def __init__(self, player = None):
        
        super(MakiSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, MakiCard)        
        
class MakiHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(MakiHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, MakiCard)   
        
class PuddingLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(PuddingLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, PuddingCard)
        
class PuddingSuperLoverAgent(SuperLoverAgent):

    def __init__(self, player = None):
        
        super(PuddingSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, PuddingCard)    
        
class PuddingHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(PuddingHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, PuddingCard) 
        
class ChopstickLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(ChopstickLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, ChopsticksCard)
        
class ChopstickHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(ChopstickHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, ChopsticksCard) 
    
class ChopstickLoverAtFirstAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(ChopstickLoverAtFirstAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
        
        if self.get_player().get_game().get_turn() < 4:                  
            return self.find_action(legal_actions, ChopsticksCard) 
        else:
            actions_without_type = []
            
            if super(ChopstickLoverAtFirstAgent, self).is_chopsticks_phase_mode():
                if ChopsticksCard in legal_actions and len(legal_actions) > 1:
                    actions_without_type = legal_actions
                    actions_without_type.remove(ChopsticksCard)
            else:
                for action in legal_actions:
                    if not action.is_any_action_of_type(ChopsticksCard):
                        actions_without_type.append(action)
                        
            if len(actions_without_type) > 0:
                return random.choice(actions_without_type)
            else:    
                return random.choice(legal_actions) 
        
class NigiriLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(NigiriLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, NigiriCard)
        
class NigiriSuperLoverAgent(SuperLoverAgent):
    
    def __init__(self, player = None):
        
        super(NigiriSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, NigiriCard)   
        
class NigiriHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(NigiriHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, NigiriCard) 
            
class WasabiLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(WasabiLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, WasabiCard) 
            
class WasabiLoverAtFirstAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(WasabiLoverAtFirstAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                
        if self.get_player().get_game().get_turn() < 4:                  
            return self.find_action(legal_actions, WasabiCard)
        else:
            actions_without_type = []
            
            if super(WasabiLoverAtFirstAgent, self).is_chopsticks_phase_mode():
                if WasabiCard in legal_actions and len(legal_actions) > 1:
                    actions_without_type = legal_actions
                    actions_without_type.remove(WasabiCard)
            else:
                for action in legal_actions:
                    if not action.is_any_action_of_type(WasabiCard):
                        actions_without_type.append(action)
                        
            if len(actions_without_type) > 0:
                return random.choice(actions_without_type)
            else:    
                return random.choice(legal_actions)
        
class WasabiHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(WasabiHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, WasabiCard) 
            