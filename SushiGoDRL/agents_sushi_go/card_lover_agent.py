import random

from agents_sushi_go.agent import Agent

# from states_sushi_go.player_state import PlayerState
# from states_sushi_go.player_state import PlayerFullState
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
            for action in legal_actions:
                if action == card_type:
                    return action
        else:
            for action in legal_actions:
                if action.is_any_action_of_type(card_type):
                    return action
                    
        return random.choice(legal_actions)
    
    def learn_from_previous_action(self, reward, done, new_legal_actions):
        pass
    
    def save_training(self):
        pass
        

class SuperLoverAgent(Agent):
        
    def __init__(self, player = None):
        
        super(SuperLoverAgent, self).__init__(player)
         
    
    def find_action(legal_actions, card_type):
                 
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
        
class HaterAgent(Agent):
        
    def __init__(self, player = None):
        
        super(HaterAgent, self).__init__(player)
         
    
    def find_action(legal_actions, card_type):
                 
        actions_without_type = []
        
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
        

class SashimiLoverAgent(LoverAgent):  
        
    def __init__(self, player = None):
        
        super(SashimiLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, SashimiCard)
                        

class SashimiSuperLoverAgent(SuperLoverAgent):
        
    def __init__(self, player = None):
        
        super(SashimiSuperLoverAgent, self).__init__(player)

    def choose_action(self, legal_actions):
                
        return SuperLoverAgent.find_action(legal_actions, SashimiCard)
                        

class SashimiHaterAgent(SuperLoverAgent):
        
    def __init__(self, player = None):
        
        super(SashimiHaterAgent, self).__init__(player)

    def choose_action(self, legal_actions):
                
        return HaterAgent.find_action(legal_actions, SashimiCard)
        
class DumplingLoverAgent(LoverAgent):
        
    def __init__(self, player = None):
        
        super(DumplingLoverAgent, self).__init__(player)

    def choose_action(self, legal_actions):
                    
        return LoverAgent.find_action(legal_actions, DumplingCard)
        
class DumplingSuperLoverAgent(SuperLoverAgent):
        
    def __init__(self, player = None):
        
        super(DumplingSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return SuperLoverAgent.find_action(legal_actions, DumplingCard)       
        
class TempuraLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(TempuraLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return LoverAgent.find_action(legal_actions, TempuraCard)
        
class TempuraSuperLoverAgent(SuperLoverAgent):

    def __init__(self, player = None):
        
        super(TempuraSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return SuperLoverAgent.find_action(legal_actions, TempuraCard)    
               
class MakiLoverAgent(LoverAgent):
    
    def __init__(self, player = None):
        
        super(MakiLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return self.find_action(legal_actions, MakiCard)
        
class MakiSuperLoverAgent(SuperLoverAgent):

    def __init__(self, player = None):
        
        super(MakiSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return SuperLoverAgent.find_action(legal_actions, MakiCard)        
        
class MakiHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(MakiHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return HaterAgent.find_action(legal_actions, MakiCard)   
        
class PuddingLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(PuddingLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return LoverAgent.find_action(legal_actions, PuddingCard)
        
class PuddingSuperLoverAgent(SuperLoverAgent):

    def __init__(self, player = None):
        
        super(PuddingSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return SuperLoverAgent.find_action(legal_actions, PuddingCard)    
        
class PuddingHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(PuddingHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return HaterAgent.find_action(legal_actions, PuddingCard) 
        
class ChopstickLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(ChopstickLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return LoverAgent.find_action(legal_actions, ChopsticksCard)
        
class ChopstickHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(ChopstickHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return HaterAgent.find_action(legal_actions, ChopsticksCard) 
    
# class ChopstickLoverAtFirstAgent(LoverAgent):

#     def __init__(self, player = None):
        
#         super(ChopstickLoverAtFirstAgent, self).__init__(player)
         
#     def choose_action(self, legal_actions):
        
#         player_state = PlayerFullState.build_by_player(self.get_player())
        
#         game_state = player_state.get_game_state()
        
#         if game_state.get_current_turn() < 4:                  
#             return LoverAgent.find_action(legal_actions, ChopsticksCard) 
#         else:
#             return HaterAgent.find_action(legal_actions, ChopsticksCard) 
        
class NigiriLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(NigiriLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return LoverAgent.find_action(legal_actions, NigiriCard)
        
class NigiriSuperLoverAgent(SuperLoverAgent):
    
    def __init__(self, player = None):
        
        super(NigiriSuperLoverAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return SuperLoverAgent.find_action(legal_actions, NigiriCard)   
        
class NigiriHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(NigiriHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return HaterAgent.find_action(legal_actions, NigiriCard) 
            
class WasabiLoverAgent(LoverAgent):

    def __init__(self, player = None):
        
        super(WasabiLoverAgent, self).__init__(player)
         
    def choose_action(self, legal_actions):
                    
        return LoverAgent.find_action(legal_actions, WasabiCard) 
            
# class WasabiLoverAtFirstAgent(LoverAgent):

#     def __init__(self, player = None):
        
#         super(WasabiLoverAtFirstAgent, self).__init__(player)
         
#     def choose_action(self, legal_actions):
        
#         player_state = PlayerFullState.build_by_player(self.get_player())
        
#         game_state = player_state.get_game_state()
        
#         if game_state.get_current_turn() < 4:                  
#             return LoverAgent.find_action(legal_actions, WasabiCard) 
#         else:
#             return HaterAgent.find_action(legal_actions, WasabiCard) 
        
class WasabiHaterAgent(HaterAgent):

    def __init__(self, player = None):
        
        super(WasabiHaterAgent, self).__init__(player)
    
    def choose_action(self, legal_actions):
                    
        return HaterAgent.find_action(legal_actions, WasabiCard) 
            