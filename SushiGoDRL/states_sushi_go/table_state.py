# -*- coding: utf-8 -*-

import numpy as np

from states_sushi_go.state import State

from model_sushi_go.card import TempuraCard
from model_sushi_go.card import SashimiCard
from model_sushi_go.card import DumplingCard

class TableState(State):
    
    tempura_values = 2
    sashimi_values = 3
    dumplings_values = 6
    maki_values = 10 # value 9 represents 9 or more
    pudding_values = 5 # value 4 represents 4 or more
    chopsticks_values = 2 # value 1 represents 1 or more
    wasabi_values = 2 # value 1 represents 1 or more
        
    def __init__(self):
                
        self.__tempuras = 0
        self.__sashimis = 0
        self.__dumplings = 0
        self.__makis = 0
        self.__puddings = 0
        self.__chopsticks = 0
        self.__wasabis = 0                 
        
    def get_tempuras(self):
        
        return self.__tempuras
        
    def __set_tempuras(self, tempuras):
        
        self.__tempuras = tempuras   
        
    def __increase_tempuras(self):
        
        self.__tempuras += 1    
        
    def get_sashimis(self):
        
        return self.__sashimis
        
    def __set_sashimis(self, sashimis):
        
        self.__sashimis = sashimis 
        
    def __increase_sashimis(self):
        
        self.__sashimis += 1   
        
    def get_dumplings(self):
        
        return self.__dumplings
        
    def __set_dumplings(self, dumplings):
        
        self.__dumplings = dumplings
        
    def __increase_dumplings(self):
        
        self.__dumplings += 1
        
    def get_makis(self):
        
        return self.__makis
        
    def __set_makis(self, makis):
        
        self.__makis = makis
        
    def get_puddings(self):
        
        return self.__puddings
        
    def __set_puddings(self, puddings):
        
        self.__puddings = puddings
        
    def get_chopsticks(self):
        
        return self.__chopsticks
        
    def __set_chopsticks(self, chopsticks):
        
        self.__chopsticks = chopsticks
        
    def get_wasabis(self):
        
        return self.__wasabis
        
    def __set_wasabis(self, wasabis):
        
        self.__wasabis = wasabis
         
    def __get_values_from_table(self, table):
                
        for card in table.get_cards():
            
            if card.get_number() == TempuraCard.get_number():
                self.__increase_tempuras()
            elif card.get_number() == SashimiCard.get_number():
                self.__increase_sashimis()
            elif card.get_number() == DumplingCard.get_number():
                self.__increase_dumplings()        
                
        self.__set_makis(table.get_maki_rolls())
        self.__set_puddings(table.get_puddings())
        self.__set_chopsticks(table.get_chopsticks())
        self.__set_wasabis(table.get_wasabis())  
        
        self.__cut_by_num_of_values()
     
    def __cut_by_num_of_values(self):

        self.__set_tempuras(self.get_tempuras() % self.tempura_values)
        self.__set_sashimis(self.get_sashimis() % self.sashimi_values)
        self.__set_dumplings(min(self.get_dumplings(), self.dumplings_values - 1))
        self.__set_makis(min(self.get_makis(), self.maki_values - 1))
        self.__set_puddings(min(self.get_puddings(), self.pudding_values - 1))
        self.__set_chopsticks(min(self.get_chopsticks(), self.chopsticks_values - 1))        
        self.__set_wasabis(min(self.get_wasabis(), self.wasabi_values - 1))
                            
    def __get_values_from_number(self, number):
        
        self.__set_tempuras(number % TableState.tempura_values)
        number = int(number / TableState.tempura_values) 
        
        self.__set_sashimis(number % TableState.sashimi_values)
        number = int(number / TableState.sashimi_values) 
        
        self.__set_dumplings(number % TableState.dumplings_values)
        number = int(number / TableState.dumplings_values) 
        
        self.__set_makis(number % TableState.maki_values)
        number = int(number / TableState.maki_values) 
        
        self.__set_puddings(number % TableState.pudding_values)
        number = int(number / TableState.pudding_values) 
        
        self.__set_chopsticks(number % TableState.chopsticks_values)
        number = int(number / TableState.chopsticks_values) 
        
        self.__set_wasabis(number % TableState.wasabi_values)
        number = int(number / TableState.wasabi_values)     
                       
    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_tempuras() * base
        base *= self.tempura_values
        
        accumulated_value += self.get_sashimis() * base
        base *= self.sashimi_values
        
        accumulated_value += self.get_dumplings() * base
        base *= self.dumplings_values
        
        accumulated_value += self.get_makis() * base
        base *= self.maki_values
        
        accumulated_value += self.get_puddings() * base
        base *= self.pudding_values
        
        accumulated_value += self.get_chopsticks() * base
        base *= self.chopsticks_values
        
        accumulated_value += self.get_wasabis() * base
        base *= self.wasabi_values
        
        return accumulated_value    
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_tempuras(observation[0])
        self.__set_sashimis(observation[1])   
        self.__set_dumplings(observation[2])   
        self.__set_makis(observation[3])   
        self.__set_puddings(observation[4])   
        self.__set_chopsticks(observation[5])   
        self.__set_wasabis(observation[6])  
            
    def get_as_observation(self):
        
        state = (self.get_tempuras(), 
                 self.get_sashimis(), 
                 self.get_dumplings(), 
                 self.get_makis(), 
                 self.get_puddings(), 
                 self.get_chopsticks(), 
                 self.get_wasabis())
        
        return np.array(state)
        
    def build_by_player(player):
        
        table = player.get_table()
        
        table_state = TableState()        
        table_state.__get_values_from_table(table)
        
        return table_state

    def build_by_number(number):
        
        table_state = TableState()        
        table_state.__get_values_from_number(number)
        
        return table_state
     
    def build_by_observation(observation):
        
        table_state = TableState()
        table_state.__get_values_from_observation(observation)             
        
        return table_state       

    def get_total_numbers():
        
        base = 1    
        base *= TableState.tempura_values
        base *= TableState.sashimi_values
        base *= TableState.dumplings_values
        base *= TableState.maki_values
        base *= TableState.pudding_values
        base *= TableState.chopsticks_values
        base *= TableState.wasabi_values
        
        return base    
    
    def get_low_values():
        return np.array([0] * 7)
    
    def get_high_values():
        
        high_values = [TableState.tempura_values, 
                       TableState.sashimi_values, 
                       TableState.dumplings_values, 
                       TableState.maki_values, 
                       TableState.pudding_values, 
                       TableState.chopsticks_values, 
                       TableState.wasabi_values]
        
        return np.array(high_values)
    
    
    def get_number_of_observations():
        
        return 7
            
    def get_expected_distribution():
        
        distribution = ['Uniform', 'Uniform', 'Poisson', 'Poisson', 'Poisson', 'Uniform', 'Uniform']
        
        return np.array(distribution)
    
class TableSimplState(State):
    
    tempura_values = 2
    sashimi_values = 3
    dumplings_values = 3 # value 1 represents 3-4, 2 represents 5 or more 
    maki_values = 2 # value 1 represents 6 or more
    pudding_values = 2 # value 1 represents 3 or more
    chopsticks_values = 2 # value 1 represents 1 or more
    wasabi_values = 2 # value 1 represents 1 or more
        
    def __init__(self):
                
        self.__tempuras = 0
        self.__sashimis = 0
        self.__dumplings = 0
        self.__makis = 0
        self.__puddings = 0
        self.__chopsticks = 0
        self.__wasabis = 0                 
        
    def get_tempuras(self):
        
        return self.__tempuras
        
    def __set_tempuras(self, tempuras):
        
        self.__tempuras = tempuras   
        
    def __increase_tempuras(self):
        
        self.__tempuras += 1    
        
    def get_sashimis(self):
        
        return self.__sashimis
        
    def __set_sashimis(self, sashimis):
        
        self.__sashimis = sashimis 
        
    def __increase_sashimis(self):
        
        self.__sashimis += 1   
        
    def get_dumplings(self):
        
        return self.__dumplings
        
    def __set_dumplings(self, dumplings):
        
        self.__dumplings = dumplings
        
    def __increase_dumplings(self):
        
        self.__dumplings += 1
        
    def get_makis(self):
        
        return self.__makis
        
    def __set_makis(self, makis):
        
        self.__makis = makis
        
    def get_puddings(self):
        
        return self.__puddings
        
    def __set_puddings(self, puddings):
        
        self.__puddings = puddings
        
    def get_chopsticks(self):
        
        return self.__chopsticks
        
    def __set_chopsticks(self, chopsticks):
        
        self.__chopsticks = chopsticks
        
    def get_wasabis(self):
        
        return self.__wasabis
        
    def __set_wasabis(self, wasabis):
        
        self.__wasabis = wasabis
         
    def __get_values_from_table(self, table):
                
        for card in table.get_cards():
            
            if card.get_number() == TempuraCard.get_number():
                self.__increase_tempuras()
            elif card.get_number() == SashimiCard.get_number():
                self.__increase_sashimis()
            elif card.get_number() == DumplingCard.get_number():
                self.__increase_dumplings()        
                
        self.__set_makis(table.get_maki_rolls())
        self.__set_puddings(table.get_puddings())
        self.__set_chopsticks(table.get_chopsticks())
        self.__set_wasabis(table.get_wasabis())  
        
        self.__cut_by_num_of_values()
     
    def __cut_by_num_of_values(self):

        self.__set_tempuras(self.get_tempuras() % self.tempura_values)
        self.__set_sashimis(self.get_sashimis() % self.sashimi_values)
        
        if self.get_dumplings() < 3:
            self.__set_dumplings(0)
        elif self.get_dumplings() < 5:            
            self.__set_dumplings(1)
        else:                      
            self.__set_dumplings(2)
            
        if self.get_makis() < 6:            
            self.__set_makis(0)
        else:                      
            self.__set_makis(1)
            
        if self.get_puddings() < 3:            
            self.__set_puddings(0)
        else:                      
            self.__set_puddings(1)
            
        if self.get_chopsticks() == 0:            
            self.__set_chopsticks(0)
        else:                      
            self.__set_chopsticks(1)
            
        if self.get_wasabis() == 0:            
            self.__set_wasabis(0)
        else:                      
            self.__set_wasabis(1)
                            
    def __get_values_from_number(self, number):
        
        self.__set_tempuras(number % TableSimplState.tempura_values)
        number = int(number / TableSimplState.tempura_values) 
        
        self.__set_sashimis(number % TableSimplState.sashimi_values)
        number = int(number / TableSimplState.sashimi_values) 
        
        self.__set_dumplings(number % TableSimplState.dumplings_values)
        number = int(number / TableSimplState.dumplings_values) 
        
        self.__set_makis(number % TableSimplState.maki_values)
        number = int(number / TableSimplState.maki_values) 
        
        self.__set_puddings(number % TableSimplState.pudding_values)
        number = int(number / TableSimplState.pudding_values) 
        
        self.__set_chopsticks(number % TableSimplState.chopsticks_values)
        number = int(number / TableSimplState.chopsticks_values) 
        
        self.__set_wasabis(number % TableSimplState.wasabi_values)
        number = int(number / TableSimplState.wasabi_values)     
                       
    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_tempuras() * base
        base *= self.tempura_values
        
        accumulated_value += self.get_sashimis() * base
        base *= self.sashimi_values
        
        accumulated_value += self.get_dumplings() * base
        base *= self.dumplings_values
        
        accumulated_value += self.get_makis() * base
        base *= self.maki_values
        
        accumulated_value += self.get_puddings() * base
        base *= self.pudding_values
        
        accumulated_value += self.get_chopsticks() * base
        base *= self.chopsticks_values
        
        accumulated_value += self.get_wasabis() * base
        base *= self.wasabi_values
        
        return accumulated_value    
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_tempuras(observation[0])
        self.__set_sashimis(observation[1])   
        self.__set_dumplings(observation[2])   
        self.__set_makis(observation[3])   
        self.__set_puddings(observation[4])   
        self.__set_chopsticks(observation[5])   
        self.__set_wasabis(observation[6])   
    
    
    def get_as_observation(self):
        
        state = (self.get_tempuras(), 
                 self.get_sashimis(), 
                 self.get_dumplings(), 
                 self.get_makis(), 
                 self.get_puddings(), 
                 self.get_chopsticks(), 
                 self.get_wasabis())
        
        return np.array(state)
        
    def build_by_player(player):
        
        table = player.get_table()
        
        table_state = TableSimplState()        
        table_state.__get_values_from_table(table)
        
        return table_state

    def build_by_number(number):
        
        table_state = TableSimplState()        
        table_state.__get_values_from_number(number)
        
        return table_state
     
    def build_by_observation(observation):
        
        table_state = TableSimplState()
        table_state.__get_values_from_observation(observation)             
        
        return table_state       

    def get_total_numbers():
        
        base = 1    
        base *= TableSimplState.tempura_values
        base *= TableSimplState.sashimi_values
        base *= TableSimplState.dumplings_values
        base *= TableSimplState.maki_values
        base *= TableSimplState.pudding_values
        base *= TableSimplState.chopsticks_values
        base *= TableSimplState.wasabi_values
        
        return base    
    
    def get_low_values():
        return np.array([0] * 7)
    
    def get_high_values():
        
        high_values = [TableSimplState.tempura_values, 
                       TableSimplState.sashimi_values, 
                       TableSimplState.dumplings_values, 
                       TableSimplState.maki_values, 
                       TableSimplState.pudding_values, 
                       TableSimplState.chopsticks_values, 
                       TableSimplState.wasabi_values]
        
        return np.array(high_values)
    
    def get_number_of_observations():
        
        return 7
            
    def get_expected_distribution():
        
        distribution = ['Uniform', 'Uniform', 'Uniform', 'Uniform', 'Uniform', 'Uniform', 'Uniform']
        
        return np.array(distribution)
    
class TableFullState(State):
    
    tempura_values = 2
    sashimi_values = 3
    dumplings_values = 6
    maki_values = 30 
    pudding_values = 10
    chopsticks_values = 6
    wasabi_values = 6
        
    def __init__(self):
                
        self.__tempuras = 0
        self.__sashimis = 0
        self.__dumplings = 0
        self.__makis = 0
        self.__puddings = 0
        self.__chopsticks = 0
        self.__wasabis = 0                 
        
    def get_tempuras(self):
        
        return self.__tempuras
        
    def __set_tempuras(self, tempuras):
        
        self.__tempuras = tempuras   
        
    def __increase_tempuras(self):
        
        self.__tempuras += 1    
        
    def get_sashimis(self):
        
        return self.__sashimis
        
    def __set_sashimis(self, sashimis):
        
        self.__sashimis = sashimis 
        
    def __increase_sashimis(self):
        
        self.__sashimis += 1   
        
    def get_dumplings(self):
        
        return self.__dumplings
        
    def __set_dumplings(self, dumplings):
        
        self.__dumplings = dumplings
        
    def __increase_dumplings(self):
        
        self.__dumplings += 1
        
    def get_makis(self):
        
        return self.__makis
        
    def __set_makis(self, makis):
        
        self.__makis = makis
        
    def get_puddings(self):
        
        return self.__puddings
        
    def __set_puddings(self, puddings):
        
        self.__puddings = puddings
        
    def get_chopsticks(self):
        
        return self.__chopsticks
        
    def __set_chopsticks(self, chopsticks):
        
        self.__chopsticks = chopsticks
        
    def get_wasabis(self):
        
        return self.__wasabis
        
    def __set_wasabis(self, wasabis):
        
        self.__wasabis = wasabis
         
    def __get_values_from_table(self, table):
                
        for card in table.get_cards():
            
            if card.get_number() == TempuraCard.get_number():
                self.__increase_tempuras()
            elif card.get_number() == SashimiCard.get_number():
                self.__increase_sashimis()
            elif card.get_number() == DumplingCard.get_number():
                self.__increase_dumplings()        
                
        self.__set_makis(table.get_maki_rolls())
        self.__set_puddings(table.get_puddings())
        self.__set_chopsticks(table.get_chopsticks())
        self.__set_wasabis(table.get_wasabis())  
        
        self.__cut_by_num_of_values()
     
    def __cut_by_num_of_values(self):

        self.__set_tempuras(self.get_tempuras() % self.tempura_values)
        self.__set_sashimis(self.get_sashimis() % self.sashimi_values)
        self.__set_dumplings(min(self.get_dumplings(), 5))
        self.__set_makis(self.get_makis())
        self.__set_puddings(self.get_puddings())
        self.__set_chopsticks(min(self.get_chopsticks(),5))
        self.__set_wasabis(min(self.get_wasabis(),5))
                            
    def __get_values_from_number(self, number):
        
        self.__set_tempuras(number % TableState.tempura_values)
        number = int(number / TableState.tempura_values) 
        
        self.__set_sashimis(number % TableState.sashimi_values)
        number = int(number / TableState.sashimi_values) 
        
        self.__set_dumplings(number % TableState.dumplings_values)
        number = int(number / TableState.dumplings_values) 
        
        self.__set_makis(number % TableState.maki_values)
        number = int(number / TableState.maki_values) 
        
        self.__set_puddings(number % TableState.pudding_values)
        number = int(number / TableState.pudding_values) 
        
        self.__set_chopsticks(number % TableState.chopsticks_values)
        number = int(number / TableState.chopsticks_values) 
        
        self.__set_wasabis(number % TableState.wasabi_values)
        number = int(number / TableState.wasabi_values)     
                       
    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_tempuras() * base
        base *= self.tempura_values
        
        accumulated_value += self.get_sashimis() * base
        base *= self.sashimi_values
        
        accumulated_value += self.get_dumplings() * base
        base *= self.dumplings_values
        
        accumulated_value += self.get_makis() * base
        base *= self.maki_values
        
        accumulated_value += self.get_puddings() * base
        base *= self.pudding_values
        
        accumulated_value += self.get_chopsticks() * base
        base *= self.chopsticks_values
        
        accumulated_value += self.get_wasabis() * base
        base *= self.wasabi_values
        
        return accumulated_value    
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_tempuras(observation[0])
        self.__set_sashimis(observation[1])   
        self.__set_dumplings(observation[2])   
        self.__set_makis(observation[3])   
        self.__set_puddings(observation[4])   
        self.__set_chopsticks(observation[5])   
        self.__set_wasabis(observation[6])   
        
    def get_as_observation(self):
        
        state = (self.get_tempuras(), 
                 self.get_sashimis(), 
                 self.get_dumplings(), 
                 self.get_makis(), 
                 self.get_puddings(), 
                 self.get_chopsticks(), 
                 self.get_wasabis())
        
        return np.array(state)
        
    def build_by_player(player):
        
        table = player.get_table()
        
        table_state = TableFullState()        
        table_state.__get_values_from_table(table)
        
        return table_state

    def build_by_number(number):
        
        table_state = TableFullState()        
        table_state.__get_values_from_number(number)
        
        return table_state
     
    def build_by_observation(observation):
        
        table_state = TableFullState()
        table_state.__get_values_from_observation(observation)             
        
        return table_state       

    def get_total_numbers():
        
        base = 1    
        base *= TableFullState.tempura_values
        base *= TableFullState.sashimi_values
        base *= TableFullState.dumplings_values
        base *= TableFullState.maki_values
        base *= TableFullState.pudding_values
        base *= TableFullState.chopsticks_values
        base *= TableFullState.wasabi_values
        
        return base    
    
    def get_low_values():
        return np.array([0] * 7)
    
    def get_high_values():
        
        high_values = [TableFullState.tempura_values, 
                       TableFullState.sashimi_values, 
                       TableFullState.dumplings_values, 
                       TableFullState.maki_values, 
                       TableFullState.pudding_values, 
                       TableFullState.chopsticks_values, 
                       TableFullState.wasabi_values]
        
        return np.array(high_values)
    
    
    def get_number_of_observations():
        
        return 7   
    
    def get_expected_distribution():
        
        distribution = ['Uniform', 'Uniform', 'Poisson', 'Poisson', 'Poisson', 'Poisson', 'Poisson']
        
        return np.array(distribution)
    