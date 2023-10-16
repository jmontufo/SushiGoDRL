
import numpy as np


from states_sushi_go.state import State
from states_sushi_go.hand_state import HandState, HandStateWithoutZeroValue
from states_sushi_go.table_state import TableState, TableSimplState, TableFullState
from states_sushi_go.game_state import GameState

class PlayerState(State):    
    

    def __init__(self):         

        self.__hand_state = None
        self.__table_state = None
        
    def get_hand_state(self):
        
        return self.__hand_state
        
    def __set_hand_state(self, hand_state):
        
        self.__hand_state = hand_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_hand_state(HandState.build_by_player(player))        
        self.__set_table_state(TableState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        hand_state_total_values = HandState.get_total_values()
        hand_state_number = number % hand_state_total_values   
    
        self.__set_hand_state(HandState.build_by_number(hand_state_number))
        
        table_state_number = int(number / hand_state_total_values) 
        
        self.__set_table_state(TableState.build_by_number(table_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_hand_state(HandState.build_by_observation(observation[0:3]))
        self.__set_table_state(TableState.build_by_observation(observation[3:]))
        
    def get_as_observation(self):
        
        hand_state_observation = self.get_hand_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        
        return np.concatenate((hand_state_observation, table_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= HandState.get_total_numbers()
        base *= TableState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((HandState.get_low_values(), TableState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((HandState.get_high_values(), TableState.get_high_values()))    
    
    def get_number_of_observations():
    
        return HandState.get_number_of_observations() + TableState.get_number_of_observations()
             
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(HandState.get_expected_distribution())
        distribution.append(TableState.get_expected_distribution())
                
        return np.concatenate(distribution)       

class PlayerSimplState(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__table_state = None
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameState.build_by_player(player))        
        self.__set_table_state(TableSimplState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameState.get_total_values()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameState.build_by_number(game_state_number))
        
        table_state_number = int(number / game_state_total_values) 
        
        self.__set_table_state(TableSimplState.build_by_number(table_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableSimplState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableSimplState.build_by_observation(observation[2:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerSimplState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerSimplState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerSimplState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameState.get_total_numbers()
        base *= TableSimplState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameState.get_low_values(), TableSimplState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameState.get_high_values(), TableSimplState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameState.get_number_of_observations() + TableSimplState.get_number_of_observations()
        
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameState.get_expected_distribution())
        distribution.append(TableSimplState.get_expected_distribution())
                
        return np.concatenate(distribution)
    
class PlayerSimpleStateV2(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def get_hand_state(self):
        
        return self.__hand_state
        
    def __set_hand_state(self, table_state):
        
        self.__hand_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameState.build_by_player(player))        
        self.__set_table_state(TableSimplState.build_by_player(player))
        self.__set_hand_state(HandState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameState.get_total_numbers()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameState.build_by_number(game_state_number))
        
        number = int(number / game_state_total_values) 
        
        table_state_total_values = TableSimplState.get_total_numbers()
        table_state_number = number % table_state_total_values   
    
        self.__set_game_state(TableSimplState.build_by_number(table_state_number))
        
        hand_state_number = int(number / table_state_total_values) 
        
        self.__set_table_state(HandState.build_by_number(hand_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableSimplState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableSimplState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandState.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerSimpleStateV2()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerSimpleStateV2()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerSimpleStateV2()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameState.get_total_numbers()
        base *= TableSimplState.get_total_numbers()
        base *= HandState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameState.get_low_values(), TableSimplState.get_low_values(), HandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameState.get_high_values(), TableSimplState.get_high_values(), HandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameState.get_number_of_observations() + TableSimplState.get_number_of_observations() + HandState.get_number_of_observations()
      
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameState.get_expected_distribution())
        distribution.append(TableSimplState.get_expected_distribution())                
        distribution.append(HandState.get_expected_distribution())
                
        return np.concatenate(distribution)
    
class PlayerMidState(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def get_hand_state(self):
        
        return self.__hand_state
        
    def __set_hand_state(self, table_state):
        
        self.__hand_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameState.build_by_player(player))        
        self.__set_table_state(TableState.build_by_player(player))
        self.__set_hand_state(HandState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameState.get_total_numbers()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameState.build_by_number(game_state_number))
        
        number = int(number / game_state_total_values) 
        
        table_state_total_values = TableState.get_total_numbers()
        table_state_number = number % table_state_total_values   
    
        self.__set_game_state(TableState.build_by_number(table_state_number))
        
        hand_state_number = int(number / table_state_total_values) 
        
        self.__set_table_state(HandState.build_by_number(hand_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandState.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerMidState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerMidState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerMidState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameState.get_total_numbers()
        base *= TableState.get_total_numbers()
        base *= HandState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameState.get_low_values(), TableState.get_low_values(), HandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameState.get_high_values(), TableState.get_high_values(), HandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameState.get_number_of_observations() + TableState.get_number_of_observations() + HandState.get_number_of_observations()
            
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameState.get_expected_distribution())
        distribution.append(TableState.get_expected_distribution())                
        distribution.append(HandState.get_expected_distribution())
                
        return np.concatenate(distribution)
    
class PlayerMidStateV2(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__table_state = None
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameState.build_by_player(player))        
        self.__set_table_state(TableState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameState.get_total_values()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameState.build_by_number(game_state_number))
        
        table_state_number = int(number / game_state_total_values) 
        
        self.__set_table_state(TableState.build_by_number(table_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableState.build_by_observation(observation[2:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerMidStateV2()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerMidStateV2()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerMidStateV2()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameState.get_total_numbers()
        base *= TableState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameState.get_low_values(), TableState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameState.get_high_values(), TableState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameState.get_number_of_observations() + TableState.get_number_of_observations()
                    
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameState.get_expected_distribution())
        distribution.append(TableState.get_expected_distribution())    
                
        return np.concatenate(distribution)
    
class PlayerFullState(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def get_hand_state(self):
        
        return self.__hand_state
        
    def __set_hand_state(self, table_state):
        
        self.__hand_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameState.build_by_player(player))        
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(HandState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameState.get_total_numbers()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameState.build_by_number(game_state_number))
        
        number = int(number / game_state_total_values) 
        
        table_state_total_values = TableFullState.get_total_numbers()
        table_state_number = number % table_state_total_values   
    
        self.__set_game_state(TableFullState.build_by_number(table_state_number))
        
        hand_state_number = int(number / table_state_total_values) 
        
        self.__set_table_state(HandState.build_by_number(hand_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableFullState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableFullState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandState.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerFullState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerFullState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerFullState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameState.get_total_numbers()
        base *= TableFullState.get_total_numbers()
        base *= HandState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameState.get_low_values(), TableFullState.get_low_values(), HandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameState.get_high_values(), TableFullState.get_high_values(), HandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameState.get_number_of_observations() + TableFullState.get_number_of_observations() + HandState.get_number_of_observations()
    
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameState.get_expected_distribution())
        distribution.append(TableFullState.get_expected_distribution())
        distribution.append(HandState.get_expected_distribution())  
                
        return np.concatenate(distribution)
    
  
class PlayerFullStateV2(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def get_hand_state(self):
        
        return self.__hand_state
        
    def __set_hand_state(self, table_state):
        
        self.__hand_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameState.build_by_player(player))        
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(HandStateWithoutZeroValue.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameState.get_total_values()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameState.build_by_number(game_state_number))
        
        number = int(number / game_state_total_values) 
        
        table_state_total_values = TableFullState.get_total_values()
        table_state_number = number % table_state_total_values   
    
        self.__set_game_state(TableFullState.build_by_number(table_state_number))
        
        hand_state_number = int(number / table_state_total_values) 
        
        self.__set_table_state(HandStateWithoutZeroValue.build_by_number(hand_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableFullState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableFullState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandStateWithoutZeroValue.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerFullStateV2()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerFullStateV2()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerFullStateV2()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameState.get_total_numbers()
        base *= TableFullState.get_total_numbers()
        base *= HandStateWithoutZeroValue.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameState.get_low_values(), TableFullState.get_low_values(), HandStateWithoutZeroValue.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameState.get_high_values(), TableFullState.get_high_values(), HandStateWithoutZeroValue.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameState.get_number_of_observations() + TableFullState.get_number_of_observations() + HandStateWithoutZeroValue.get_number_of_observations()
                
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameState.get_expected_distribution())
        distribution.append(TableFullState.get_expected_distribution())                
        distribution.append(HandStateWithoutZeroValue.get_expected_distribution())
                
        return np.concatenate(distribution)

