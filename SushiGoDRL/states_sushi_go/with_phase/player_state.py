
import numpy as np


from states_sushi_go.state import State
from states_sushi_go.hand_state import HandState, HandStateWithoutZeroValue, CompleteHandState
from states_sushi_go.table_state import TableState, TableSimplState, TableFullState
from states_sushi_go.with_phase.game_state import GameWithPhaseState
      

    # GameWithPhaseState
    # TableSimplState
class PlayerSimplWithPhaseState(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
    
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableSimplState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameWithPhaseState.get_total_values()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameWithPhaseState.build_by_number(game_state_number))
        
        table_state_number = int(number / game_state_total_values) 
        
        self.__set_table_state(TableSimplState.build_by_number(table_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameWithPhaseState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableSimplState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        firstElementLen = GameWithPhaseState.get_number_of_observations()
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:firstElementLen]))
        self.__set_table_state(TableSimplState.build_by_observation(observation[firstElementLen:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerSimplWithPhaseState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerSimplWithPhaseState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerSimplWithPhaseState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameWithPhaseState.get_total_numbers()
        base *= TableSimplState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableSimplState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableSimplState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableSimplState.get_number_of_observations()
        
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableSimplState.get_expected_distribution())
                
        return np.concatenate(distribution)
    
    # GameWithPhaseState
    # HandState
    # TableSimplState
class PlayerSimpleWithPhaseStateV2(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
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
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableSimplState.build_by_player(player))
        self.__set_hand_state(HandState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameWithPhaseState.get_total_numbers()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameWithPhaseState.build_by_number(game_state_number))
        
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
        base *= GameWithPhaseState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableSimplState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableSimplState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandState.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerSimpleWithPhaseStateV2()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerSimpleWithPhaseStateV2()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerSimpleWithPhaseStateV2()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameWithPhaseState.get_total_numbers()
        base *= TableSimplState.get_total_numbers()
        base *= HandState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableSimplState.get_low_values(), HandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableSimplState.get_high_values(), HandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableSimplState.get_number_of_observations() + HandState.get_number_of_observations()
      
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableSimplState.get_expected_distribution())                
        distribution.append(HandState.get_expected_distribution())
                
        return np.concatenate(distribution)
    
    # GameWithPhaseState
    # HandState    
    # TableState
class PlayerMidWithPhaseState(State):    
    

    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
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
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableState.build_by_player(player))
        self.__set_hand_state(HandState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameWithPhaseState.get_total_numbers()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameWithPhaseState.build_by_number(game_state_number))
        
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
        base *= GameWithPhaseState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandState.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerMidWithPhaseState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerMidWithPhaseState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerMidWithPhaseState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameWithPhaseState.get_total_numbers()
        base *= TableState.get_total_numbers()
        base *= HandState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableState.get_low_values(), HandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableState.get_high_values(), HandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableState.get_number_of_observations() + HandState.get_number_of_observations()
            
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableState.get_expected_distribution())                
        distribution.append(HandState.get_expected_distribution())
                
        return np.concatenate(distribution)
    
    # GameWithPhaseState   
    # TableState
class PlayerMidWithPhaseStateV2(State):    
    
    def __init__(self):         

        self.__game_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
    def get_game_state(self):
        
        return self.__game_state
        
    def __set_game_state(self, game_state):
        
        self.__game_state = game_state   
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def __get_values_from_player(self, player):
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameWithPhaseState.get_total_values()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameWithPhaseState.build_by_number(game_state_number))
        
        table_state_number = int(number / game_state_total_values) 
        
        self.__set_table_state(TableState.build_by_number(table_state_number))


    def get_state_number(self):
        
        accumulated_value = 0
        base = 1
        
        accumulated_value += self.get_game_state().get_state_number() * base
        base *= GameWithPhaseState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableState.build_by_observation(observation[2:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerMidWithPhaseStateV2()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerMidWithPhaseStateV2()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerMidWithPhaseStateV2()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameWithPhaseState.get_total_numbers()
        base *= TableState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableState.get_number_of_observations()
                    
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableState.get_expected_distribution())    
                
        return np.concatenate(distribution)
    
    # GameWithPhaseState
    # HandState    
    # TableFullState
class PlayerFullWithPhaseState(State):    
    
    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
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
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(HandState.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameWithPhaseState.get_total_numbers()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameWithPhaseState.build_by_number(game_state_number))
        
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
        base *= GameWithPhaseState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableFullState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:2]))
        self.__set_table_state(TableFullState.build_by_observation(observation[2:9]))
        self.__set_hand_state(HandState.build_by_observation(observation[9:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerFullWithPhaseState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerFullWithPhaseState()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerFullWithPhaseState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameWithPhaseState.get_total_numbers()
        base *= TableFullState.get_total_numbers()
        base *= HandState.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableFullState.get_low_values(), HandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableFullState.get_high_values(), HandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableFullState.get_number_of_observations() + HandState.get_number_of_observations()
    
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableFullState.get_expected_distribution())
        distribution.append(HandState.get_expected_distribution())  
                
        return np.concatenate(distribution)  
    
    
    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
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
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(HandStateWithoutZeroValue.build_by_player(player))
                    
    def __get_values_from_number(self, number):
             
        game_state_total_values = GameWithPhaseState.get_total_values()
        game_state_number = number % game_state_total_values   
    
        self.__set_game_state(GameWithPhaseState.build_by_number(game_state_number))
        
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
        base *= GameWithPhaseState.get_total_numbers()
        
        accumulated_value += self.get_table_state().get_state_number() * base
        base *= TableFullState.get_total_numbers()    
        
        accumulated_value += self.get_hand_state().get_state_number() * base
        base *= HandState.get_total_numbers()    
        
        return accumulated_value
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:3]))
        self.__set_table_state(TableFullState.build_by_observation(observation[3:10]))
        self.__set_hand_state(HandStateWithoutZeroValue.build_by_observation(observation[10:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerFullWithPhaseStateV2()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_number(number):
        
        player_state = PlayerFullWithPhaseStateV2()        
        player_state.__get_values_from_number(number)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerFullWithPhaseStateV2()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_total_numbers():
        
        base = 1        
        base *= GameWithPhaseState.get_total_numbers()
        base *= TableFullState.get_total_numbers()
        base *= HandStateWithoutZeroValue.get_total_numbers()
        
        return base
                
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableFullState.get_low_values(), HandStateWithoutZeroValue.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableFullState.get_high_values(), HandStateWithoutZeroValue.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableFullState.get_number_of_observations() + HandStateWithoutZeroValue.get_number_of_observations()
                
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableFullState.get_expected_distribution())                
        distribution.append(HandStateWithoutZeroValue.get_expected_distribution())
                
        return np.concatenate(distribution)
    
    # GameWithPhaseState
    # CompleteHandState    
    # TableFullState
class PlayerCompleteState(State):    
    
    def __init__(self):         

        self.__game_state = None
        self.__hand_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
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
        
        self.__set_game_state(GameWithPhaseState.build_by_player(player))        
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(CompleteHandState.build_by_player(player))                    
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_game_state(GameWithPhaseState.build_by_observation(observation[0:3]))
        self.__set_table_state(TableFullState.build_by_observation(observation[3:10]))
        self.__set_hand_state(CompleteHandState.build_by_observation(observation[10:]))
        
    def get_as_observation(self):
        
        game_state_observation = self.get_game_state().get_as_observation()
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((game_state_observation, table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = PlayerCompleteState()        
        player_state.__get_values_from_player(player)
        
        return player_state
    
    def build_by_observation(observation):
        
        player_state = PlayerCompleteState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
        
    def get_low_values():
        
        return np.concatenate((GameWithPhaseState.get_low_values(), TableFullState.get_low_values(), CompleteHandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((GameWithPhaseState.get_high_values(), TableFullState.get_high_values(), CompleteHandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return GameWithPhaseState.get_number_of_observations() + TableFullState.get_number_of_observations() + CompleteHandState.get_number_of_observations()
                
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(GameWithPhaseState.get_expected_distribution())
        distribution.append(TableFullState.get_expected_distribution())                
        distribution.append(CompleteHandState.get_expected_distribution())
                
        return np.concatenate(distribution)

    # CompleteHandState    
    # TableFullState
class OtherPlayerCompleteState(State):    
    
    def __init__(self):         

        self.__hand_state = None
        self.__table_state = None
        
    def trained_with_chopsticks_phase():
        return True
        
    def get_table_state(self):
        
        return self.__table_state
        
    def __set_table_state(self, table_state):
        
        self.__table_state = table_state   
        
    def get_hand_state(self):
        
        return self.__hand_state
        
    def __set_hand_state(self, table_state):
        
        self.__hand_state = table_state   
        
    def __get_values_from_player(self, player):
              
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(CompleteHandState.build_by_player(player))      

    def __build_when_unknown_hand(self, player):
              
        self.__set_table_state(TableFullState.build_by_player(player))
        self.__set_hand_state(CompleteHandState.build_when_unknown_hand())                   
                            
    def __get_values_from_observation(self, observation):
        
        self.__set_table_state(TableFullState.build_by_observation(observation[0:7]))
        self.__set_hand_state(CompleteHandState.build_by_observation(observation[7:]))
        
    def get_as_observation(self):
        
        table_state_observation = self.get_table_state().get_as_observation()   
        hand_state_observation = self.get_hand_state().get_as_observation()   
        
        return np.concatenate((table_state_observation, hand_state_observation))       
    
    def build_by_player(player):
                
        player_state = OtherPlayerCompleteState()        
        player_state.__get_values_from_player(player)
        
        return player_state    
    
    def build_when_unknown_hand(player):
                
        player_state = OtherPlayerCompleteState()        
        player_state.__build_when_unknown_hand(player)
        
        return player_state
        
    def build_by_observation(observation):
        
        player_state = OtherPlayerCompleteState()        
        player_state.__get_values_from_observation(observation)
        
        return player_state
                
    def get_low_values():
        
        return np.concatenate((TableFullState.get_low_values(), CompleteHandState.get_low_values()))
    
    def get_high_values():
        
        return np.concatenate((TableFullState.get_high_values(), CompleteHandState.get_high_values()))    
    
    def get_number_of_observations():
    
        return TableFullState.get_number_of_observations() + CompleteHandState.get_number_of_observations()
    
    def get_expected_distribution():
        
        distribution = list()
        
        distribution.append(TableFullState.get_expected_distribution())
        distribution.append(CompleteHandState.get_expected_distribution())  
                
        return np.concatenate(distribution)  