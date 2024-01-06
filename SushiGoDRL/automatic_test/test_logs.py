# -*- coding: utf-8 -*-

import pickle


class AgentTypeLog(object):
    
    def __init__(self):   
        
        self.num_of_games = 0
        
        self.victories_by_player = [0, 0] 
        self.points_by_player = [0,0]    
        
        self.action_used = [0]*9
        self.action_n = [0]*9
        
        self.action_used_by_turn = []
        self.action_n_by_turn = []
        
        for i in range(0,20):
            self.action_used_by_turn.append([0]*9)
            self.action_n_by_turn.append([0]*9)
        


class AgentTypeLogSet(object):
    
    def __init__(self):
        
       self.log_by_agent = dict()
       
    def getLogForAgent(self, agentType):
        
        if agentType not in self.log_by_agent:
            self.log_by_agent[agentType] = AgentTypeLog()
            
        return self.log_by_agent[agentType]
    
    def load(filename):
                
        atl_input = open(filename + ".pkl", 'rb')
        agentTypeLogSet = pickle.load(atl_input)
        atl_input.close()
        
        return agentTypeLogSet
    
    def save(self, filename):
        
        output = open(filename + ".pkl", "wb")
        pickle.dump(self, output)
        output.close()

        

