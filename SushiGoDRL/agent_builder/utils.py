# -*- coding: utf-8 -*-


import gym
import pickle
            
def save_batches(batches, filename):
    
    f = open(filename + ".txt", "a")
    for batch_info in batches:
        f.write(str(batch_info.times_exploration) + "\t\t")
        f.write(str(batch_info.times_explotation) + "\t\t")
        f.write(str(batch_info.total_reward) + "\t\t")
        f.write(str(batch_info.epsilon_at_end) + "\t\n")
    f.close()

class BatchInfo(object):
    
    def __init__(self):
        self.times_exploration = 0
        self.times_explotation = 0
        self.total_reward = 0
        self.n = 0   
        self.epsilon_at_end = 0
        
