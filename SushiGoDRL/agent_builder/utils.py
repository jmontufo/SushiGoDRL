# -*- coding: utf-8 -*-


import gym
import pickle
            
def save_batches(batches, filename):
    
    f = open(filename + ".txt", "a")
    
    batch_with_max_score = 0
    batch_with_max_wins = 0
    max_score_batch = batches[0]
    max_wins_batch = batches[0]
    
    current_batch = 0
    
    for batch_info in batches:
        f.write(str(batch_info.times_exploration) + "\t\t")
        f.write(str(batch_info.times_explotation) + "\t\t")
        f.write(str(batch_info.total_reward) + "\t\t")
        f.write(str(batch_info.points) + "\t\t")
        f.write(str(batch_info.points_by_victory) + "\t\t")
        f.write(str(batch_info.epsilon_at_end) + "\t\n")
        
        if batch_info.points > max_score_batch.points:
            max_score_batch = batch_info
            batch_with_max_score = current_batch
            
        if batch_info.points_by_victory > max_wins_batch.points_by_victory:
            max_wins_batch = batch_info
            batch_with_max_wins = current_batch
        
        current_batch += 1
        
    f.write ("Batch with max score: " + str(batch_with_max_score) + "\t\t")
    f.write (str(max_score_batch.points) + "\t\t")
    f.write (str(max_score_batch.points_by_victory) + "\t\n")
    f.write ("Batch with max wins: " + str(batch_with_max_wins) + "\t\t")
    f.write (str(max_wins_batch.points) + "\t\t")
    f.write (str(max_wins_batch.points_by_victory) + "\t\n")
    f.close()

class BatchInfo(object):
    
    def __init__(self):
        self.times_exploration = 0
        self.times_explotation = 0
        self.total_reward = 0
        self.points = 0
        self.points_by_victory = 0
        self.n = 0   
        self.epsilon_at_end = 0
        self.score = 0
        
