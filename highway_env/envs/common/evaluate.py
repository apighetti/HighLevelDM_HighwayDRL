import pandas as pd
import numpy as np


class PrintMetrics():
    def __init__(self, collision_num = 0,
                 all_km_travelled = [],
                 all_decision_change_nums = [],
                 all_decision_change_rates = [],
                 all_episode_durations = []
                 ) -> None:
                
                 self.collisions_num = collision_num
                 self.all_km_traveled = all_km_travelled
                 self.all_decision_change_nums = all_decision_change_nums
                 self.all_decision_change_rates = all_decision_change_rates
                 self.all_episode_durations = all_episode_durations
        
    def printEpisode(self, km_travelled, decision_change_num, decision_change_rate, collision, episode_duration, curr_episode_num) -> None:
        self.all_km_traveled.append(km_travelled)
        self.all_decision_change_nums.append(decision_change_num)
        self.all_decision_change_rates.append(decision_change_rate)
        self.all_episode_durations.append(episode_duration)
        self.collisions_num += collision
        
        print(f"\nepisode {curr_episode_num} ended, metrics: \n\tepisode duration: {round(episode_duration)} seconds, \n\tkm travelled: {round(km_travelled,2)},\
             \n\tdecision changes: {decision_change_num} \n\tdecision change rate: {round(decision_change_rate, 2)}")


    def printRecap(self, episode_num) -> None:
        mean_km_travelled, total_km_travelled = np.mean(self.all_km_traveled), np.sum(self.all_km_traveled)
        mean_decision_change_num, total_decision_change_num = np.mean(self.all_decision_change_nums), np.sum(self.all_decision_change_nums)
        mean_decision_change_rate = np.mean(self.all_decision_change_rates)
        mean_episode_duration = np.mean(self.all_episode_durations)

        print(f"\nevaluation ended, metrics:\
            \nmeans: \n\tmean episode duration: {round(mean_episode_duration)} seconds, \n\tmean km travelled: {round(mean_km_travelled,2)}, \n\tmean decision changes: {round(mean_decision_change_num)}, \n\tmean decision change rate: {round(mean_decision_change_rate,2)},\
            \ntotal:\n\ttotal km travelled: {round(total_km_travelled,2)}, \n\ttotal decision changes: {total_decision_change_num},\
            \n\ttotal collision num: {self.collisions_num}, \n\ttotal episode num: {episode_num}")