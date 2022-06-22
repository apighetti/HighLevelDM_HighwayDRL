import pandas as pd
import numpy as np

class PrintMetrics():
    def __init__(self, collision_num = 0,
                 all_km_travelled = [],
                 all_decision_changes = [],
                 all_decision_change_rates = [],
                 all_left_lane_changes = [],
                 all_left_lane_change_rates = [],
                 all_right_lane_changes = [],
                 all_right_lane_change_rates = [],
                 all_episode_durations = [],
                 mean_speeds = [],
                 mean_accelerations = [],
                 mean_decelerations = []
                 ) -> None:
                
                 self.collisions_num = collision_num
                 self.all_km_traveled = all_km_travelled
                 self.all_decision_changes = all_decision_changes
                 self.all_decision_change_rates = all_decision_change_rates
                 self.all_left_lane_changes = all_left_lane_changes
                 self.all_left_lane_change_rates = all_left_lane_change_rates
                 self.all_right_lane_changes = all_right_lane_changes
                 self.all_right_lane_change_rates = all_right_lane_change_rates
                 self.all_episode_durations = all_episode_durations
                 self.mean_speeds = mean_speeds
                 self.mean_accelerations = mean_accelerations
                 self.mean_decelerations = mean_decelerations



                 header_ep = ['ep_duration', 'has_collided', 'km_travelled', 'mean_speed', 'mean_accl', 'mean_decel', 'decision_changes', 'left_lane_changes', 'right_lane_changes']

                 self.episode_df = pd.DataFrame(columns=header_ep)
                 self.episode_df.index.name = 'episode'

                 header_recap = ['mean_episode_duration', 'collisions', 'mean_km_travelled', 'overall_mean_speed', 'overall_mean_acceleration', 'overall_mean_deceleration', 'mean_decision_changes', 'mean_left_lane_changes', 'mean_right_lane_changes']
                 self.recap_df = pd.DataFrame(columns=header_recap)




        
    def printEpisode(self, km_travelled, decision_change_num, decision_change_rate,\
        left_lane_change_num, left_lane_change_rate, right_lane_change_num, right_lane_change_rate,\
        mean_speed, mean_acceleration, mean_deceleration, collision, episode_duration, curr_episode_num) -> None:

        self.all_episode_durations.append(episode_duration)
        self.all_km_traveled.append(km_travelled)
        self.all_decision_changes.append(decision_change_num)
        self.all_decision_change_rates.append(decision_change_rate)
        self.all_left_lane_changes.append(left_lane_change_num)
        self.all_left_lane_change_rates.append(left_lane_change_rate)
        self.all_right_lane_changes.append(right_lane_change_num)
        self.all_right_lane_change_rates.append(right_lane_change_rate)
        self.mean_speeds.append(mean_speed)
        self.mean_accelerations.append(mean_acceleration)
        self.mean_decelerations.append(mean_deceleration)
        self.collisions_num += collision

        series = pd.Series([round(episode_duration), "YES" if collision else "NO", round(km_travelled,2), round(mean_speed,2), round(mean_acceleration,3), round(mean_deceleration,3), decision_change_num, left_lane_change_num, right_lane_change_num], name=curr_episode_num, index=self.episode_df.columns)

        self.episode_df = self.episode_df.append(series)

        
        print(f"\nepisode {curr_episode_num} ended, metrics:\
             \n\tepisode duration: {round(episode_duration)} seconds,\n\tcollision? {'YES' if collision else 'NO'} \n\tkm travelled: {round(km_travelled,2)},\
             \n\tdecision changes: {decision_change_num}, \n\tmean speed: {round(mean_speed,2)} km/h, \n\tmean acceleration: {round(mean_acceleration,3)} m/s, \n\tmean deceleration: {round(mean_deceleration,3)} m/s,\
             \n\tdecision change rate: {round(decision_change_rate, 2)}, \n\tleft lane changes: {left_lane_change_num}, \n\tleft lane change rate: {round(left_lane_change_rate, 2)}\
             \n\tright lane changes: {right_lane_change_num}, \n\tright lane change rate: {round(right_lane_change_rate, 2)}")


    def printRecap(self, episode_num, path) -> None:

        mean_km_travelled, total_km_travelled = np.mean(self.all_km_traveled), np.sum(self.all_km_traveled)
        mean_decision_changes, total_decision_change_num = np.mean(self.all_decision_changes), np.sum(self.all_decision_changes)
        mean_decision_change_rate = np.mean(self.all_decision_change_rates)
        mean_left_lane_changes, total_left_lane_changes = np.mean(self.all_left_lane_changes), np.sum(self.all_left_lane_changes)
        mean_left_lane_change_rate = np.mean(self.all_left_lane_change_rates)
        mean_right_lane_changes, total_right_lane_changes = np.mean(self.all_right_lane_changes), np.sum(self.all_right_lane_changes)
        mean_right_lane_change_rate = np.mean(self.all_right_lane_change_rates)
        mean_episode_duration = np.mean(self.all_episode_durations)
        overall_mean_speed = np.mean(self.mean_speeds)
        overall_mean_acceleration = np.mean(self.mean_accelerations)
        overall_mean_deceleration = np.mean(self.mean_decelerations)

        series = pd.Series([round(mean_episode_duration), self.collisions_num, round(mean_km_travelled,2), round(overall_mean_speed,2), round(overall_mean_acceleration,3), round(overall_mean_deceleration,3), mean_decision_changes, mean_left_lane_changes, mean_right_lane_changes], index=self.recap_df.columns)

        self.recap_df = self.recap_df.append(series, ignore_index=True)

        self.episode_df.to_csv(path + "/episode_data.csv")
        self.recap_df.to_csv(path + "/recap_data.csv", index=False)

        print(f"\nevaluation ended, metrics:\
            \nmeans: \n\tmean episode duration: {round(mean_episode_duration)} seconds, \n\tmean km travelled: {round(mean_km_travelled,2)},\
            \n\toverall mean speed: {round(overall_mean_speed,2)} km/h, \n\toverall mean acceleration: {round(overall_mean_acceleration,3)} m/s, \n\toverall mean deceleration: {round(overall_mean_deceleration,3)} m/s\
            \n\tmean decision changes: {round(mean_decision_changes)}, \n\tmean decision change rate: {round(mean_decision_change_rate,2)},\
            \n\tmean left lane changes: {round(mean_left_lane_changes)}, \n\tmean left lane change rate: {round(mean_left_lane_change_rate,2)},\
            \n\tmean right lane changes: {round(mean_right_lane_changes)}, \n\tmean right lane change rate: {round(mean_right_lane_change_rate,2)}, \
            \ntotal:\n\ttotal km travelled: {round(total_km_travelled,2)}, \n\ttotal decision changes: {total_decision_change_num},\
            \n\ttotal left lane changes: {total_left_lane_changes}, \n\ttotal right lane changes: {total_right_lane_changes}, \
            \n\ttotal collision num: {self.collisions_num} out of {episode_num} episodes")