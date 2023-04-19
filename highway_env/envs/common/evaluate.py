import pandas as pd
import numpy as np


class PrintMetrics():
    def __init__(self, collision_num=0,
                 env_names=[]
                 ) -> None:
        self.env_names = env_names
        self.collisions_num = collision_num

        header_ep = ['environment', 'ep_duration', 'has_collided', 'km_travelled', 'mean_speed',
                     'mean_accl', 'mean_decel', 'decision_changes', 'left_lane_changes', 'right_lane_changes']
        self.episode_df = pd.DataFrame(columns=header_ep)
        self.episode_df.index.name = 'episode'

    def printEpisode(self, env_name, km_travelled, decision_change_num, decision_change_rate,
                     left_lane_change_num, left_lane_change_rate, right_lane_change_num, right_lane_change_rate,
                     mean_speed, mean_acceleration, mean_deceleration, collision, episode_duration, curr_episode_num) -> None:

        self.env_names.append(env_name)
        
        series = pd.Series([env_name, round(episode_duration), 1 if collision else 0, round(km_travelled, 2), round(mean_speed, 2), round(mean_acceleration, 3), round(
            mean_deceleration, 3), decision_change_num, left_lane_change_num, right_lane_change_num], name=curr_episode_num, index=self.episode_df.columns)

        self.episode_df = self.episode_df.append(series)
        
        # self.episode_df = pd.concat([self.episode_df, pd.DataFrame.from_records(series)])

        # print(f"\nepisode {curr_episode_num}: {env_name} ended, metrics:\
        #      \n\tepisode duration: {round(episode_duration)} seconds,\n\tcollision? {'YES' if collision else 'NO'} \n\tkm travelled: {round(km_travelled,2)},\
        #      \n\tdecision changes: {decision_change_num}, \n\tmean speed: {round(mean_speed,2)} km/h, \n\tmean acceleration: {round(mean_acceleration,3)} m/s, \n\tmean deceleration: {round(mean_deceleration,3)} m/s,\
        #      \n\tdecision change rate: {round(decision_change_rate, 2)}, \n\tleft lane changes: {left_lane_change_num}, \n\tleft lane change rate: {round(left_lane_change_rate, 2)}\
        #      \n\tright lane changes: {right_lane_change_num}, \n\tright lane change rate: {round(right_lane_change_rate, 2)}")

    def printRecap(self, path, csv_id) -> None:

        recap_df = self.episode_df.groupby('environment').agg(episode_num=('environment', np.count_nonzero),
                                                              mean_episode_duration=(
                                                                  'ep_duration', lambda x: np.round(np.mean(x))),
                                                              collisions=(
                                                                  'has_collided', 'sum'),
                                                              mean_km_travelled=(
                                                                  'km_travelled', lambda x: np.round(np.mean(x), 2)),
                                                              overall_mean_speed=(
                                                                  'mean_speed', lambda x: np.round(np.mean(x))),
                                                              overall_mean_acceleration=(
                                                                  'mean_accl', lambda x: np.round(np.mean(x), 3)),
                                                              overall_mean_deceleration=(
                                                                  'mean_decel', lambda x: np.round(np.mean(x), 3)),
                                                              mean_decision_changes=(
                                                                  'decision_changes', lambda x: np.round(np.mean(x))),
                                                              mean_left_lane_changes=(
                                                                  'left_lane_changes', lambda x: np.round(np.mean(x))),
                                                              mean_right_lane_changes=(
                                                                  'right_lane_changes', lambda x: np.round(np.mean(x)))
                                                              ).round(2)

        self.episode_df.to_csv(path + f"/episode_data_{csv_id}.csv")
        recap_df.to_csv(path + f"/recap_data_{csv_id}.csv")
