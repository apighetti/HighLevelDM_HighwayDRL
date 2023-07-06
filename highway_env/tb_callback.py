import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.high_speed_reward = 0
        self.rml_reward = 0
        self.km_dense_reward = 0
        self.distance_to_victim_reward = 0
        self.dense_reward = 0
        
        self.collision_reward = 0
        self.km_sparse_reward = 0
        self.sparse_reward = 0
        
        self.cumulative_reward = 0
        
        self.num_episodes = 5
        self.episodes_count = 0
        self.episode_steps = 0
        self.episodes_lens = np.zeros(self.num_episodes)
    
    def _on_step(self) -> bool:
        self.episode_steps += 1
                
        self.high_speed_reward += self.training_env.get_attr('high_speed_reward')[0]
        self.rml_reward += self.training_env.get_attr('rml_reward')[0]
        self.km_dense_reward += self.training_env.get_attr('km_dense_reward')[0]
        self.distance_to_victim_reward += self.training_env.get_attr('distance_to_victim_reward')[0]
        self.dense_reward += self.training_env.get_attr('dense_reward')[0]
        
        self.collision_reward += self.training_env.get_attr('collision_reward')[0]
        self.km_sparse_reward += self.training_env.get_attr('km_sparse_reward')[0]
        self.sparse_reward += self.training_env.get_attr('sparse_reward')[0]
        
        self.cumulative_reward += self.training_env.get_attr('final_reward')[0]
                        
        if self.training_env.get_attr('terminal')[0]:
            self.episodes_lens[self.episodes_count] = self.episode_steps
            self.episodes_count += 1
            self.episode_steps = 0
            
        if self.episodes_count == self.num_episodes:
            self.episodes_count = 0
            
            self.logger.record('mean_reward/high_speed', self.high_speed_reward / self.num_episodes)
            self.logger.record('mean_reward/rml', self.rml_reward / self.num_episodes)
            self.logger.record('mean_reward/km_dense', self.km_dense_reward / self.num_episodes)
            self.logger.record('mean_reward/distance_to_victim', self.distance_to_victim_reward)
            self.logger.record('mean_reward/dense', self.dense_reward / self.num_episodes)
            
            self.logger.record('mean_reward/collision', self.collision_reward / self.num_episodes)
            self.logger.record('mean_reward/km_sparse', self.km_sparse_reward / self.num_episodes)
            self.logger.record('mean_reward/sparse', self.sparse_reward / self.num_episodes)
            
            self.logger.record('mean_reward/cumulative', self.cumulative_reward / self.num_episodes)
            
            self.logger.record('episode/mean_length', self.episodes_lens.sum() / self.num_episodes / self.training_env.get_attr('tot_duration')[0])

            self.high_speed_reward = 0
            self.rml_reward = 0
            self.km_dense_reward = 0
            self.distance_to_victim_reward = 0
            self.dense_reward = 0
            
            self.collision_reward = 0
            self.km_sparse_reward = 0
            self.sparse_reward = 0
            
            self.cumulative_reward = 0
            
        return True