import numpy as np
import matplotlib.pyplot as plt
import highway_env
from typing import Union, Optional
import gym

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.callbacks import EvalCallback

class MetricsCallback(BaseCallback):
    def __init__(self,
                eval_env: Union[gym.Env, VecEnv],
                verbose=0):
        self.eval_env = eval_env
        super(MetricsCallback, self).__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("metrics/ego_speed", )
        plt.close()
        return True