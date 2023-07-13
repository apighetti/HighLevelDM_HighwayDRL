from distutils.command.config import config
import random
from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

COL_REWARDS = [-.1, -1, -3, -5]

class OVTKDecisionMakingEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    
    LAST_ACTION = ""
    LAST_LANE_IDX = 1000

    DECISION_CHANGE = 0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 7
            },
            "action": {
                "type": "DecisionMakingAction",
            },
            "other_vehicles_type": "highway_env.vehicle.behavior.HazardousVehicle",
            "lanes_count": 2,
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "duration": 60,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 0.7,
            "not_in_right_lane_reward": -0.45,  # The reward received when driving on the right-most lanes, linearly mapped to
            "decision_change": -0.4,
            "high_speed_reward": 0.4,        # The reward received when driving at full speed, linearly mapped to zero for
                                                 # lower speeds according to config["reward_speed_range"].
            "reward_speed_range": [30, 36],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        w = self.vehicles_distribution()
        self._create_road()
        self._create_vehicles(w)
        
    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=36),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def vehicles_distribution(self):
        '''Create array of weights that will be used to spawn vehicles.'''
        
        n = self.config['lanes_count']
        lanes_list = range(0,n)
        weights = [None]*n
        
        for i in lanes_list: 
            weights[i] = (lanes_list[i])*(10**i)+1
            
        return weights
    
    def get_npc_speed(self, aux):
        '''Compute speed of a spawned vehicle according to its position.'''
        
        speed = utils.lmap(aux, [0, self.config['lanes_count']], [36, 20])
        return speed

    def _create_vehicles(self, vehicle_distribution) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for i in range(others):
                vehicle = other_vehicles_type.create_random(self.road, speed = 45,\
                    lane_id = self.config["initial_lane_id"], spacing=1 / self.config["vehicles_density"]) #edit NPC
                vehicle.position = [60.0, 4.]
                self.road.vehicles.append(vehicle)


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        
        self.DECISION_CHANGE = 0
        if self.LAST_ACTION != self.vehicle.current_action:
            if self.LAST_ACTION != "":
                self.DECISION_CHANGE = 1
            self.LAST_ACTION = self.vehicle.current_action
        
        collision_index = int(utils.lmap(abs(self.steps - self.config['duration']), [0,self.config['duration']], [3,0]))

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
 
        reward = COL_REWARDS[collision_index] * self.vehicle.crashed \
            + self.config["not_in_right_lane_reward"] * (1 - (lane / max(len(neighbours) - 1, 1))) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["decision_change"] * self.DECISION_CHANGE
            
        reward = utils.lmap(reward,
                          [self.config["not_in_right_lane_reward"] + self.config["decision_change"],
                           self.config["high_speed_reward"]],
                          [0, 1])
        reward += COL_REWARDS[collision_index] * self.vehicle.crashed
        reward = 0 if not self.vehicle.on_road else reward
        return reward
    
    def random_action(self):
        actions = [self.action_type.actions_indexes['ACC'], self.action_type.actions_indexes['OVERTAKE'], self.action_type.actions_indexes['RIGHTMOSTLANE']]
        random_action = random.choice(actions)
        return random_action


    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

register(
    id='overtaken-dm-env-v0',
    entry_point='highway_env.envs:OVTKDecisionMakingEnv',
)