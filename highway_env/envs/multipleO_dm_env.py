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
from highway_env.vehicle.objects import LaneIndex

SPACINGS = [1, 2, 3]
NUM_NPCS = np.arange(1, 11)

class MultipleOvertakeDecisionMakingEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
        
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        # Reward initialization
        self.high_speed_reward = 0
        self.rml_reward = 0
        self.km_dense_reward = 0
        self.dense_reward = 0
        
        self.collision_reward = 0
        self.km_sparse_reward = 0
        self.sparse_reward = 0
        
        self.final_reward = 0
        
        self.tot_duration = self.config['duration']
        self.total_speed = 0
        self.km_travelled = 0
        self.terminal = False
        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DecisionMakingAction",
            },
            "lanes_count": 2,
            "controlled_vehicles": 1,
            "duration": 120,  # [s*2]
            "initial_lane_id": None,
            "ego_spacing": 2,
            "vehicles_density": 0.5,
            "offroad_terminal": False,
            "enable_npc_lane_change": False,
            
            "collision_reward": -10,
            "km_sparse_reward": 10
        })
        return config

    def _reset(self) -> None:
        w = [0,1]
        self._create_road()
        self._create_vehicles(w)
        
    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=36),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def vehicles_distribution(self):
        '''Create array of weights that will be used to spawn vehicles.'''

        n = self.config['lanes_count']
        lanes_list = range(0, n)
        weights = [None]*n

        for i in lanes_list:
            weights[i] = (lanes_list[i])*(10**i)+1

        return weights

    def get_npc_speed(self, aux):
        '''Compute speed of a spawned vehicle according to its position.'''

        speed = utils.lmap(aux, [0, self.config['lanes_count']-1], [36, 20])
        return speed

    def _create_vehicles(self, vehicle_distribution) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        npcs_num = random.choice(NUM_NPCS)
        sp = random.choice(SPACINGS)

        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            npcs_num, num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for i in range(others):
                aux = random.choices(range(0, self.config['lanes_count']), weights=vehicle_distribution, k=1)[0]
                vehicle = other_vehicles_type.create_random(self.road, speed=self.get_npc_speed(aux),
                                                            lane_id=1, spacing=sp / self.config["vehicles_density"])  # edit NPC
                vehicle.randomize_behavior()
                vehicle.enable_lane_change = self.config['enable_npc_lane_change']
                self.road.vehicles.append(vehicle)
        
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # Reset rewards
        self.high_speed_reward = 0
        self.rml_reward = 0
        self.km_dense_reward = 0
        self.dense_reward = 0
        
        self.collision_reward = 0
        self.km_sparse_reward = 0
        self.sparse_reward = 0
        
        self.final_reward = 0
        
        self.terminal = False
        
        self.total_speed += self.vehicle.speed
        self.km_travelled = utils.lmap(round(self.total_speed,3), [0,36*self.tot_duration], [0,1])

        if self._is_terminal():
            self.terminal = True
            self.collision_reward = self.config["collision_reward"] * self.vehicle.crashed
            self.km_sparse_reward = self.config["km_sparse_reward"] * self.km_travelled if not self.vehicle.crashed else 0
                
            self.sparse_reward = \
                   + self.km_sparse_reward \
                   + self.collision_reward
            
            self.final_reward = self.sparse_reward
            
            self.final_reward = utils.lmap(self.final_reward,
                                           [self.config["collision_reward"], self.config["km_sparse_reward"]],
                                           [0,1])
            
            # Reset counters
            self.total_speed = 0
            self.km_travelled = 0
        else:
            self.final_reward = 0
            
        self.final_reward = 0 if not self.vehicle.on_road else self.final_reward
        return self.final_reward
        
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
    id='multipleO-dm-env-v0',
    entry_point='highway_env.envs:MultipleOvertakeDecisionMakingEnv',
)
