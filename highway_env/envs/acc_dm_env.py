import numpy as np
import random
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class ACCDecisionMakingEnv(AbstractEnv):
    """
        ACC action test 1-lane, 1-vehicle environment
    """

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
            "lanes_count": 5,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": 2,
            "npc_initial_lane_id": 0,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config
    
    def _reset(self) -> None:
        w = self.vehicles_distribution()
        self._create_road()
        self._create_vehicles(w)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def vehicles_distribution(self):
        '''Create array of weights that will be used to spawn vehicles.'''
        
        n = self.config['lanes_count']
        lanes_list = range(0,n)
        weights = [None]*n
        
        for i in lanes_list: 
            weights[i] = (lanes_list[i])*(10**i)+1
            
        return weights
    
    def get_npc_speed(self, aux, sequence):
        '''Compute speed of a spawned vehicle according to its position.'''
        
        if aux == 0:
            aux = 1
            
        speed = (25 + 3*sequence[-aux])
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
                aux = random.choices(range(0,self.config['lanes_count']), weights = vehicle_distribution, k=1)[0]
                # vehicle = other_vehicles_type.create_random(self.road, lane_id=self.config["npc_initial_lane_id"], spacing=1 / self.config["vehicles_density"])
                vehicle = other_vehicles_type.create_random(self.road, speed = self.get_npc_speed(aux,range(0,self.config['lanes_count'])),\
                    lane_id = aux, spacing=1 / self.config["vehicles_density"]) #edit NPC
                vehicle.randomize_behavior()
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
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)


    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

register(
    id='acc-dm-v0',
    entry_point='highway_env.envs:ACCDecisionMakingEnv',
)