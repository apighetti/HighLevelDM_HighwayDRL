from distutils.command.config import config
from xmlrpc.client import Boolean
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import LaneIndex

# START_SEC = 120
COL_REWARDS = [-0.1, -0.5, -1, -2.5] # ordini di grandezza differenti

class DecisionMakingEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    LAST_STEPS = 1
    TOTAL_SPACE = 0
    LAST_VEHICLE_SPEED = 0
    # LAST_ACTION = ""

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
            "lanes_count": 3,
            "vehicles_count": 35, # curriculum learning su lanes e npc-vehicles
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 120,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 0.7,
            # "collision_reward": -3,            // COLLISION IS DIVIDED # The reward received when colliding with a vehicle.
            "not_in_right_lane_reward": -0.004,  # The reward received when driving on the right-most lanes, linearly mapped to
                                                 # zero for other lanes.
            "distance_to_tv_reward": -0.07,      # -0.015 // non basta come incentivo alla velocità
            # "decision_change_reward": -0.25,   // NOT IMPLEMENTED YET
            "distance_reward": 0.05,
            # "high_speed_reward": 0.001,        # The reward received when driving at full speed, linearly mapped to zero for
                                                 # lower speeds according to config["reward_speed_range"].
            # "lane_change_reward": -0.005,      # The reward received at each lane change action.
            # "reward_speed_range": [30, 36],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        # f = open(r'/Users/fornerispighetti/HighwayDRL/highway_env/ACC_data.csv', 'a')
        # f.write("ego_speed,front_vehicle_speed,throttle,distance,gap,counter" + "\n")
        # f.close()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=20,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )

            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)


    # def _is_lane_empty(self, lane_index, right = True) -> bool:
    #     if (right):
    #         right_lane_index = (lane_index[0], lane_index[1], lane_index[2]+1)
    #         front_right_vehicle, rear_right_vehicle = self.road.neighbour_vehicles(self.vehicle, right_lane_index)
            
    #         if rear_right_vehicle and not front_right_vehicle:
    #             rear_gap = self.vehicle.time_gap_error(2, rear_right_vehicle, self.vehicle)
    #             if rear_gap > 0:
    #                 # print("only Rear Right Vehicle: " + str(rear_right_vehicle)+"\n")
    #                 return True
    #         elif front_right_vehicle and not rear_right_vehicle:
    #             front_gap = self.vehicle.time_gap_error(2, self.vehicle, front_right_vehicle)
    #             if front_gap > 0:
    #                 # print("only Front Right Vehicle: " + str(front_right_vehicle)+"\n")
    #                 return True
            
    #         elif front_right_vehicle and rear_right_vehicle:
    #             rear_gap = self.vehicle.time_gap_error(2, rear_right_vehicle, self.vehicle)  
    #             front_gap = self.vehicle.time_gap_error(2, self.vehicle, front_right_vehicle)
    #             if front_gap > 0 and rear_gap > 0:
    #                 # print("Front Right Vehicle: " + str(front_right_vehicle)+"\n")
    #                 # print("Rear Right Vehicle: " + str(rear_right_vehicle)+"\n")
    #                 return True
    #     else:
    #         left_lane_index = (lane_index[0], lane_index[1], lane_index[2]-1)
    #         front_left_vehicle, rear_left_vehicle = self.road.neighbour_vehicles(self.vehicle, left_lane_index)

    #         if rear_left_vehicle and not front_left_vehicle:
    #             rear_gap = self.vehicle.time_gap_error(2, rear_left_vehicle, self.vehicle)
    #             if rear_gap > 0:
    #                 # print("only Rear Left Vehicle: " + str(rear_left_vehicle)+"\n")
    #                 return True  
            
    #         elif front_left_vehicle and not rear_left_vehicle:
    #             front_gap = self.vehicle.time_gap_error(2, self.vehicle, front_left_vehicle)
    #             if front_gap > 0:
    #                 # print("only Rear Left Vehicle: " + str(rear_left_vehicle)+"\n")
    #                 return True
            
    #         elif front_left_vehicle and rear_left_vehicle:
    #             rear_gap = self.vehicle.time_gap_error(2, rear_left_vehicle, self.vehicle)  
    #             front_gap = self.vehicle.time_gap_error(2, self.vehicle, front_left_vehicle)
    #             if front_gap > 0 and rear_gap > 0:
    #                 # print("Rear Left Vehicle: " + str(rear_left_vehicle)+"\n")
    #                 # print("Front Left Vehicle: " + str(front_left_vehicle)+"\n")
    #                 return True
    #     # print("no negative reward")
    #     return False



    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        

        # lanes_count = len(self.road.network.lanes_list())

        # if(self.vehicle.lane_index[2] != lanes_count-1):
        #     not_in_rl = 1 if self._is_lane_empty(self.vehicle.lane_index) \
        #                 and self.vehicle.lane_index[2] + 1 != self.vehicle.target_lane_index[2] else 0
        # else:
        #     not_in_rl = 0

        speed_diff = utils.lmap((36 - self.vehicle.speed), [0,36] , [0,1])

        # duration_diff = utils.lmap((self.config['duration'] - self.steps), [self.config['duration'],0], [0,1])
        self.TOTAL_SPACE += abs(self.vehicle.speed*(self.steps - self.LAST_STEPS))
        self.LAST_VEHICLE_SPEED = self.vehicle.speed
        self.LAST_STEPS = self.steps
        # print(round(self.TOTAL_SPACE,3))

        km_travelled = utils.lmap(round(self.TOTAL_SPACE,3), [0,36*self.config['duration']], [0,1])


        # if self.LAST_ACTION != self.vehicle.current_action:

        # self.LAST_ACTION = self.vehicle.current_action
        
        # print(f"\ndistance to td reward {self.config['distance_reward'] * km_travelled}")

        collision_index = int(utils.lmap(abs(self.steps - self.config['duration']), [0,self.config['duration']], [3,0]))

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        reward = COL_REWARDS[collision_index] * self.vehicle.crashed \
            + self.config["distance_to_tv_reward"] * speed_diff \
            + self.config["not_in_right_lane_reward"] * (1 - (lane / max(len(neighbours) - 1, 1))) \
            + self.config["distance_reward"] * km_travelled
          # + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
 
        # reward = utils.lmap(reward,
        #                   [self.config["distance_to_tv_reward"],
        #                    self.config["not_in_right_lane_reward"]],
        #                   [0, 1])

        reward = 0 if not self.vehicle.on_road else reward
        # print(f"\nreward: {reward}, \ndense rewards:\n\ttarget velocity reward: {self.config['distance_to_tv_reward'] * speed_diff},\n\tnot in RL reward:{self.config['not_in_right_lane_reward'] * (1 - (lane / max(len(neighbours) - 1, 1)))},\n\tduration reward: {self.config['distance_reward'] * km_travelled} \
        #     \nsparse rewards:\n\tcollision reward: {COL_REWARDS[collision_index]}")
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        self.LAST_STEPS = 1
        self.TOTAL_SPACE = 0
        self.LAST_VEHICLE_SPEED = 0
        # self.LAST_ACTION = ""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

register(
    id='dm-env-v0',
    entry_point='highway_env.envs:DecisionMakingEnv',
)