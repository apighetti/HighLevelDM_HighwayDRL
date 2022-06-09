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

# START_SEC = 120
# COL_REWARDS = [-0.5, -1, -3, -5] # ordini di grandezza differenti
# COL_REWARDS = [-3, -2.5, -2, -1.5] # ZZ try

class OVTKDecisionMakingEnv(AbstractEnv):
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
            "other_vehicles_type": "highway_env.vehicle.behavior.HazardousVehicle",
            "lanes_count": 2,
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "duration": 60,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 0.7,
            "collision_reward": -0.5,            # The reward received when colliding with a vehicle.
            "not_in_right_lane_reward": -0.45,  # The reward received when driving on the right-most lanes, linearly mapped to
            #                                      # zero for other lanes.
            # "distance_to_tv_reward": -0.4,      # -0.015 // non basta come incentivo alla velocità
            # "decision_change_reward": -0.25,   // NOT IMPLEMENTED YET
            # "distance_reward": 0.08,
            "high_speed_reward": 0.4,        # The reward received when driving at full speed, linearly mapped to zero for
                                                 # lower speeds according to config["reward_speed_range"].
            # "lane_change_reward": -0.005,      # The reward received at each lane change action.
            "reward_speed_range": [30, 36],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        w = self.vehicles_distribution()
        self._create_road()
        self._create_vehicles(w)
        # f = open(r'C:\Users\luka-\Desktop\ACC_data.csv', 'a')
        # f.write("ego_speed,front_vehicle_speed,throttle,distance,gap,counter" + "\n")
        # f.close()

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
                aux = random.choices(range(0,self.config['lanes_count']), weights = vehicle_distribution, k=1)[0]
                # vehicle = other_vehicles_type.create_random(self.road, lane_id=self.config["npc_initial_lane_id"], spacing=1 / self.config["vehicles_density"]) // self.get_npc_speed(aux))
                vehicle = other_vehicles_type.create_random(self.road, speed = 45,\
                    lane_id = self.config["initial_lane_id"], spacing=1 / self.config["vehicles_density"]) #edit NPC
                vehicle.position = [60.0, 4.]
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

        # speed_diff = utils.lmap((36 - self.vehicle.speed), [0,36] , [0,1])

        # duration_diff = utils.lmap((self.config['duration'] - self.steps), [self.config['duration'],0], [0,1])
        # self.TOTAL_SPACE += abs(self.vehicle.speed*(self.steps - self.LAST_STEPS))
        # self.LAST_VEHICLE_SPEED = self.vehicle.speed
        # self.LAST_STEPS = self.steps
        # # print(round(self.TOTAL_SPACE,3))

        # km_travelled = utils.lmap(round(self.TOTAL_SPACE,3), [0,36*self.config['duration']], [0,1])
        # print(f'km travelled: {km_travelled}')


        # if self.LAST_ACTION != self.vehicle.current_action:

        # self.LAST_ACTION = self.vehicle.current_action
        
        # print(f"\ndistance to td reward {self.config['distance_reward'] * km_travelled}")

        # collision_index = int(utils.lmap(abs(self.steps - self.config['duration']), [0,self.config['duration']], [3,0]))
        # print(collision_index)

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
 
        # print(f'dist rew: {self.config["distance_reward"] * km_travelled}')
        # print(f'nrl rew: {self.config["not_in_right_lane_reward"] * (1 - (lane / max(len(neighbours) - 1, 1)))} driving in lane: {lane}')
        # print(f'dist to tv rew: {self.config["distance_to_tv_reward"] * speed_diff} driving at {self.vehicle.speed}')
        
        # COL_REWARDS[collision_index]

        reward = self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["not_in_right_lane_reward"] * (1 - (lane / max(len(neighbours) - 1, 1))) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
            

            # + self.config["distance_to_tv_reward"] * speed_diff \
            # + self.config["distance_reward"] * km_travelled
            # + self.config["distance_to_tv_reward"] * speed_diff \

        reward = utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["not_in_right_lane_reward"],
                           self.config["high_speed_reward"]],
                          [0, 1])

        reward = 0 if not self.vehicle.on_road else reward
        # print(f"\nreward: {reward}, \ndense rewards:\n\ttarget velocity reward: {self.config['distance_to_tv_reward'] * speed_diff},\n\tnot in RL reward:{self.config['not_in_right_lane_reward'] * (1 - (lane / max(len(neighbours) - 1, 1)))},\n\tduration reward: {self.config['distance_reward'] * km_travelled} \
        #     \nsparse rewards:\n\tcollision reward: {COL_REWARDS[collision_index]}")

        # print(f"\nreward: {reward}, \ndense rewards:\n\tnot in RL reward:{self.config['not_in_right_lane_reward'] * (1 - (lane / max(len(neighbours) - 1, 1)))} \
        #     \nsparse rewards:\n\tcollision reward: {self.config['collision_reward']}")
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
    id='overtaken-dm-env-v0',
    entry_point='highway_env.envs:OVTKDecisionMakingEnv',
)