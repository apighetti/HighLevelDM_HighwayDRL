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

COL_REWARDS = [-.05,-.5,-1]
NUM_NPCS = np.arange(15,20)

class DecisionMakingEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    LAST_STEPS = 1
    TOTAL_SPACE = 0
    LAST_VEHICLE_SPEED = 0

    LAST_ACTION = ""
    LAST_LANE_IDX = 1000
    CURR_STEPS = 0
    
    DECISION_CHANGE = 0
    
    
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.collision_reward = 0
        self.high_speed_reward = 0
        # self.km_goal_reward = 0
        # self.negative_speed_reward = 0
        self.rml_reward = 0

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
            "vehicles_count": 25, # curriculum learning su lanes e npc-vehicles
            "policy_frequency": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 120,  # [s*2]
            "ego_spacing": 1,
            "vehicles_density": 0.8,
            "collision_reward": -0.1,              # The reward received when colliding with a vehicle.
            "km_goal_reward": 1,
            "right_lane_reward": 0,            # The reward received when driving on the right-most lanes, linearly mapped to
            # "decision_change": -0.1,             # working, to be tested
            "distance_reward": 0.6,
            "high_speed_reward": 1,        # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
            "reward_speed_range": [32, 36],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        # w = self.vehicles_distribution()
        # self.c = 0
        w = [0.1, 0.4, 0.5]

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

        speed = utils.lmap(aux, [0, self.config['lanes_count']-1], [36, 20])
        return speed

    def _create_vehicles(self, vehicle_distribution) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        npcs_num = self.config["vehicles_count"] # random.choice(NUM_NPCS)       
        
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(npcs_num, num_bins=self.config["controlled_vehicles"])

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
                vehicle = other_vehicles_type.create_random(self.road, speed = self.get_npc_speed(aux),\
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
            
        # lanes_count = len(self.road.network.lanes_list())

        # duration_diff = utils.lmap((self.config['duration'] - self.steps), [self.config['duration'],0], [0,1])
        self.TOTAL_SPACE += abs(self.vehicle.speed*(self.steps - self.LAST_STEPS))
        self.LAST_VEHICLE_SPEED = self.vehicle.speed
        self.LAST_STEPS = self.steps
        km_travelled = utils.lmap(round(self.TOTAL_SPACE,3), [0,36*self.config['duration']], [0,1])

        self.DECISION_CHANGE = 0
        if self.LAST_ACTION != self.vehicle.current_action:
            if self.LAST_ACTION != "":
                self.DECISION_CHANGE = 1
            self.LAST_ACTION = self.vehicle.current_action
            
        # print(f"\ndistance to td reward {self.config['distance_reward'] * km_travelled}")
        # collision_index = int(utils.lmap(abs(self.steps - self.config['duration']), [0,self.config['duration']], [2,0]))
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # negative_scaled_speed = utils.lmap(forward_speed, [0, self.config["reward_speed_range"][0]], [-1, 0])

        # print(f'dist rew: {self.config["distance_reward"] * km_travelled}')
        # print(f'nrl rew: {self.config["not_in_right_lane_reward"] * (1 - (lane / max(len(neighbours) - 1, 1)))} driving in lane: {lane}')
        # print(f'dist to tv rew: {self.config["distance_to_tv_reward"] * speed_diff} driving at {self.vehicle.speed}')
        self.callback_listener(scaled_speed, lane, neighbours)
        # self.km_goal_reward = (self.config["km_goal_reward"] * km_travelled) if not self.vehicle.crashed else 0
        # self.negative_speed_reward = self.config["high_speed_reward"] * np.clip(negative_scaled_speed, 0, 1)
        
        reward = \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["distance_reward"] * km_travelled
            # + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \


            # + self.negative_speed_reward
            # + self.config["decision_change"] * self.DECISION_CHANGE \
            # + self.config["distance_to_tv_reward"] * speed_diff \
                   
        # self.c += 1
        
        reward = utils.lmap(reward,
                          [0,
                           self.config["high_speed_reward"] + self.config["distance_reward"]], # + self.config["right_lane_reward"]
                          [0, 1])
        
        reward += self.config["collision_reward"] * self.vehicle.crashed
        
        
        if(self._is_terminal()):
            self.CURR_STEPS += self.steps
            # reward = \
                # + self.km_goal_reward \
            #     + self.collision_reward
                    
        # print(f"\nreward: {reward}, \ndense rewards:\n\trml reward: {self.rml_reward},\n\thigh speed reward: {self.high_speed_reward}\
        #     \nsparse rewards:\n\tcollision reward 1: {self.collision_reward}, \n\t2: ", self.config["collision_reward"] * self.vehicle.crashed)
            
        reward = 0 if not self.vehicle.on_road else reward
            
        # print(f"high speed rew: {self.high_speed_reward}, final rew: {reward}")
        return reward
    
    def callback_listener(self, scaled_speed, lane, neighbours):
        # Sparse rewards
        self.collision_reward = self.config["collision_reward"] * self.vehicle.crashed
        
        # Dense rewards
        self.high_speed_reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        self.rml_reward = self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1)
        
    def random_action(self):
        actions = [self.action_type.actions_indexes['ACC'], self.action_type.actions_indexes['OVERTAKE'], self.action_type.actions_indexes['RIGHTMOSTLANE']]
        random_action = random.choice(actions)
        return random_action

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        self.LAST_STEPS = 1
        self.TOTAL_SPACE = 0
        self.LAST_VEHICLE_SPEED = 0
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