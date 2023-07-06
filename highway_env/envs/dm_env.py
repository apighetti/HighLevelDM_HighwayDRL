from distutils.command.config import config
import random
from gym.envs.registration import register
import numpy as np
import math

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv,Observation
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import VictimVehicle
from highway_env.vehicle.objects import LaneIndex
from stable_baselines3 import PPO

NUM_NPCS = np.arange(10,15)

class DecisionMakingEnv(AbstractEnv):
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
                "type": "Kinematics",
                "vehicles_count": 7
            },
            "action": {
                "type": "DecisionMakingAction",
            },
            "lanes_count": 3,
            "simulation_frequency": 5,
            "policy_frequency": 1,
            "controlled_vehicles": 1,
            "duration": 60,  # [s*2]
            "initial_lane_id": None,
            "ego_spacing": 1,
            "vehicles_density": 0.5,
            "offroad_terminal": False,
            
            "collision_reward": -30,
            # "km_sparse_reward": 10,
            "rml_reward": 0.8,
            # "km_dense_reward": 0.6,
            "high_speed_reward": 0,
            "reward_speed_range": [30, 36]
        })
        return config

    def _reset(self) -> None:
        w = [0.1, 0.4, 0.5]
        self._create_road()
        self._create_vehicles(w)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=36),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
    
    def get_npc_speed(self, aux):
        '''Compute speed of a spawned vehicle according to its position.'''

        speed = utils.lmap(aux, [0, self.config['lanes_count']-1], [36, 20])
        return speed

    def _create_vehicles(self, vehicle_distribution) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        npcs_num = random.choice(NUM_NPCS)
           
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
                    
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        self.rml_reward = self.config["rml_reward"] * lane / max(len(neighbours) - 1, 1)
        
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        self.high_speed_reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        
        
        self.final_reward = self.dense_reward = \
            + self.high_speed_reward \
            + self.rml_reward
                       
        self.final_reward = self.dense_reward = utils.lmap(self.final_reward,
                [0,
                 self.config["high_speed_reward"] + self.config["rml_reward"]],
                [0, 0.1])
        
        self.collision_reward = self.config["collision_reward"] * self.vehicle.crashed

        self.sparse_reward = self.collision_reward
        
        self.final_reward += self.sparse_reward
        
        if self._is_terminal():
            self.terminal = True
            # self.km_sparse_reward = self.config["km_sparse_reward"] * self.km_travelled if not self.vehicle.crashed else 0
            
                # + self.km_sparse_reward
                
            # Reset counters
            # self.total_speed = 0
            # self.km_travelled = 0
        
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
    id='dm-env-v0',
    entry_point='highway_env.envs:DecisionMakingEnv',
)
       

class MultiAgentDecisionMakingEnv(DecisionMakingEnv):
    """
    A variant of the original high-level decision-making environment
    with a pseudo-multiagent setting to enable adversarial policy training.
    """
    
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.terminal = False

        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "PseudoMultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 7
                }
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "controlled_vehicles": 1,
            "lanes_count": 3,
            "victim_initial_lane_id": None,
            "victim_loaded_model": PPO.load('/home/pigo/HighwayDRL/final_models/ppo_standard_200k_FORNO'),
            "victim_spacing": 1.5,
            "vehicles_density": 0.5,
            
            "training_total_timesteps": 3e5,
            
            "distance_to_victim_reward": -0.1,
            
            "victim_collision_reward": +10,
            "self_collision_reward": -5,
            "game_over_reward": -5
        })
        return config
    
    def reset(self) -> Observation:
        self.obs = super().reset()
        self.victim_vehicle.update_obs(self.observation_type.victim_observe())
        return self.obs
        
    
    def _reset(self) -> None:
        w = [0.1, 0.4, 0.5]
        self._create_road()
        self._create_vehicles(w)
        
    def _create_vehicles(self, vehicle_distribution) -> None:
        super()._create_vehicles(vehicle_distribution)

        self.victim_vehicle = VictimVehicle.create_random(self.road,
                                                    speed=25,
                                                    lane_id=self.config['victim_initial_lane_id'],
                                                    spacing=self.config['victim_spacing'])
        self.victim_vehicle = VictimVehicle(self.road, [self.vehicle.position[0]-50, self.vehicle.position[1]] , self.victim_vehicle.heading,\
            self.victim_vehicle.speed, victim_model=self.config['victim_loaded_model'])
        self.road.vehicles.append(self.victim_vehicle)
            
    def step(self, action: Action):
        self.obs, reward, terminal, info = super().step(action)
        self.victim_vehicle.update_obs(self.observation_type.victim_observe())
        return self.obs, reward, terminal, info
    
    def _reward(self, action: Action) -> float:
        
        # Reset rewards
        self.distance_to_victim_reward = 0
        self.dense_reward = 0
        self.collision_reward = 0
        self.sparse_reward = 0
        self.final_reward = 0
        self.terminal = False
        
                
        euc_distance = math.sqrt((self.vehicle.position[0] - self.victim_vehicle.position[0])**2 + (self.vehicle.position[1] - self.victim_vehicle.position[1])**2)
        
        self.distance_to_victim_reward =  np.interp(euc_distance, (0, 80), (0, 1)) * self.config['distance_to_victim_reward']
        
        self.dense_reward = self.distance_to_victim_reward
        
        self.collision_reward = self.victim_vehicle.crashed * self.config['victim_collision_reward'] \
            + self.vehicle.crashed * self.config['self_collision_reward']        
        
        if self._is_terminal():            
            self.sparse_reward = self.collision_reward if self.collision_reward > 0 \
                else self.config['game_over_reward'] + self.collision_reward
                
            self.terminal = True
            
        self.final_reward += self.dense_reward + self.sparse_reward

        self.final_reward = 0 if not self.vehicle.on_road else self.final_reward        
        return self.final_reward
    
    def _is_terminal(self) -> bool:
        """The episode is over if the victim vehicle crashed or the time is out."""
        return (any([self.victim_vehicle.crashed, self.vehicle.crashed]) or \
            self.steps >= self.config["duration"] or \
            self.config["offroad_terminal"] and not self.vehicle.on_road)
    

register(
    id='dm-multi-agent-v0',
    entry_point='highway_env.envs:MultiAgentDecisionMakingEnv',
)