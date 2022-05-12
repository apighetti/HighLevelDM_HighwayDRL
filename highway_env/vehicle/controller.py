import math
from optparse import Option
from pickle import TRUE
import time
from typing import List, Tuple, Union, Optional
from xmlrpc.client import Boolean
from cv2 import threshold

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle
import time


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    # KP_A = 1 / TAU_ACC - 0.
    KP_A = 0.5
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 6  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 phy_action: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        self.phy_action = phy_action

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
                
        if self.phy_action:
            action = self.phy_action if action == "ACC" else \
                {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.phy_action["acceleration"]}     # our defined action
            # print(f"phy_action output: {action}")
        else:
            action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.speed_control(self.target_speed)}
            
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        slip_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command, -1, 1))
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index+1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                     for t in times]))

class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""
    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 20)

    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: Optional[LaneIndex] = None,
                 target_speed: Optional[float] = None,
                 target_speeds: Optional[Vector] = None,
                 route: Optional[Route] = None) -> None:
        """
        Initializes an MDPVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        """
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else self.DEFAULT_TARGET_SPEEDS
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 5
        else:
            super().act(action)

        self.speed_index = int(np.clip(self.speed_index, 0, self.target_speeds.size - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()


    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """

        x = (speed - self.target_speeds[0]) / (self.target_speeds[-1] - self.target_speeds[0])
        return np.int(np.clip(np.round(x * (self.target_speeds.size - 1)), 0, self.target_speeds.size - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0])
        return np.int(np.clip(
            np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)), 0, cls.DEFAULT_TARGET_SPEEDS.size - 1))

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states

class PID:
    
    def __init__(self,
                 K_P: float,
                 K_I: float,
                 K_D: float) -> None:
        
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.prev_error = 0
        self.integral_error = 0
        self.last_time = time.perf_counter()
        
    def clear(self):
        self.prev_error = 0
        self.integral_error = 0
        
    def get_value(self, value, target_value):
        error = target_value - value
        t_m = time.perf_counter()
        d_error = (error - self.prev_error)/(t_m-self.last_time)
        i_error= self.integral_error + error*(t_m-self.last_time)
        t = self.K_P * error + self.K_D * d_error + self.K_I * i_error
        # print(t_m-self.last_time)
        self.prev_error = error 
        self.integral_error = i_error
        self.last_time = t_m
        
        # print(f"value: {value}, target value: {target_value}, throttle: {t}")
        
        return t
    

##### Thesis add-on #####
class DecisionMakingVehicle(MDPVehicle):
        
    """An MDP vehicle which performs high-level decision making actions."""

    MAX_SPEED = 36 # m/s
    TTG = 2
    old_action = ""
    
    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: Optional[LaneIndex] = None,
                 target_speed: Optional[float] = None,
                 target_speeds: Optional[Vector] = None,
                 route: Optional[Route] = None,
                 front_vehicle: Optional[Vehicle] = None,
                 velocity_integral : Optional[float] = 0.0,
                 prev_velocity : Optional[float] = 0.0,
                 acc_flag: Optional[Boolean] = False,
                 rml_flag: Optional[Boolean] = False,
                 overtake_flag: Optional[Boolean] = False,
                 throttle: Optional[float] = 0.0,
                 timer: Optional[int] = 0,
                 my_lane: Optional[int] = 0,
                 last_throttle: Optional[float] = 0.0
                 ) -> None:
                 
        """
        Initializes a DecisionMakingVehicle

        """
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, target_speeds, route)
        self.front_vehicle = front_vehicle
        self.acc_flag = acc_flag
        self.rml_flag = rml_flag
        self.overtake_flag = overtake_flag
        self.throttle = throttle
        self.timer = timer
        self.velocity_integral = velocity_integral
        self.prev_velocity = prev_velocity
        self.my_lane = my_lane
        self.last_throttle = last_throttle
        self.pid_brake = PID(0.65, 0, 0.9)
        self.pid_acc = PID(0.3, 0, 0.2) # 0.8
        self.pid_overtake = PID(0.3, 0, 0)

    def act(self, action: Union[dict, str] = None) -> None:
        
        """
        Perform a high-level decision making action.

        - If the action is a decision making-level action, choose action from high-level decision space.
        - Else, forward action to the MDPVehicle handler.

        :param action: a high-level action
        """
        if action == "ACC":
            # print("\nACC")
            if(self.rml_flag or self.overtake_flag):
                self.rml_flag = False
                self.overtake_flag = False
                self.pid_acc.clear()
                self.pid_brake.clear()

            if(not self.acc_flag):
                self.acc_flag = True
                # print("ACC ON")
                        
        elif action == "OVERTAKE":
            # print("\nOVRTK")
            if(self.acc_flag or self.rml_flag):
                self.acc_flag = False
                self.rml_flag = False
                self.pid_acc.clear()
                self.pid_brake.clear()

            if(not self.overtake_flag):
                self.overtake_flag = True
                self.my_lane = self.lane_index[2] - 1
                # print("OVERTAKE ON")

        elif action == "RIGHTMOSTLANE":
            # print("\nRML")
            if(self.acc_flag or self.overtake_flag):
                self.acc_flag = False
                self.overtake_flag = False
                self.pid_acc.clear()
                self.pid_brake.clear()

            if(not self.rml_flag):
                self.rml_flag = True
                self.timer = -1
                self.phy_action = None
                # self.my_lane = self.lane_index[2] + 1
                # print("RIGHTMOSTLANE ON")
        else:
            super().act(action)
            return
        super().act()

    # Utilities / Computations
    
    def get_front_vehicle(self) -> Vehicle:
        front_vehicle , _ = self.road.neighbour_vehicles(self, self.lane_index)
        return front_vehicle

    def get_safe_distance(self) -> float:
        return (self.speed * 3.6 / 10)**2

    def time_gap_error(self, target_time_gap: int, vehicleA: Vehicle, vehicleB: Vehicle) -> float:
        
        if not vehicleB:
            return None
        
        clearance = vehicleB.position[0] - vehicleA.position[0] #[m]
        time_gap = clearance / (vehicleA.speed + 0.0001) #[s]
        gap = time_gap - target_time_gap
        # print(f"\ngap: {gap}")

        return gap

    # def throttle_map(self, throttle, norm_factor) -> float:
    #     if norm_factor == 0.012:
    #         if self.last_throttle < 0:
    #             self.last_throttle = 0
    #         if abs(throttle - self.last_throttle) > norm_factor:
    #             if throttle < self.last_throttle:
    #                 throttle = self.last_throttle - norm_factor
    #             else:
    #                 throttle = self.last_throttle + norm_factor
    #     else:
    #         if self.last_throttle > 0:
    #             self.last_throttle = 0
    #         if throttle + self.last_throttle < norm_factor:
    #             if throttle < self.last_throttle:
    #                 throttle = self.last_throttle + norm_factor
    #             else:
    #                 throttle = self.last_throttle - norm_factor

    #     if throttle > 5:
    #         throttle = 5
    #     elif throttle < -5:
    #         throttle = -5
    #     self.last_throttle = throttle
    #     return throttle

    def throttle_map(self, throttle):
        if throttle > 5:
            throttle = 5
        elif throttle < -5:
            throttle = -5
        return throttle

    def physical_validity_modifier(self, target_speed = None , target_time_gap = None, is_overtaking = False):
        # print(target_time_gap)
        # throttle_brk = 0
        # throttle_accl = 0
        if(target_time_gap):
            
            # print(self.pid_brake.get_value(self.time_gap_error(2, self, self.front_vehicle), target_time_gap),\
            #     self.pid_acc.get_value(self.time_gap_error(2, self, self.front_vehicle), target_time_gap))
            
            if abs(target_time_gap) < 0.1:
                return 0

            if target_time_gap < 0:
                throttle = -self.pid_brake.get_value(target_time_gap, self.TTG)

            else :
                throttle = -self.pid_acc.get_value(target_time_gap, self.TTG)
            # return 0.9 * target_time_gap + 0.005*(self.speed - self.prev_speed)/0.05 if target_time_gap < 0 else target_time_gap * 0.7 + 0.005*(self.speed - self.prev_speed)/0.05
        else:
            # throttle = self.speed_control(target_speed)
            throttle = self.pid_overtake.get_value(self.speed, target_speed) if is_overtaking \
                else self.pid_acc.get_value(self.speed, target_speed)

        return self.throttle_map(throttle)
        # return self.throttle_map(throttle_accl, 0.012) if throttle_accl != 0 else self.throttle_map(throttle_brk, -0.07)
        
    def tactical_dm(self, action: Union[dict, str] = None) -> None:
        
        ''' Tactical module that handles heuristic for each possible action'''
        
        if(action == "ACC"):
            
            '''Adaptive Cruise Control. The ego vehicle keeps the time gap from the front vehicle '''

            # print(self.front_vehicle)

            if(self.front_vehicle):
                gap = self.time_gap_error(self.TTG, self, self.front_vehicle)
                d_speed = self.front_vehicle.speed
                self.distance = self.lane_distance_to(self.front_vehicle, self.lane)

                if(d_speed > self.MAX_SPEED):
                    phy_acceleration = self.physical_validity_modifier(target_speed=self.MAX_SPEED)
                else:
                    phy_acceleration = self.physical_validity_modifier(target_time_gap=gap)

            else:
                phy_acceleration = self.physical_validity_modifier(target_speed=self.MAX_SPEED)
                
            self.throttle = phy_acceleration
            phy_steering = 0.0
            self.phy_action = {"steering": phy_steering, "acceleration": phy_acceleration}
            
            f = open(r'C:\Users\luka-\OneDrive\Documenti\Università\Laurea Magistrale\Final Thesis\HighwayDRL\highway_env\ACC_data.csv', 'a')
            f.write(str(self.speed) + "," + str(self.front_vehicle.speed) + "," \
            + str(self.phy_action['acceleration']) + "," \
            + str(self.front_vehicle.position[0] - self.position[0]) + "," + str(gap) + "," + str(time.perf_counter()) +"\n")
            

        elif(action == "OVERTAKE"):
            
            '''Left overtake action. If the ego vehicle is driving faster than the front vehicle
               then it will perform a left overtake.  '''

            curr_lane_index = self.lane_index
            phy_acceleration = self.physical_validity_modifier(target_speed=self.MAX_SPEED, is_overtaking=True)
            self.throttle = phy_acceleration
            self.phy_action = {"steering": 0.0, "acceleration": phy_acceleration}

            if (self.my_lane == curr_lane_index[2] - 1):
                if (self.lane_index[2] > 0):
                    if (self.front_vehicle):
                        gap = self.time_gap_error(2, self, self.front_vehicle)
                        # print(f"time gap to front vehicle: {gap}, current lane {self.lane_index}")
                        
                        if(gap < 0):
                            self.my_lane = curr_lane_index[2] - 2
                            super().act("LANE_LEFT")

                    # left_lane_index = self.lane_index[0], self.lane_index[1], self.lane_index[2]-1
                    # front_left_vehicle, rear_left_vehicle = self.road.neighbour_vehicles(self, left_lane_index)
                    
                    # if rear_left_vehicle and not front_left_vehicle:
                    #     print("Rear Left Vehicle: " + str(rear_left_vehicle)+"\n")
                    #     rear_gap, _ = self.time_gap_error(rear_left_vehicle, self)  
                    #     print(str(rear_gap)+"\n")
                    #     if rear_gap > 0:
                    #         super().act("LANE_LEFT")  
                    
                    # elif front_left_vehicle and not rear_left_vehicle:
                    #     print("Front Left Vehicle: " + str(front_left_vehicle)+"\n")
                    #     front_gap, _ = self.time_gap_error(self, front_left_vehicle)
                    #     print(str(front_gap)+"\n")
                    #     if front_gap > 0:
                    #         super().act("LANE_LEFT")
                    
                    # elif front_left_vehicle and rear_left_vehicle:
                    #     print("Rear Left Vehicle: " + str(rear_left_vehicle) +"\n")
                    #     print("Front Left Vehicle: " + str(front_left_vehicle) +"\n")
                    #     rear_gap, _ = self.time_gap_error(rear_left_vehicle, self)  
                    #     front_gap, _ = self.time_gap_error(self, front_left_vehicle)
                    #     print(str(rear_gap)+"\n")
                    #     print(str(front_gap)+"\n")
                    #     if front_gap > 0 and rear_gap > 0:
                    #         super().act("LANE_LEFT")
                            
                    # else:
                    #     super().act("LANE_LEFT")
        
        elif(action == "RIGHTMOSTLANE"):
            lanes_count = len(self.road.network.lanes_list())
            curr_lane_index = self.lane_index
            self.throttle = 0.0
            self.phy_action = {"steering": 0.0, "acceleration": self.throttle}
            # if (self.my_lane == curr_lane_index[2] + 1):

            # print(f"curr lane index: {curr_lane_index[2]}, lanes count: {lanes_count-1}")
            if(curr_lane_index[2] != lanes_count-1):
                if(self.timer == -1):
                    super().act("LANE_RIGHT")
                if(self.timer != 60):
                    super().act("IDLE")
                    self.timer += 1
                else:
                    super().act("LANE_RIGHT")
                    self.timer = 0         
                    
                    # next_lane_index = (curr_lane_index[0], curr_lane_index[1], curr_lane_index[2] + 1)
                    # right_front_vehicle, right_rear_vehicle = self.road.neighbour_vehicles(self, next_lane_index)

                    # if(right_rear_vehicle and not right_front_vehicle):
                    #     if(self.time_gap_error(1.5, right_rear_vehicle, self) > 0):
                    #         self.my_lane = curr_lane_index[2] + 2
                    #         super().act("LANE_RIGHT")
                    # elif(right_front_vehicle and not right_rear_vehicle):
                    #     if(self.time_gap_error(2, self, right_front_vehicle) > 0):
                    #         self.my_lane = curr_lane_index[2] + 2
                    #         super().act("LANE_RIGHT")
                    # elif(right_front_vehicle and right_rear_vehicle):
                    #     if(self.time_gap_error(2, self, right_front_vehicle) > 0 and self.time_gap_error(1.5, right_rear_vehicle, self) > 0):
                    #         self.my_lane = curr_lane_index[2] + 2
                    #         super().act("LANE_RIGHT")                               


    # Override simulation step method to implement continuous DM actions
    def step(self, dt: float) -> None:
        self.front_vehicle = self.get_front_vehicle()
        if(self.acc_flag):
            # print(dt)
            self.tactical_dm("ACC")
            
            
        elif(self.overtake_flag):
             self.tactical_dm("OVERTAKE")
        elif(self.rml_flag):
            self.tactical_dm("RIGHTMOSTLANE")        

        super().step(dt)

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states