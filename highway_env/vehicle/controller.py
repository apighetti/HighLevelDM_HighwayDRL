from pickle import TRUE
import time
from typing import List, Tuple, Union, Optional
from xmlrpc.client import Boolean

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


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
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 acceleration: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        self.acceleration = acceleration

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
                
        if self.acceleration:
            action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.acceleration}
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


##### TO-DO 
class DecisionMakingVehicle(MDPVehicle):
    
    ACCELERATION_VALUES = np.linspace(-20,20,50)    #acceleration values in m/s^2
    
    """A controlled vehicle which performs high-level decision making actions."""

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
                 ttc: Optional[float] = None,
                 acc_flag: Optional[Boolean] = False,
                 distance: Optional[float] = None,
                 safe_distance: Optional[float] = None) -> None:
        """
        Initializes a DecisionMakingVehicle

        """
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, target_speeds, route)
        self.front_vehicle = front_vehicle
        self.ttc = ttc
        self.acc_flag = acc_flag
        self.distance = distance
        self.safe_distance = safe_distance

    def act(self, action: Union[dict, str] = None) -> None:
        
        """
        Perform a high-level action.

        - If the action is a decision making-level action, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        
        if action == "ACC":
            if(not self.acc_flag):
                self.acc_flag = True
                print("ACC ON")
            else:
                self.acc_flag = False
                print("ACC OFF")
        
            print(f"My speed: {self.speed}")
            print(f"front_vehicle: {self.front_vehicle}")
            print("Time to collision :", str(self.ttc))
                
        elif action == "OVERTAKE":
            # DO SOMETHING
            print("OVERTAKE")

            if self.lane_index[2] > 0:
                print("ENTRY POINT 1")
                front_vehicle_init , _ = self.road.neighbour_vehicles(self, self.lane_index)
                left_li = (self.lane_index[0], self.lane_index[1], self.lane_index[2] - 1)
                # front_vehicle_left, rear_vehicle_left = self.road.neighbour_vehicles(self, left_li)
                print(f"\nFront vehicle speed: {front_vehicle_init.speed}, My speed: {self.speed}\n")
                if(front_vehicle_init.speed < self.speed):
                    print("ENTRY POINT 2")
                    
                    # if(rear_vehicle_left):
                    #     overtake_gap = (self.position[0] + self.LENGTH) - (rear_vehicle_left.position[0] + rear_vehicle_left.LENGTH)
                    #     if(self.speed < rear_vehicle_left.speed or overtake_gap <= 25):
                    #         while():
                    #             super().act("SLOWER")
                    super().act("LANE_LEFT")
                    print(self.target_lane_index[2])
                    # right_li = (self.lane_index[0], self.lane_index[1], self.lane_index[2] + 1)
                    # _, rear_vehicle_right = self.road.neighbour_vehicles(self, right_li)

                    # print(f"\nAfter LANE_LEFT:\nFront vehicle init: {front_vehicle_init}, Rear vehicle right: {rear_vehicle_right}\n")
                
                    # while(front_vehicle_init != rear_vehicle_right):
                    #     super().act("FASTER")
                    
                    # reenter_gap = (self.position[0] + self.LENGTH) - (rear_vehicle_right.position[0] + rear_vehicle_right.LENGTH)
                    # if(reenter_gap >= 25):
                    #     super().act("LANE_RIGHT")

        elif action == "RIGHT-MOST_LANE":
            # DO SOMETHING
            print("RIGHT-MOST LANE")
        else:
            super().act(action)
            return
        super().act()

    # Utilities / Computations
    def get_front_vehicle(self) -> Vehicle:
        front_vehicle , _ = self.road.neighbour_vehicles(self, self.lane_index)
        return front_vehicle

    def get_ttc_distance(self, front_vehicle: Vehicle) -> float:
        distance = self.lane_distance_to(front_vehicle, self.lane)
        other_projected_speed = front_vehicle.speed * np.dot(front_vehicle.direction, self.direction)
        time_to_collision = distance / utils.not_zero(self.speed - other_projected_speed)
        return distance, time_to_collision

    def get_safe_distance(self) -> float:
        return (self.speed * 3.6 / 10)**2

    def compute_acceleration(self, ttc: float, target_speed: float) -> float:
        
        '''compute acceleration using the formula K/a(c + ttc)'''
        
        K = 0.6
        alpha = 0.2
        
        return super().speed_control(target_speed) - K/alpha*(1+ttc)


    def acc_on(self) -> None:
        """
        ACC Turn on module
        """
        self.front_vehicle = self.get_front_vehicle()

        ''' acceleration formula a_id - a_ttc'''

        if(self.front_vehicle):
            self.distance, self.ttc = self.get_ttc_distance(self.front_vehicle)
            if((self.ttc > 12.5 or self.ttc < 0) and (self.distance > self.get_safe_distance())):
                
                #super().act("FASTER")
                self.acceleration = self.compute_acceleration(self.front_vehicle.speed)
                
                print(f"going slower, acceleration: {self.acceleration}")
                
            elif (self.ttc < 12.5 and self.ttc > 11.5 and self.distance > self.get_safe_distance()):
                super().act("IDLE")
                print(f"idle, ttc: {self.ttc}, safe distance: {self.safe_distance}")
            else:
                #super().act("SLOWER")
                self.acceleration = self.compute_acceleration(self.front_vehicle.speed)
                print(f"going slower, acceleration: {self.acceleration}")
        else:
            super().act()

    def step(self, dt: float) -> None:
        if(self.acc_flag):
            
            '''dt = 0.02 s
                a = dv / dt'''
            
            self.acc_on()

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