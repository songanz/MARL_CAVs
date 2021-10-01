from __future__ import division, print_function, absolute_import

from abc import ABC

import numpy as np
import pandas
from gym import GoalEnv, spaces
from gym.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib

from highway_env.envs.common.abstractGoal import AbstractEnvGoal
from highway_env.road.lane import StraightLane, LineType, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle


class ParkingEnv(AbstractEnvGoal, GoalEnv, ABC):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """
    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    REWARD_WEIGHTS = [1 / 100, 1 / 100, 1 / 100, 1 / 100, 1 / 10, 1 / 10]
    SUCCESS_GOAL_REWARD = 0.15

    num_controlled_vehicles = 4

    n_a = 2
    n_s = 25

    color_list = []
    for c in plt.cm.Set2.colors:
        color_list.append(tuple([int(i * 255) for i in c]))
    # obs = self.observation_type.observe() --> tuple(obs_type.observe() for obs_type in self.agents_observation_types)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scale": 100,
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": (-cls.ACCELERATION_RANGE, cls.ACCELERATION_RANGE),
                "steering_range": (-cls.STEERING_RANGE, cls.STEERING_RANGE)
            },
            "controlled_vehicles": 1,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [3.62, 0.6],
            "scaling": 6,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 5,  # [Hz]
            "traffic_density": 1,
            "action_masking": False,  # for continuous action space: turn off the action_masking
            "safety_guarantee": False  # for continuous action space: turn off the safety_guarantee
        })
        return config

    def step(self, action):
        # Forward action to the vehicle

        agents_info = []
        obs, reward, terminal, _ = super().step(action)
        reward = sum(self.compute_reward(ob['achieved_goal'], ob['desired_goal'], {}) for ob in obs) \
                 / len(self.controlled_vehicles)
        for i, v in enumerate(self.controlled_vehicles):
            agents_info.append({
                "reward": self.compute_reward(obs[i]['achieved_goal'], obs[i]['desired_goal'], {}),
                "agent_is_success": self._agent_is_success(i, obs),
                "agent_is_crashed": v.crashed,
                "agent_position": v.position,
                "agent_speed": v.speed,
                "agent_direction": v.direction  # np.array([np.cos(v.heading), np.sin(v.heading)])
            })
        info = {"agents_info": agents_info}
        terminal = self._is_terminal(obs=obs)
        return obs, reward, terminal, info

    def _reset(self, is_training=True, testing_seeds=0, num_CAV=num_controlled_vehicles):
        self._make_road()
        self._make_vehicles(num_CAV=num_CAV)
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self, spots=15):
        """
            Create a road composed of straight adjacent lanes.
        """
        net = RoadNetwork()

        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        width = 4.0

        self.x_range = ((0 - spots // 2) * (width + x_offset) - width / 2,
                        (14 - spots // 2) * (width + x_offset) - width / 2)

        self.y_range = (-y_offset - length, y_offset + length)

        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset + length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset - length], width=width, line_types=lt))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_vehicles(self, num_CAV=0):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.controlled_vehicles = []
        self.goals = []

        for i in range(num_CAV):
            x = self.np_random.uniform(self.x_range[0], self.x_range[1])
            y = self.np_random.uniform(self.y_range[0], self.y_range[1])
            ego_vehicle = self.action_type.vehicle_class(self.road, [x, y], 2 * np.pi * self.np_random.rand(), 0)
            for other in self.road.vehicles:
                ego_vehicle.check_collision(other)
                if ego_vehicle.crashed:
                    i -= 1
                    continue
            ego_vehicle.color = self.color_list[i]
            self.controlled_vehicles.append(ego_vehicle)
            self.road.vehicles.append(ego_vehicle)

            lane = self.np_random.choice(self.road.network.lanes_list())
            goal = Obstacle(self.road, lane.position(lane.length / 2, 0), heading=lane.heading)
            goal.COLLISIONS_ENABLED = False
            goal.color = self.color_list[i]
            self.goals.append(goal)
            self.road.objects.append(goal)


    def compute_reward(self, achieved_goal, desired_goal, info, p=0.5):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: an info dictionary with additional information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return - np.power(
            np.dot(self.observation_type.observation_config['scale'] * np.abs(achieved_goal - desired_goal),
                   np.array(self.REWARD_WEIGHTS)), p)

    def _reward(self, action, obs=None, info=None):
        return sum(self.compute_reward(ob['achieved_goal'], ob['desired_goal'], info) for ob in obs) \
               / len(self.controlled_vehicles)

    def _agent_is_success(self, index, obs):
        achieved_goal = obs[index]['achieved_goal']
        desired_goal = obs[index]['desired_goal']
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.SUCCESS_GOAL_REWARD

    def _agent_is_terminal(self, index, obs=None):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        done = self.controlled_vehicles[index].crashed
        if obs is not None:
            done = done or \
                   self._agent_is_success(index, obs) or \
                   self.steps >= self.config["duration"] * self.config["policy_frequency"]
        return done

    def _is_terminal(self, obs=None) -> bool:
        return all(self._agent_is_terminal(index, obs) for index in range(len(self.controlled_vehicles)))


class ParkingEnvMARL(ParkingEnv, ABC):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "ContinuousAction",
                    "lateral": True,
                    "longitudinal": True,
                    "acceleration_range": (-cls.ACCELERATION_RANGE, cls.ACCELERATION_RANGE),
                    "steering_range": (-cls.STEERING_RANGE, cls.STEERING_RANGE)
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "KinematicsGoal",
                    "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                    "scale": 100,
                    "normalize": False
                }},
            "controlled_vehicles": cls.num_controlled_vehicles
        })
        return config


register(
    id='parking-v1',
    entry_point='highway_env.envs:ParkingEnv',
)

register(
    id='parking-multi-agent-v0',
    entry_point='highway_env.envs:ParkingEnvMARL',
)
