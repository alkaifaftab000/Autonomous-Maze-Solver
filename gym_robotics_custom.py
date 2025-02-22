# robot_wrapper.py
from gymnasium import ObservationWrapper
import numpy as np

class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(RoboGymObservationWrapper, self).__init__(env)

    def reset(self):
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.process_observation(observation)
        return observation, reward, terminated, truncated, info

    def process_observation(self, observation):
        obs_map = observation['observation']  # Fixed typo from 'obseravtion'
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']
        obs_concatenated = np.concatenate((obs_map, obs_achieved_goal, obs_desired_goal))
        return obs_concatenated