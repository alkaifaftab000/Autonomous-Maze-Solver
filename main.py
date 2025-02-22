# main.py
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper  # Import the wrapper class

# Register robotics environments
gymnasium_robotics.register_robotics_envs()

if __name__ == '__main__':
    env_name = "PointMaze_Large-v3"
    max_episode_steps = 10000000

    # Defining structure of the maze
    STRAIGHT_MAZE =    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # Create the environment
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        render_mode="human",
        maze_map=STRAIGHT_MAZE
    )

    # Wrap the environment with the custom wrapper
    env = RoboGymObservationWrapper(env)

    # Reset the environment
    observation, info = env.reset()

    # Run for 100 steps
    for i in range(max_episode_steps):
        action = env.action_space.sample()  # Sample a random action
        observation, reward, terminated, truncated, info = env.step(action)

    print("Final observation:", observation)

    # Close the environment
    env.close()