# main.py
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper  # Import the wrapper class

# Register robotics environments
gymnasium_robotics.register_robotics_envs()

if __name__ == '__main__':

    replay_buffer_size = 1000000
    episodes = 1000
    warmup = 64
    batch_size = 256
    updates_per_step = 1
    gamma = 0.99
    tau = 0.99
    alpha = 0.12
    target_update_interval = 1
    hidden_size = 512
    learning_rate = 0.0001
    exploration_scaling_factor = 1.5
    # evaluate = False

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
        # render_mode="human",
        maze_map=STRAIGHT_MAZE
    )

    # Wrap the environment with the custom wrapper
    env = RoboGymObservationWrapper(env)

    # Reset the environment
    observation, info = env.reset()

    # critic = Critic(1,1,1)

    observation_size = observation.shape[0]

    # Agent
    agent = Agent(
        observation_size,
        env.action_space.shape,
        hidden_size,
        replay_buffer_size,
        gamma =gamma,
        tau = tau,
        alpha=alpha,
        target_update_interval=target_update_interval,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        exploration_scaling_factor=exploration_scaling_factor
    )

    memory = ReplayBuffer(replay_buffer_size,input_size=observation_size,n_actions=env.action_space.shape[0])
    # Run for 100 steps
    for i in range(max_episode_steps):
        action = env.action_space.sample()  # Sample a random action
        observation, reward, terminated, truncated, info = env.step(action)

    print("Final observation:", observation)

    # # Close the environment
    env.close()