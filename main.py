# main.py
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper  # Import the wrapper class
from agent import Agent
from buffer import ReplayBuffer

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

    observation_size = observation.shape[0]  # This gets the correct state dimension

    # Agent
    agent = Agent(
        num_inputs=observation_size,          # Changed from observation_size
        action_space=env.action_space,        # Changed from action_shape
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        target_update_interval=target_update_interval,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        exploration_scaling_factor=exploration_scaling_factor
    )   
    memory = ReplayBuffer(
        max_size=replay_buffer_size,
        input_size=observation_size,  # This should match the actual observation size
        n_actions=env.action_space.shape[0]
    )
    # Run for 100 steps

    agent.train(env = env, env_name=env_name, episodes=100, warmup=warmup, batch_size=batch_size, updates_per_step=updates_per_step, summary_writer_name = f"straight_maze{alpha}_lr = {learning_rate}_hs{hidden_size}_a ={alpha}", memory=memory)
    # for i in range(max_episode_steps):
    #     action = env.action_space.sample()  # Sample a random action
    #     observation, reward, terminated, truncated, info = env.step(action)

    # print("Final observation:", observation)

    # # # Close the environment
    # env.close()