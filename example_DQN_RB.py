import numpy as np
import matplotlib.pyplot as plt
import mujoco  # Add back mujoco import

from mecanum_gen import generate_scene_empty

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.envs.registration import register

import copy
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
from IPython import display
import math
import torchvision.transforms as T
import time

GENERATE_SCENE = False

def generate_coordinates(n, init_pos=(0, 0), x_range=(-1, 1), y_range=(-1, 1)):
    """Generates N random (x, y) coordinate pairs within specified ranges.

    Args:
        n (int): Number of coordinate pairs.
        init_pos (tuple): initial coordinates
        x_range (tuple): Range (min, max) for x values.
        y_range (tuple): Range (min, max) for y values.

    Returns:
        numpy.ndarray: Array of shape (N, 2) containing (x, y) pairs.
    """
    x_values = init_pos[0] + np.random.uniform(x_range[0], x_range[1], n)
    y_values = init_pos[1] + np.random.uniform(y_range[0], y_range[1], n)

    coordinates = np.column_stack((x_values, y_values))  # Combine x and y into pairs
    return coordinates

def show_state(environment, episode=0, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(environment.render())#mode='rgb_array'))
    plt.title("%s | Eposide: %d | Step: %d %s" % ('Mecanum platform', episode, step, info))
    plt.axis('off')


class DQN():
    def __init__(self, state_dim, action_dim, hidden_dim=10,alpha=0.001):
        self.criterion=torch.nn.MSELoss()
        self.model=torch.nn.Sequential(
            torch.nn.Linear(state_dim,hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim,2*hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2*hidden_dim,action_dim)
        )
        self.optimizer=torch.optim.Adam(self.model.parameters(), alpha)
        self.scheduler=StepLR(self.optimizer, step_size=1,gamma=0.5)

    def update(self, state, y):
        y_pred=self.model(torch.Tensor(state))
        loss=self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def replay(self, memory, replay_size, gamma):
        if len(memory) < replay_size:
            return

        batch = random.sample(memory, replay_size)

        batch_t = list(map(list, zip(*batch)))
        states = batch_t[0]
        actions = batch_t[1]
        next_states = batch_t[2]
        rewards = batch_t[3]
        is_done = batch_t[4]

        states = torch.Tensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.Tensor(next_states)
        rewards = torch.Tensor(rewards)
        is_dones_tensor = torch.Tensor(is_done)

        is_done_indices = torch.where(is_dones_tensor==True)[0]

        #predict q_values
        all_q_values = self.model(states)
        all_q_values_next = self.model(next_states)

        #update q_values
        all_q_values[range(len(all_q_values)), actions] = rewards + gamma*torch.max(all_q_values_next, axis=1).values
        all_q_values[is_done_indices.tolist(), actions[is_done].tolist()] = rewards[is_done_indices.tolist()]

        self.update(states.tolist(), all_q_values)
    

if __name__ == '__main__':
    if GENERATE_SCENE:
        spec = generate_scene_empty([0, 0, 0])
        spec.add_sensor(name='pos_c', type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
        spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_FRAMELINVEL, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
        spec.add_sensor(name='gyro_c', type=mujoco.mjtSensor.mjSENS_FRAMEANGVEL, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
        spec.add_sensor(name='xaxis_c', type=mujoco.mjtSensor.mjSENS_FRAMEXAXIS, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
        
        spec.compile()
        model_xml = spec.to_xml()
        with open("mecanum.xml", "w") as text_file:
                text_file.write(model_xml)

    CAMERA_CONFIG = {
        "type": 1,
        "trackbodyid": 1,
        "distance": 5.,
        "lookat": np.array((0.0, 0.0, 1.15)),
        "elevation": -50.0,
    }


    EPISODES_MAX = 15*1000000
    EPISODES_N = 15*1000000
    STEPS_MAX = 10000
    register(
        id="Mecanum-v0",
        entry_point="gym_env_mecanum:MecanumEnv",
        max_episode_steps=STEPS_MAX,
        reward_threshold=4600,
    )
    env_boudary_radius = 10.
    coords = generate_coordinates(EPISODES_N)
    env = gym.make('Mecanum-v0',coordinates=coords, camera_config=CAMERA_CONFIG, environment_boundary_radius=env_boudary_radius)
                #    , render_mode="human")
    # env = RecordVideo(env, video_folder="mecanum-platform", name_prefix="eval",
    #               episode_trigger=lambda x: True)
    observation, info = env.reset()
    print('Space shapes', env.observation_space.shape, env.action_space.shape)

    episode_over = False
    
    action_space = np.array([
        [ 0,   0,   0,   0  ],  # static - no movement
        [ 1,   0,   0,   1  ],  # quadrant 2, high speed
        [ 1,  -1,  -1,   1  ],  # x negative, high speed
        [ 0,  -1,  -1,   0  ],  # quadrant 3, high speed
        [-1,  -1,  -1,  -1  ],  # y negative, high speed
        [-1,   0,   0,  -1  ],  # quadrant 4, high speed
        [-1,   1,   1,  -1  ],  # x positive, high speed
        [ 0,   1,   1,   0  ],  # quadrant 1, high speed
        [ 1,   1,   1,   1  ],  # y positive, high speed
        [ 0.5,  0,   0,   0.5],  # quadrant 2, low speed
        [ 0.5, -0.5, -0.5,  0.5],  # x negative, low speed
        [ 0,  -0.5, -0.5,  0  ],  # quadrant 3, low speed
        [-0.5, -0.5, -0.5, -0.5],  # y negative, low speed
        [-0.5,  0,   0,  -0.5],  # quadrant 4, low speed
        [-0.5,  0.5,  0.5, -0.5],  # x positive, low speed
        [ 0,   0.5,  0.5,  0  ],  # quadrant 1, low speed
        [ 0.5,  0.5,  0.5,  0.5],  # y positive, low speed
        [ 1,  -1,  1,   -1  ],  # CCW rotation, high speed
        [-1,  1,   -1,  1  ],  # CW rotation, high speed
    ])
    
    # Swap the last two columns. In the article, first two columns stand for RF and LF wheels,
    #  but in our model it's vice-versa 
    action_space[:, [2, 3]] = action_space[:, [3, 2]]
    action_space = -1. * action_space 

    i = 0

    n_states = 11
    n_actions = len(action_space)

    # Learning Parameters
    epsilon = 1.0 # gready threashold
    alpha = 0.01 #5e-5 # learning rate
    gamma = 0.99 # reward discount factor

    agent = DQN(n_states, n_actions, hidden_dim=12,alpha=alpha)
    memory = []
    replay_size = 100

    # Execution parameters
    SHOW_ANIMATION = False
    DESIRED_STEPS = 10000

    # Loggers
    log_steps_number = np.zeros(EPISODES_MAX)
    log_episode_rewards = np.zeros(EPISODES_MAX)
    fails_by_cart_pos = 0
    fails_by_pole_angle = 0
    fails_both = 0

    # Q-learning
    for i_episode in range(EPISODES_MAX):
        print("episode #" + str(i_episode))
        state, _info = env.reset()
        # show results
        if (i_episode + 1) % 200 == 0:
            plt.figure(1)
            plt.clf()
            plt.plot([0,i_episode], [195, 195], label="threshold")
            plt.plot(range(0,i_episode), log_episode_rewards[0:i_episode], label="solution 1")
            plt.xlabel('episode')
            plt.ylabel('episode reward')
            plt.legend()
            plt.title('epsilon={} alpha={}'.format(epsilon, agent.scheduler.get_last_lr()[0]))
            display.clear_output(wait=True)
            plt.show()

        if (i_episode + 1) % 50 == 0: #10
            epsilon = epsilon*0.95 #0.85

        if (i_episode + 1) % 30 == 0: #50
            agent.scheduler.step()

        for t in range(STEPS_MAX):
            # env.render()
            q_values = agent.predict(state)

            if np.random.random_sample() < epsilon:
                action_n = env.action_space.sample()
                action = action_space[action_n]
            else:
                action_n = torch.argmax(q_values).item()
                action = action_space[action_n]


            next_state, reward, done, _truncated, info = env.step(action)
            memory.append((state, action_n, next_state, reward, done))

            if done:
                #update qnetwork
                q_values[action_n] = reward
                agent.update(state, q_values)

                log_steps_number[i_episode] = t
                log_episode_rewards[i_episode] = reward
                break

            if len(memory) < replay_size:
                #update qnetwork
                q_values_next = agent.predict(next_state)
                q_values[action_n] = reward + gamma * torch.amax(q_values_next)
                agent.update(state, q_values)
            else:
                agent.replay(memory, replay_size, gamma)

            #update current state
            state = next_state

    print("done")

    # for episode_counter, action in enumerate(action_space):
    #     episode_over = False
    #     i = 0
    #     print(action)
    #     traj = []
    #     while not episode_over:
    #         # action = env.action_space.sample()  # agent policy that uses the observation and info
    #         # action = action_space[np.random.choice(action_space.shape[0])]
    #         # action = -1. * np.array([-0.5, -0.5, -0.5, -0.5])  # y positive, low speed
    #         observation, reward, terminated, truncated, info = env.step(action)
    #         x, y, v_x, v_y, omega_z, x_t, y_t, e_x, e_y, E_dist, e_phi_deg = observation
    #         traj.append([x ,y])
    #         i+=1
    #         episode_over = (i >= STEPS_MAX) or terminated
    #         if episode_over:
    #             show_state(env, episode_counter, i, str(info))
    #
    #     traj = np.array(traj)
    #     # Create figure
    #     plt.figure(figsize=(8, 6))
    #     # Plot trajectory as a connected line
    #     plt.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', marker='o', markersize=5, alpha=0.7, label="Trajectory")
    #     # Highlight the last coordinate
    #     plt.scatter(traj[-1, 0], traj[-1, 1], color='green', edgecolors='black', s=100, label="Last Point", zorder=3)
    #     # Add the circle
    #     circle = plt.Circle((0,0), env_boudary_radius, color='red', fill=False, linewidth=2, linestyle='dashed')
    #     plt.gca().add_patch(circle)  # Add circle to the current plot
    #     # Labels and title
    #     plt.xlabel("X Coordinate")
    #     plt.ylabel("Y Coordinate")
    #     plt.title("Trajectory Plot with Last Point Highlighted")
    #     plt.grid(True)
    #     plt.axhline(0, color='black', linewidth=0.8)  # X-axis reference line
    #     plt.axvline(0, color='black', linewidth=0.8)  # Y-axis reference line
    #     # Set equal aspect ratio to ensure the circle looks correct
    #     plt.axis("equal")
    #     # Add legend
    #     plt.legend()
    #     # Show plot
    #     plt.show()
    #
    #     env.reset()
    #
    # env.close()


    # plt.figure(figsize=(8, 6))  # Set figure size
    # plt.scatter(coords[:, 0], coords[:, 1], c='blue', marker='o', alpha=0.7, edgecolors='black')
    # # Add the circle
    # circle = plt.Circle((0,0), env_boudary_radius, color='red', fill=False, linewidth=2, linestyle='dashed')
    # plt.gca().add_patch(circle)  # Add circle to the current plot
    #
    # # Labels and title
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.title("Scatter Plot of Random (x, y) Coordinates")
    # plt.grid(True)  # Show grid
    # plt.axhline(0, color='black', linewidth=0.8)  # X-axis reference line
    # plt.axvline(0, color='black', linewidth=0.8)  # Y-axis reference line

# Show plot
    plt.show()
