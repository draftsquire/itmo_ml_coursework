import numpy as np
import matplotlib.pyplot as plt
import mujoco  # Add back mujoco import

from mecanum_gen import generate_scene_empty
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.envs.registration import register
from logger import Logger
from rl_utils import ReplayMemory, huber

import copy
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
from IPython import display
import math
import torchvision.transforms as T
import time



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
    return coordinates #torch.tensor(coordinates, dtype=torch.float32)



GENERATE_SCENE = True
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Using device:", device)
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


    EPISODES_MAX = 1500
    MAX_EPISODE_STEPS = 10000
    MAX_LEARNING_STEPS = 15000000
    register(
        id="Mecanum-v0",
        entry_point="gym_env_mecanum:MecanumEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=6000,
    )
    env_boudary_radius = 3.
    # coords = generate_coordinates(10)
    coords = np.array([(0, -1), (0, 1)])
    env = gym.make('Mecanum-v0', render_mode='human', coordinates=coords, max_steps=MAX_EPISODE_STEPS, camera_config=CAMERA_CONFIG, environment_boundary_radius=env_boudary_radius)
    observation, info = env.reset()
    print('Space shapes', env.observation_space.shape, env.action_space.shape)

    episode_over = False
    
    action_space = np.array([
        [ 0,   0,   0,   0  ],  # static - no movement         0
        [ 1,   0,   0,   1  ],  # quadrant 2, high speed       1
        [ 1,  -1,  -1,   1  ],  # x negative, high speed       2
        [ 0,  -1,  -1,   0  ],  # quadrant 3, high speed       3
        [-1,  -1,  -1,  -1  ],  # y negative, high speed       4
        [-1,   0,   0,  -1  ],  # quadrant 4, high speed       5
        [-1,   1,   1,  -1  ],  # x positive, high speed       6
        [ 0,   1,   1,   0  ],  # quadrant 1, high speed       7
        [ 1,   1,   1,   1  ],  # y positive, high speed       8
        [ 0.5,  0,   0,   0.5],  # quadrant 2, low speed       9
        [ 0.5, -0.5, -0.5,  0.5],  # x negative, low speed     10
        [ 0,  -0.5, -0.5,  0  ],  # quadrant 3, low speed      11
        [-0.5, -0.5, -0.5, -0.5],  # y negative, low speed     12
        [-0.5,  0,   0,  -0.5],  # quadrant 4, low speed       13
        [-0.5,  0.5,  0.5, -0.5],  # x positive, low speed     14
        [ 0,   0.5,  0.5,  0  ],  # quadrant 1, low speed      15
        [ 0.5,  0.5,  0.5,  0.5],  # y positive, low speed     16
        [ 0.5,  -0.5,  0.5, -0.5],  # CCW rotation, high speed 17
        [-0.5,   0.5, -0.5,  0.5],  # CW rotation, high speed  18
    ])
        # Swap the last two columns. In the article, first two columns stand for RF and LF wheels,
    #  but in our model it's vice-versa
    action_space[:, [2, 3]] = action_space[:, [3, 2]]
    action_space = -1. * action_space 
    action_space = 1. * action_space
    i = 0

    n_states = 11
    n_actions = len(action_space)

    # for action in action_space:
    action = action_space[14]
    episode_over = False
    i = 0
    print(action)
    traj = []
    while not episode_over:
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        # action = action_space[np.random.choice(action_space.shape[0])]
        # action = -1. * np.array([-0.5, -0.5, -0.5, -0.5])  # y positive, low speed
        observation, reward, terminated, truncated, info = env.step(action)
        x, y, v_x, v_y, omega_z, x_t, y_t, e_x, e_y, E_dist, e_phi_deg = observation
        print("e_phi_deg:" + str(e_phi_deg))
        traj.append([x ,y])
        i+=1
        episode_over = terminated or terminated
    traj = np.array(traj)
    # Create figure
    plt.figure(figsize=(8, 6))
    # Plot trajectory as a connected line
    plt.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', marker='o', markersize=5, alpha=0.7, label="Trajectory")
    # Highlight the last coordinate
    plt.scatter(coords[0][0], coords[0][1], color='green', edgecolors='black', s=100, label="Start Point", zorder=3)
    plt.scatter(coords[1][0], coords[1][1], color='green', edgecolors='black', s=100, label="Target Point", zorder=3)
    # Add the circle
    circle = plt.Circle((0,0), env_boudary_radius, color='red', fill=False, linewidth=2, linestyle='dashed')
    plt.gca().add_patch(circle)  # Add circle to the current plot
    # Labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Trajectory Plot with Last Point Highlighted")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.8)  # X-axis reference line
    plt.axvline(0, color='black', linewidth=0.8)  # Y-axis reference line
    # Set equal aspect ratio to ensure the circle looks correct
    plt.axis("equal")
    # Add legend
    plt.legend()
    # Show plot
    plt.show()

    env.reset()

    env.close()


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

