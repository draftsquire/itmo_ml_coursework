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

def show_state(environment, episode=0, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(environment.render())#mode='rgb_array'))
    plt.title("%s | Eposide: %d | Step: %d %s" % ('Mecanum platform', episode, step, info))
    plt.axis('off')


class DQN_QR(torch.nn.Module):
    def __init__(self, len_state, num_quant, num_actions, hidden_dim=256, lr=1e-3):
        torch.nn.Module.__init__(self)

        self.num_quant = num_quant
        self.num_actions = num_actions

        self.model=torch.nn.Sequential(
            torch.nn.Linear(len_state, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, num_actions*num_quant)
        )
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr)

    def forward(self, x):
        y = self.model(x)
        return y.view(-1, self.num_actions, self.num_quant)

    def select_action(self, state, eps, device="cuda", training_started=True):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state])
        state = state.to(device)

        if not training_started:
            action = torch.randint(0, self.num_actions, (1,), device=device)
            return int(action)

        if (random.random() > eps):
            action = self.forward(state).mean(2).max(1)[1]
        else:
            action = torch.randint(0, self.num_actions, (1,), device=device)

        return int(action)



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

    MAX_EPISODE_STEPS = 6600
    MAX_LEARNING_STEPS = 1600000
    EPISODES_MAX = int(MAX_LEARNING_STEPS / MAX_EPISODE_STEPS)


    register(
        id="Mecanum-v0",
        entry_point="gym_env_mecanum:MecanumEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=6000,
    )
    env_boudary_radius = 3.
    coordinates = np.array([[0, -0.5],
                           [0, 0],
                           [-0.5, -1.],
                           [-1., 0.],
                           [-0.5, 0.5],
                           [-1, 1.],
                           [0., 1.],
                           [0.5, 0.5],
                           [1., 1.],
                           [1., 0.],
                           [0.5, -1.]])
    # coords = generate_coordinates(10, x_range=(-1, 1), y_range=(-1, 1))
    env = gym.make('Mecanum-v0', render_mode="human", coordinates=coordinates[:2], max_steps=MAX_EPISODE_STEPS, camera_config=CAMERA_CONFIG, environment_boundary_radius=env_boudary_radius)
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
        [ 0.5,  -0.5,  0.5, -0.5],  # CCW rotation, high speed
        [-0.5,   0.5, -0.5,  0.5],  # CW rotation, high speed
    ])
        # Swap the last two columns. In the article, first two columns stand for RF and LF wheels,
    #  but in our model it's vice-versa
    action_space = action_space
    action_space[:, [2, 3]] = action_space[:, [3, 2]]
    action_space = -1. * action_space 

    i = 0

    n_states = 11
    n_actions = len(action_space)

    # Learning Parameters
    eps_start, eps_end, eps_dec = 1.0, 0.05, EPISODES_MAX
    eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)


    Z = DQN_QR(len_state=n_states, num_quant=110, num_actions=n_actions, hidden_dim=256)
    state_dict = torch.load('dqn_checkpoint_copy.pth')['model_state_dict']
    print(state_dict.keys())
    Z.load_state_dict(state_dict)
    Z.eval()

    running_reward=None
    i = 1
    
    # for episode in range(len(coords)): 
    traj = []
    sum_reward = 0
    state, _info = env.reset()
    x_s, y_s, v_x, v_y, omega_z, x_t, y_t, e_x, e_y, E_dist, e_phi_deg = state
    
    traj.append([x_s, y_s])
    
    steps_local = 0
    while True:

        steps_local += 1
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state])
        state = state.to(device)
        action = Z.forward(state).mean(2).max(1)[1]

        state, reward, terminated, truncated, info = env.step(action_space[action])
        done = (terminated or truncated)
        if done:
            break
        sum_reward += reward
        x, y, v_x, v_y, omega_z, x_t, y_t, e_x, e_y, E_dist, e_phi_deg = state
        traj.append([x ,y])
        if steps_local % 600 == 0:
            i += 1
            if i>=11:
                break
            else:
                env.unwrapped.set_target((x, y), (coordinates[i][0],coordinates[i][1]))
            
    print("Total steps: " + str(i))    
    traj = np.array(traj)
    # Create figure
    plt.figure(figsize=(8, 6))
    # Plot trajectory as a connected line
    plt.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', marker='o', markersize=1, alpha=0.7, label="Robot's Path")
    # Highlight the last coordinate
    plt.scatter(x_s, y_s, color='green', edgecolors='black', s=100, label="Start Point", zorder=3)
    coordinates[[0, 1]] = coordinates[[1, 0]]
    plt.plot(coordinates[:, 0], coordinates[:, 1], linestyle='--', color='red', marker='o', markersize=3, alpha=0.7, label="Target Path")
    # Labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.8)  # X-axis reference line
    plt.axvline(0, color='black', linewidth=0.8)  # Y-axis reference line
    # Set equal aspect ratio to ensure the circle looks correct
    plt.axis("equal")
    # Add legend
    plt.legend()
    # Show plot
    plt.show()           

    env.close()                
                

    print("done")
