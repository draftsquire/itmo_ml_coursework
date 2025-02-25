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
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, num_actions*num_quant)
        )
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr)
    def forward(self, x):
        y = self.model(x)
        return y.view(-1, self.num_actions, self.num_quant)
    
    def select_action(self, state, eps, device="cpu", training_started=True):
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


    EPISODES_MAX = 10000000
    EPISODES_N = EPISODES_MAX
    STEPS_MAX = 10000
    register(
        id="Mecanum-v0",
        entry_point="gym_env_mecanum:MecanumEnv",
        max_episode_steps=STEPS_MAX,
        reward_threshold=4600,
    )
    env_boudary_radius = 500.
    coords = generate_coordinates(10000)
    env = gym.make('Mecanum-v0',coordinates=coords, max_steps=STEPS_MAX, camera_config=CAMERA_CONFIG, environment_boundary_radius=env_boudary_radius)
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
    eps_start, eps_end, eps_dec = 1.0, 0.05, EPISODES_N 
    eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)

    memory = ReplayMemory(6e+6)
    logger = Logger('q-net', fmt={'loss': '.5f'})

    Z = DQN_QR(len_state=n_states, num_quant=10, num_actions=n_actions, hidden_dim=256)
    Ztgt = DQN_QR(len_state=n_states, num_quant=10, num_actions=n_actions, hidden_dim=256)
    optimizer = torch.optim.Adam(Z.parameters(), 1e-3) #5-e5
    running_reward=None
    steps_done = 0
    gamma, batch_size = 0.99, 32 
    tau = torch.Tensor((2 * np.arange(Z.num_quant) + 1) / (2.0 * Z.num_quant)).view(1, -1)
    LEARNING_STARTS = 2e+5
    training_started = False
    for episode in range(EPISODES_MAX): 
        # print("episode #" + str(episode))
        sum_reward = 0
        state, _info = env.reset()
        steps_local = 0
        if episode % 1000 == 0:
            print(str(episode) + " " + str(steps_done))
        while True:
                steps_done += 1
                steps_local += 1
                action = Z.select_action(torch.Tensor(np.array(state)), eps(steps_done))
                next_state, reward, terminated, truncated, info = env.step(action_space[action])
                done = terminated or truncated
                memory.push(state, action, next_state, reward, float(done))
                sum_reward += reward

                if (steps_done < LEARNING_STARTS) or (len(memory) < batch_size):
                    if done:
                        running_reward = sum_reward
                        break
                    else:
                        state = next_state  # Ensure to update state even when not learning
                        continue
                else:
                    training_started = True            
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                theta = Z(states)[np.arange(batch_size), actions]
                
                Znext = Ztgt(next_states).detach()
                Znext_max = Znext[np.arange(batch_size), Znext.mean(2).max(1)[1]]
                Ttheta = rewards + gamma * (1 - dones) * Znext_max
                
                diff = Ttheta.t().unsqueeze(-1) - theta 
                loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
                
                if steps_done % 1000 == 0: # Target Network update interval
                    Ztgt.load_state_dict(Z.state_dict())

                if done and episode % 50 == 0:
                    logger.add(episode, steps=steps_local, running_reward=running_reward, loss=loss.data.numpy())
                    logger.iter_info()

                if done: 
                    running_reward = sum_reward
                    break
    


                    
    env.close()                
                

    print("done")
