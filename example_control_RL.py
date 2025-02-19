import numpy as np
import matplotlib.pyplot as plt
import mujoco  # Add back mujoco import

from mecanum_sim import SensorOutput, SimHandler
from mecanum_gen import generate_scene_empty
from control_RL import Summator3, ctrl_bodyCTC
from traj import point_traj, unpack_traj, rotate_traj_2, z_traj, rotate_traj, round_shaped_traj, round_shaped_traj_2

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.envs.registration import register

GENERATE_SCENE = True
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

    STEPS_MAX = 150
    register(
        id="Mecanum-v0",
        entry_point="gym_env_mecanum:MecanumEnv",
        max_episode_steps=STEPS_MAX,
        reward_threshold=1.76*300,
    )

    env = gym.make('Mecanum-v0', camera_config=CAMERA_CONFIG, render_mode="rgb_array", width=1280, height=720)
    env = RecordVideo(env, video_folder="mecanum-platform", name_prefix="eval",
                  episode_trigger=lambda x: True)
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
    # for action in action_space[-1:] :
    action = action_space[14] # x positive, low speed
    print(action)
    i = 0
    episode_over = False
    env.reset()
    while not episode_over:
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        # action = action_space[np.random.choice(action_space.shape[0])]
        # action = -1. * np.array([-0.5, -0.5, -0.5, -0.5])  # y positive, low speed
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation)
        i+=1
        episode_over = (i >= STEPS_MAX)
        

    env.close()