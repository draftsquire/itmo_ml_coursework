from typing import Dict, Tuple, Union
import os

import mujoco
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from scipy.spatial.transform import Rotation

DEFAULT_CAMERA_CONFIG = {
        "type": 1,
        "trackbodyid": 1,
        "distance": 2.,
        "lookat": np.array((0.0, 0.0, 1.15)),
        "elevation": -50.0,
    }

def calculate_phi_from_loc_x_axis(x_axis):
    """Calculates phi angle (robot's rotation) from unit vector of robot's local X_axis

    Args:
        x_axis (numpy.ndarray):  unit vector from x-axis sensor, [x, y, z]

    Returns:
        numpy.ndaray: counter-clockwise rotation between robot's X and global frame X [-pi; pi]
    """
    # Convert to NumPy array if not already
    
    np.array(x_axis, dtype=float)
    # Use atan2 to calculate angle directly
    angle_rad = np.arctan2(x_axis[1], x_axis[0])
    
    return angle_rad


class MecanumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is made to adapt Muecanum-wheeled robot model and simulation for use with RL algotithms
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        mecanum_xml: str =  os.path.join(os.getcwd(),"mecanum.xml"),
        camera_config=DEFAULT_CAMERA_CONFIG,
        init_pos=(0., 0.),
        reset_noise_scale=1.,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            reset_noise_scale,
            # forward_reward_weight,
            # ctrl_cost_weight,
            # healthy_reward,
            # terminate_when_unhealthy,
            # healthy_state_range,
            # healthy_z_range,
            # healthy_angle_range,
            # reset_noise_scale,
            # exclude_current_positions_from_observation,
            **kwargs,
        )

        self._camera_config = camera_config
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
            )
        MujocoEnv.__init__(
            self,
            mecanum_xml,
            4,
            observation_space=observation_space,
            default_camera_config=self._camera_config,
            **kwargs,
        )

    @property
    def terminated(self):
        return False

    def _get_obs(self):
        # without compensation it wont stabilize to 1,1,pi/2
        x, y = self.data.sensordata[0], self.data.sensordata[1]
        v_x, v_y = self.data.sensordata[3], self.data.sensordata[4]
        omega_z = self.data.sensordata[8]
        x_axis_data = np.array([self.data.sensordata[9], self.data.sensordata[10], self.data.sensordata[11]])
        phi = calculate_phi_from_loc_x_axis(x_axis_data);
        # e_x, e_y = 
        # E_dist =
        # e_phi = 
        # position = self.data.qpos.flat.copy()
        # velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)
        observation = np.array([x, y, v_x, v_y, omega_z, phi])
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # ctrl_cost = self.control_cost(action)

        # healthy_reward = self.healthy_reward

        rewards = 0 #+ healthy_reward
        # costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos
        qpos[0] = self.init_qpos[0] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=1)#X-coordinate
        qpos[1] = self.init_qpos[1] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=1)#Y-coordinate
        qpos[2] = self.init_qpos[2] # Z-coordinate remains zero (likely)
        
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in self._camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)