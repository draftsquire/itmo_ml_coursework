from typing import Dict, Tuple, Union
import os

import numpy as np
import math

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Discrete

from scipy.spatial.transform import Rotation

DEFAULT_CAMERA_CONFIG = {
        "type": 1,
        "trackbodyid": 1,
        "distance": 2.,
        "lookat": np.array((0.0, 0.0, 1.15)),
        "elevation": -50.0,
    }

def calculate_phi_from_loc_x_axis_deg(x_axis):
    """Calculates phi angle (robot's rotation) from unit vector of robot's local X_axis

    Args:
        x_axis (numpy.ndarray or list): Unit vector from x-axis sensor, [x, y, z]

    Returns:
        float: Rotation between robot's X and global frame X in range [-180, 180] degrees.
    """
    # Convert to NumPy array if not already
    x_axis = np.array(x_axis, dtype=float)

    # Use atan2 to calculate angle in radians
    angle_rad = np.arctan2(x_axis[1], x_axis[0])

    # Convert negative angles to positive by adding 360 degrees
    angle_deg = np.degrees(angle_rad)
    # if angle_deg < 0:
    #     angle_deg += 360

    return angle_deg

def calculate_phi_from_loc_x_axis_rad(x_axis):
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
    # Convert negative angles to positive by adding 2π
    # if angle_rad < 0:
    #     angle_rad += 2 * np.pi

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
    DEFAULT_MAX_STEPS=1500
    def __init__(
        self,
        coordinates, # Used for starting and target points
        environment_boundary_radius,
        mecanum_xml: str =  os.path.join(os.getcwd(),"mecanum.xml"),
        max_steps=DEFAULT_MAX_STEPS,
        camera_config=DEFAULT_CAMERA_CONFIG,
        reset_noise_scale=1.,
        frame_skip: int = 4,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            reset_noise_scale,
            coordinates,
            max_steps,
            reset_noise_scale,
            environment_boundary_radius,
            **kwargs,
        )

        self._max_steps = max_steps
        self._camera_config = camera_config
        self._reset_noise_scale = reset_noise_scale # TODO: deprecated!
        self._coordinates = coordinates
        self._step_n = 0
        self._episode_n = 0
        self._environment_boundary_radius = environment_boundary_radius
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
            )
        # self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        MujocoEnv.__init__(
            self,
            mecanum_xml,
            frame_skip=frame_skip,
            observation_space=self.observation_space,
            default_camera_config=self._camera_config,
            **kwargs,
        )
        self.action_space = Discrete(19)
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale
        self._x_start = self._coordinates[0][0]
        self._y_start = self._coordinates[0][1]
        n = len(self._coordinates)
        self._x_target = self._coordinates[n - 1][0]
        self._y_target = self._coordinates[n - 1][1]
        self._phi_target = 0.


    # @property
    # def terminated(self):
    #     return False

    def _get_obs(self):
        x, y = self.data.sensordata[0], self.data.sensordata[1]
        x_t, y_t = self._x_target, self._y_target
        v_x, v_y = self.data.sensordata[3], self.data.sensordata[4]
        omega_z = self.data.sensordata[8]
        x_axis_data = np.array([self.data.sensordata[9], self.data.sensordata[10], self.data.sensordata[11]])
        phi = calculate_phi_from_loc_x_axis_deg(x_axis_data)#from 0 to 360
        e_x, e_y = x_t - x, y_t - y
        E_dist = math.dist([x, y], [x_t, y_t])
        e_phi = self._phi_target - phi

        observation = np.array([x, y, v_x, v_y, omega_z, x_t, y_t, e_x, e_y, E_dist, e_phi])
        return observation

    def ogrf(self, x_s, y_s, x_t, y_t, x, y, e_phi_norm, r, step, c_phi=0.4):
        """
        Orientational Gain Reward Function

        :param x_s: Start point coordinate
        :param y_s: Start point coordinate
        :param x_t: Target point coordinate
        :param y_t: Target point coordinate
        :param x: Instantaneous robot's coordinate
        :param y: Instantaneous robot's coordinate
        :param e_phi_norm: Normalized orientation error
        :param r: Environment boundary radius
        :param step: Number of performed steps
        :param c_phi: The orientation error constant

        :return: tuple: (T, R_T) - A boolean indicating whether an episode is finished,
         R_T - total reward obtained by the agent
        """
        T = False
        R_T = 0
        radius = x**2 + y**2
        R1 = np.abs(x_s - x_t) - np.abs(x - x_t)
        R2 = np.abs(y_s - y_t) - np.abs(y - y_t)
        if (radius < r**2) and (e_phi_norm < c_phi):
            R_T = (R1 + R2) * (1 - e_phi_norm)
        else:
            T = True
            steps_left = self._max_steps - step
            R_T = -steps_left 

        return T, R_T


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        # Unpack values into separate variables
        x, y, v_x, v_y, omega_z, x_t, y_t, e_x, e_y, E_dist, e_phi_deg = observation

        e_phi_norm = np.abs(e_phi_deg / 180)
        r = self._environment_boundary_radius
        
        terminated, reward =  self.ogrf(self._x_start, self._y_start, self._x_target, self._y_target, x, y, e_phi_norm, r, self._step_n)
        info = {
            "x_position": 0,
            "x_velocity": 0,
        }
        self._step_n += 1
        if self.render_mode == "human":
            self.render()

        if self._step_n >= self._max_steps:
            truncated = True
        else:
            truncated = False
        # print("rew: " + str(reward))
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """
        Generates new initial and target coordinates by running a uniform distributed number generating function which will
        add values within range of -1 to 1 to initial coordinates
        :return: observations list
        """
        n = len(self._coordinates)
        qpos = self.init_qpos
        if self._episode_n >= n:
            coordinates_index = self._episode_n % n
        else:
            coordinates_index = self._episode_n
        
        self._x_start = self._coordinates[coordinates_index][0]#X-coordinate
        self._y_start = self._coordinates[coordinates_index][1]#Y-coordinate
        qpos[0] = self._x_start
        qpos[1] = self._y_start

        qpos[2] = self.init_qpos[2] # Z-coordinate remains zero (likely)

        self._x_target = 0
        self._y_target = 0
        # self._x_target = self._coordinates[n - 1 - coordinates_index][0]
        # self._y_target = self._coordinates[n - 1 - coordinates_index][1]

        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self._step_n = 0
        self._episode_n += 1
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in self._camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)