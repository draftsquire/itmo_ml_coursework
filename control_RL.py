"""
@author: Ivan K.

@brief: File contating functions with RL conrol functions
 for mecanum-wheeled robot

"""

import numpy as np
import mujoco
from traj import round_shaped_traj, point_traj, rotate_traj, rotate_traj_2, unpack_traj, z_traj
from scipy import integrate
from scipy.spatial.transform import Rotation

# Constants for our mecanum platform
# l, w, h = .4, .2, .05
# wR = 0.04

d_x = .2 / 2
d_y = .4 / 2
R = 0.04 # Wheel radius, m

N = np.array([[-1, 1, -1, 1],
               [1, 1, 1, 1],
               [1.0 / (d_x + d_y), -1.0 /(d_x + d_y), -1.0 / (d_x + d_y), 1.0 / (d_x + d_y)]])

def Rot(phi):
    """Form rotation matrix to transform from local frame to global

    Args:
        phi (float): rotation angle
    Returns:
        (np.ndarray): 3x3 rotation matrix    
    """
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])


def h(omega_g):
    """Form transform matrix representing transition between config-space speeds and workspace speeds

    Args:
        phi_g (float): Rotation angle in global coordinate frame

    Returns:
        ndarray : matrix h
    """
    # omega = omega_g + np.pi / 4
    omega = omega_g + np.pi/4
    h = np.sqrt(2) * np.array([[-np.sin(omega), np.cos(omega), -np.sin(omega), np.cos(omega)],
                               [np.cos(omega), np.sin(omega), np.cos(omega), np.sin(omega)],
                               [1.0 /(np.cos(omega) * (d_x + d_y)), -1.0 /(np.cos(omega) *  (d_x + d_y)), -1.0 /(np.cos(omega) *  (d_x + d_y)), 1.0 / (np.cos(omega) * (d_x + d_y))]])

    return h

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


# define control function
def ctrl_f(t, data):
    """
    Simple control function

    Args:
        t (float): timestamp, seconds(?)
        data (mjData): simulation data

    Returns:
        ndArray: Inputs for each wheel (u(t))
    """
    return np.array([np.sin(6*t),-np.sin(6*t),np.sin(6*t),-np.sin(6*t)])*0.2


class Integrator:
    def __init__(self, initial_time, initial_value):
        self.current_time = initial_time
        self.current_value = np.array(initial_value, dtype=float).flatten()
        self.integrated_value = initial_value

    def update(self, new_time, input_value):
        """
        Update the integration using solve_ivp for the new time interval.
        
        :param new_time: New timestamp
        :param input_value: Current value of the signal
        :return: Updated integrated value
        """
        # Define the differential equation: dy/dt = u(t)
        def integrator(t, y):
            return input_value  # Assume input_value is constant over this small interval
        
        # Solve the differential equation for this small interval
        solution = integrate.solve_ivp(
            fun=integrator,
            t_span=(self.current_time, new_time),
            y0=[self.integrated_value],
            method='RK45'
        )
        
        # Update the integrated value and time
        self.integrated_value = solution.y[0][-1]
        self.current_time = new_time
        
        return self.integrated_value
    
class Integrator3D:
    def __init__(self, initial_time, initial_value):
        """_summary_

        Args:
            initial_time (float): timestamp
            initial_value (np.ndarray): 3x1 vector
        """
        self.current_time = initial_time
        self.current_value = initial_value
        self.integrated_value = initial_value

    def update(self, new_time, input_value):
        """
        Update the integration using solve_ivp for the new time interval.
        
        :param new_time: New timestamp
        :param input_value (np.ndarray): Current value of the signal vector [x, y, z]
        :return: Updated integrated value
        """
        # Define the differential equation: dy/dt = u(t)
        def integrator_1(t, y):
            return input_value[0]  # Assume input_value is constant over this small interval
        def integrator_2(t, y):
            return input_value[1]  # Assume input_value is constant over this small interval
        def integrator_3(t, y):
            return input_value[2]  # Assume input_value is constant over this small interval
        
        # Solve the differential equation for this small interval
        solution_1 = integrate.solve_ivp(
            fun=integrator_1,
            t_span=(self.current_time, new_time),
            y0=[self.integrated_value[0]],
            method='RK45'
        )
        solution_2 = integrate.solve_ivp(
            fun=integrator_2,
            t_span=(self.current_time, new_time),
            y0=[self.integrated_value[1]],
            method='RK45'
        )
        solution_3 = integrate.solve_ivp(
            fun=integrator_3,
            t_span=(self.current_time, new_time),
            y0=[self.integrated_value[2]],
            method='RK45'
        )
        
        # Update the integrated value and time
        self.integrated_value = np.array([solution_1.y[0][-1], solution_2.y[0][-1], solution_3.y[0][-1]])
        self.current_time = new_time
        self.current_value = self.integrated_value
        return self.integrated_value    

class Summator3:
    def __init__(self):
        self._memory = np.zeros(3)

    def add(self, value):
        self._memory = self._memory + value

    def get(self):
        return self._memory

class DerivativeCalculator:
    def __init__(self, init_value=0.0):
        self.previous_time = None
        self.previous_value = None
        self.derivative = init_value

    def update(self, current_time, current_value):
        """
        Calculate the derivative based on the current value and timestamp.
        
        :param current_time: Current timestamp
        :param current_value: Current value of the signal
        :return: Calculated derivative
        """
        if self.previous_time is not None:
            # Compute the time difference
            dt = current_time - self.previous_time
            
            if dt > 0:  # Avoid division by zero
                # Compute the derivative
                self.derivative = (current_value - self.previous_value) / dt
        
        # Update previous time and value
        self.previous_time = current_time
        self.previous_value = current_value
        
        return self.derivative
    
def saturate_control(u_q, max_torque):
    """Saturate control input in configurational space to [-max_torque; +maxtorque]

    Args:
        u_q (np.ndarray): 4x1 vector of control
        max_torque (float): maximum torque value

    Returns:
        np.ndarray: 4x1 vector of saturated control
    """
    return np.clip(u_q, -max_torque, max_torque)
        
def calculate_mz(u):
    """Calculate rotational moment Mz from control in config space

    Args:
        u (np.ndarray): 4x1 vector

    Returns:
        Scalar value of rotaion around Z-axis
    """
    # Calculate forces Fi based on torques Mi and wheel radius R
    F1, F2, F3, F4 = u / R

    # Calculate Mz
    Mz = (d_x / np.sqrt(2)) * (F1 - F2 - F3 + F4) + (d_y / np.sqrt(2)) * (F1 - F2 - F3 + F4)

    return Mz



def tau_body2wheel(tau_body):
    """Convert forces in workspace to config space

    Args:
        tau_body (_type_): _description_

    Returns:
        _type_: _description_
    """
    d_x = .2 / 2
    d_y = .4 / 2

    R = 0.04
    N = np.array([[-1, 1, -1, 1],
                    [1, 1, 1, 1],
                    [1.0 / (d_x + d_y), -1.0 /(d_x + d_y), -1.0 / (d_x + d_y), 1.0 / (d_x + d_y)]])


    Jspeed_w2b = (R/4) * N
    Jeffort_b2w = Jspeed_w2b.T

    #this version below works worse, discarding angle error
    # Jspeed_b2w = (1/R) * N.T
    # Jeffort_w2b = Jspeed_b2w.T
    # Jeffort_b2w = np.linalg.pinv(Jeffort_w2b)

    return -Jeffort_b2w @ tau_body

####################### PID Control
def ctrl_pid(t, model, data, trajectory, init_pos):
    """
    PID control function

    Args:
        t (float): timestamp, seconds(?)
        model(mjModel): Model object
        data (mjData): simulation data
        trajectory (fctn(t, init_pos)) - trajectory function
        init_pos - initial position 3x1 vector

    Returns:
        ndArray: Inputs for each wheel (u(t))
    """

    #TRAJECTORIES HERE
    X_g_des = trajectory(t, init_pos)
    #desired position vector in 
    if not hasattr(ctrl_pid, "ws_integrator"):
        ctrl_pid.ws_integrator = Integrator3D(0, np.array(init_pos))  # it doesn't exist yet, so initialize it

    if not hasattr(ctrl_pid, "error_ws_derivative"):
        ctrl_pid.error_ws_derivative = DerivativeCalculator()  # it doesn't exist yet, so initialize it   
      

    
    # X_g_des = np.array([x_des, y_des, theta_des])


    v_x_b_act, v_y_b_act = data.sensordata[0], data.sensordata[1]
    #Robot's rotation speed
    omega_z = data.sensordata[5]

    x_axis_data = np.array([data.sensordata[6], data.sensordata[7], data.sensordata[8]])
    phi = calculate_phi_from_loc_x_axis(x_axis_data);

    #Speeds in local frame
    X_b_dot_act = np.array([v_x_b_act, v_y_b_act, omega_z])
    # #Speeds in global frame
    X_g_dot_act = np.dot(Rot(phi), X_b_dot_act)
    
    # X_g_act = ctrl_pid.ws_integrator.update(t, X_g_dot_act)
    X_g_act = np.array([data.sensordata[9], data.sensordata[10], phi])
    
    # X_g_des и X_g_act разных типов?
    error_ws = X_g_des - X_g_act

    # Wrap error to the range [-pi, pi]
    error_ws[2] = (error_ws[2] + np.pi) % (2 * np.pi) - np.pi

    
    if not hasattr(ctrl_pid, "error_ws_integrator"):
        ctrl_pid.error_ws_integrator = Integrator3D(0, error_ws)  # it doesn't exist yet, so initialize it    

    maxtorque = 1000

    PID_Kp_x = 50
    PID_Ki_x = 5
    PID_Kd_x = 10

    PID_Kp_y = 50
    PID_Ki_y = 5
    PID_Kd_y = 10

    PID_Kp_phi = 5
    PID_Ki_phi = 0.07
    PID_Kd_phi = 0.8
    P = np.array([PID_Kp_x, PID_Kp_y, PID_Kp_phi])
    I = np.array([PID_Ki_x, PID_Ki_y, PID_Ki_phi])
    D = np.array([PID_Kd_x, PID_Kd_y, PID_Kd_phi])

    proportional =  P * error_ws
    integral  = I * ctrl_pid.error_ws_integrator.update(t, error_ws)
    differential = D * ctrl_pid.error_ws_derivative.update(t, error_ws)
    u_ws = proportional + integral + differential

    rot_z_inv = np.linalg.inv(Rot(phi))
    u_ws_b = np.dot(rot_z_inv, u_ws)
    u_cs = np.dot(((R/4) * N).T, u_ws_b)

    u_sat = saturate_control(u_cs, maxtorque)
    # print("error:%2f P:%.2f I:%.4f  D:%.4f Mz: %.2f" %
    #        (error_ws[2], proportional[2], integral[2], differential[2], Mz))

    return -u_sat

    

############################ Feedback Linearization

def skew(vec):
    '''
    Converts a 3 vector to a 3x3 skew symmetric matrix
    '''
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def bodytau_CTC(model, data, qdes, dqdes, ddqdes, integrator=None, use_compensation=1):    
    bodydofs=[0,1,3]
    bodyqpos=[0,1,3]

    phi = data.qpos[bodyqpos[-1]]
    Rb2g = Rotation.from_euler('z', phi, degrees=False).as_matrix()

    omega_vec = np.array([0,0,data.qvel[bodydofs[-1]]])
    Rdot_b2g = skew(omega_vec) @ Rb2g #for pi/3 better with .T
    rot_compensation = Rdot_b2g @ (Rb2g.T @ data.qvel[bodydofs])
    # without compensation it wont stabilize to 1,1,pi/2

    e = np.asarray([*data.sensordata[:2],phi])-qdes #
    e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))
    de = data.qvel[bodydofs] - Rb2g @ dqdes #TODO remember to move both vels to the same frame, qvel now is in global
    inte = 0
    if integrator is not None:
        integrator.add(e*model.opt.timestep)
        inte = integrator.get()
        print(inte)
    
    anti_windup = .7
    # kp, kd, ki = np.diag([20,20,160/24]), np.diag([9,9,56/9]), np.diag([0,0,0]) # for saturation 0.4
    # kp, kd, ki = np.diag([35,35,160/24]), np.diag([5,5,56/9]), np.diag([0,0,2]) # for saturation 0.4 and circle with fixed angle
    kp, kd, ki = np.diag([35,35,4]), np.diag([9,9,.4]), np.diag([1,1,.3]) # for saturation 0.4 and circle with changing angle
    # kp, kd, ki = np.diag([5,5,1]), np.diag([4,4,0]), np.diag([.7,.7,.3]) #windup .8
    # kp, kd, ki = np.diag([5,5,1]), np.diag([0,0,0]), np.diag([.9,.9,.3])
    u = np.zeros(model.nv)
    u[bodydofs] = Rb2g.T @ (ddqdes - kp@e - kd@de - np.clip(ki@inte,-anti_windup,anti_windup) - int(use_compensation)*rot_compensation)
    
    Mu = np.empty(model.nv)
    mujoco.mj_mulM(model, data, Mu, u)
    tau = Mu + data.qfrc_bias
    tau = tau[bodydofs]
    
    return tau

def ctrl_bodyCTC(t, model, data, trajectory, init_pos=[0,0,0], integrator=None):
    """Control by Feedback linearization

    Args:
        t (float): timestamp, seconds(?)
        model(mjModel): Model object
        data (mjData): simulation data
        trajectory (fctn(t, init_pos)) - trajectory function
        init_pos - initial position 3x1 vector

    Returns:
        _type_: _description_
    """
    qdes = np.array([6,1,np.pi/2]) #x, y, phi
    dqdes = np.zeros(3) #TODO you can set it as input vel
    ddqdes = np.zeros(3)
    if trajectory is not None:
        traj = trajectory(t, init_pos)
        qdes, dqdes, ddqdes = unpack_traj(traj)
    
    tau_body = bodytau_CTC(model, data, qdes, dqdes, ddqdes, integrator)
    tau_wheels = tau_body2wheel(tau_body)
    return np.asarray(tau_wheels)


def OGRF(max_coordinate, start_point : tuple, target_point : tuple, current_coord : tuple, eo_norm, r, step, c):
    """Orientation Gain Reward Function

    Args:
        max_coordinate (_type_): maximum coordinate
        start_point (tuple): start point coordinate
        target_point (tuple): target point coordinate
        current_coord (tuple): instantaneous robot’s coordinate
        eo_norm (_type_): normalized orientation error
        r (_type_): environment boundary radius
        step (_type_): number of performed steps
        c (_type_): the orientation error constraint 

    Returns:
        T: A Boolean indicating whether an episode is
    finished
        R_t: Ttotal reward obtained by the agent
    """
    T = False
    R_t = 0

    x, y = current_coord
    x_s, t_s = start_point
    x_t, y_t = target_point
    

    return T, R_t
