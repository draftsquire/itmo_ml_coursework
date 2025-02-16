import numpy as np
import matplotlib.pyplot as plt

def round_shaped_traj_2(t, init_pos):
    """
    Calculate the round-shaped trajectory at time t.
    
    Parameters:
    - t: float, the time input.
    - center: coordinates of trajectory start
    - th0: pi/2 - робот направлен всегда вдоль траектории,
      0 или pi - перпендикулярно
    Returns:
    - x: float, x-coordinate of the trajectory.
    - y: float, y-coordinate of the trajectory.
    - theta: float, orientation angle of the trajectory.
    """
    # Initial values
    x0 = init_pos[0]
    y0 = init_pos[1]
    th0 = init_pos[2]

    delta = 5 * np.pi / 2

    t1 = delta
    tline = 7
    t2 = np.pi / 2

    slow = 1
    r1 = 1
    r2 = 11
    alpha = 5 * np.pi / 6
    ang = np.pi / 6  # from alpha
    k = np.tan(ang)

    # Rotation matrix
    R = np.array([
        [np.cos(ang), -np.sin(ang)],
        [np.sin(ang), np.cos(ang)]
    ])

    k_th1 = delta / t1
    k_th2 = 1
    theta_line = th0 + delta - alpha

    # Line end and arc2 end calculation
    line_end = np.array([
        -tline / slow + r1 + x0,
        -k * tline / slow - r1 + y0
    ])
    arc2_end = R @ np.array([
        -r2 * np.sin(t2),
        r2 * np.cos(t2) - r2
    ]) + line_end

    # Determine trajectory based on time
    x = -r1 * np.cos(t) + r1 + x0
    y = -r1 * np.sin(t) + y0
    theta = k_th1 * t + th0



    return np.array([x, y, theta])

def round_shaped_traj(t, init_pos):
    """
    Calculate the round-shaped trajectory at time t.
    
    Parameters:
    - t: float, the time input.
    - center: coordinates of trajectory start
    - th0: pi/2 - робот направлен всегда вдоль траектории,
      0 или pi - перпендикулярно
    Returns:
    - x: float, x-coordinate of the trajectory.
    - y: float, y-coordinate of the trajectory.
    - theta: float, orientation angle of the trajectory.
    """
    # Initial values
    x0 = init_pos[0]
    y0 = init_pos[1]
    th0 = init_pos[2]

    delta = 5 * np.pi / 2

    t1 = delta
    tline = 7
    t2 = np.pi / 2

    slow = 1
    r1 = 1
    r2 = 11
    alpha = 5 * np.pi / 6
    ang = np.pi / 6  # from alpha
    k = np.tan(ang)

    # Rotation matrix
    R = np.array([
        [np.cos(ang), -np.sin(ang)],
        [np.sin(ang), np.cos(ang)]
    ])

    k_th1 = delta / t1
    k_th2 = 1
    theta_line = th0 + delta - alpha

    # Line end and arc2 end calculation
    line_end = np.array([
        -tline / slow + r1 + x0,
        -k * tline / slow - r1 + y0
    ])
    arc2_end = R @ np.array([
        -r2 * np.sin(t2),
        r2 * np.cos(t2) - r2
    ]) + line_end

    # Determine trajectory based on time
    x = -r1 * np.cos(t) + r1 + x0
    y = -r1 * np.sin(t) + y0
    # theta = k_th1 * t + th0
    theta = th0
    # if t < t1:
    #     x = -r1 * np.cos(t) + r1 + x0
    #     y = -r1 * np.sin(t) + y0
    #     theta = k_th1 * t + th0
    # elif t < t1 + tline:
    #     p = t - t1
    #     x = -p / slow + r1 + x0
    #     y = -k * p / slow - r1 + y0
    #     theta = theta_line
    # elif t < t1 + tline + t2:
    #     p = t - t1 - tline
    #     xbuf = -r2 * np.sin(p)
    #     ybuf = r2 * np.cos(p) - r2
    #     rot = R @ np.array([xbuf, ybuf]) + line_end
    #     x = rot[0]
    #     y = rot[1]
    #     theta = k_th2 * p + theta_line
    # else:
    #     x = arc2_end[0]
    #     y = arc2_end[1]
    #     theta = t2 + theta_line

    return np.array([x, y, theta])

def point_traj(t, init_pos):
    theta = init_pos[2]
    x = init_pos[0] + 1
    y = init_pos[1] + 1


    return np.array([x, y, theta])


def z_traj(t, init_pos):
    theta = init_pos[2]
    if t<5:
        x = init_pos[0] + 1
        y = init_pos[1] 
    elif t>=5 and t <10:
        x = init_pos[0]
        y = init_pos[1] - 1 
    else:
        x = init_pos[0] + 1
        y = init_pos[1] - 1 


    return np.array([x, y, theta])

def rotate_traj(t, init_pos):
    theta = init_pos[2]
    x = init_pos[0]
    y = init_pos[1]

    return np.array([x, y, theta])

def rotate_traj_2(t, init_pos):
    if t == None:
        t = 0.0
    theta = np.pi
    x = init_pos[0]
    y = init_pos[1]
    if t < 8:
        return np.array([x, y, theta])
    else:
        transition = min((t - 8) / 2, 1)  # Gradual transition over 2 seconds
        return np.array([x, y, theta * (1 - transition)])

def lissajious_traj(t, init_pos, a=0.5, b=0.5, delta=np.pi/2, omega_a = 2, omega_b = 1, theta_rate=0.1):
    """
    Generates a Lissajous-shaped trajectory.

    Args:
        t (float): Time variable.
        init_pos (list or np.ndarray): Initial position as [x0, y0, theta0].
        a (float, optional): Amplitude in the x-direction. Defaults to 1.
        b (float, optional): Amplitude in the y-direction. Defaults to 1.
        delta (float, optional): Phase difference between x and y. Defaults to π/2.
        omega_x (float, optional): Angular frequency for x. Defaults to 1.
        omega_y (float, optional): Angular frequency for y. Defaults to 1.
        theta_rate (float, optional): Rate of change of theta over time. Defaults to 0.1 rad/s.

    Returns:
        np.ndarray: Array containing the [x, y, theta] values at time t.
    """
    # Initial position
    x0, y0, theta0 = init_pos

    # Lissajous x and y trajectory
    x = x0 + a * np.sin(omega_a * t) 
    y = y0 + b * np.sin(omega_b * t + delta)

    # Theta update based on constant rate
    theta = theta0 #+ theta_rate * t

    return np.array([x, y, theta])

def unpack_traj(traj_result):
    # qdes = np.array([6,1,np.pi/2]) #x, y, phi
    qdes = np.zeros(3) #x, y, phi
    dqdes = np.zeros(3) #TODO you can set it as input vel
    ddqdes = np.zeros(3)
    if isinstance(traj_result, tuple):
        if len(traj_result) == 1:
            qdes = traj_result[0]
        elif len(traj_result) == 2:
            qdes, dqdes = traj_result
        else:
            qdes, dqdes, ddqdes = traj_result
    else:
        qdes = traj_result
    return qdes, dqdes, ddqdes
        
    
