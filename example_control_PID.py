import numpy as np
import mujoco
from typing import List, Tuple

from mecanum_gen import generate_scene, obstacle_positions
from mecanum_sim import SensorOutput, SimHandler
from control import ctrl_f, calculate_phi_from_loc_x_axis, Rot, ctrl_pid, ctrl_AITSM, Integrator, Integrator3D
from traj import rotate_traj, rotate_traj_2, point_traj, round_shaped_traj, round_shaped_traj_2, unpack_traj, z_traj
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from utils import plot_tracking



if __name__ == '__main__':


    # Define environment bounds based on the actual room size
    bounds = (-15, 15, -15, 15)  # (min_x, max_x, min_y, max_y)
    
    # Starting position
    init_pos = [-15, -15, 0]

    
    # Generate trajectory function from path with slower speed for safety
    # trajectory = generate_trajectory_from_path(path, init_pos, speed=0.2)
    trajectory = round_shaped_traj_2

    spec = generate_scene(init_pos, n_rollers=8)
    spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='gyro_c', type=mujoco.mjtSensor.mjSENS_GYRO, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='local_x_axis', type=mujoco.mjtSensor.mjSENS_FRAMEXAXIS, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='pos_c', type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.compile()
    model_xml = spec.to_xml()

    simtime = 20

    # prepare data logger
    simout = SensorOutput(sensor_names=[sen.name for sen in spec.sensors],
                          sensor_dims=[3, 3, 3, 3])

    CAMERA_CONFIG = {
        "type": 1,
        "trackbodyid": 1,
        "distance": 2.,
        "lookat": np.array((0.0, 0.0, 1.15)),
        "elevation": -50.0,
    }
    # prepare sim params
    simh = SimHandler(model_xml, None, simlength=simtime, simout=simout, cam_config=CAMERA_CONFIG)  
    
    # run MuJoCo simulation
    fin_dur = simh.simulate(is_slowed=False, control_func=ctrl_pid, control_func_args=(trajectory, init_pos,))

    # Plotting graphs
    x = np.array(simout.sensordata["pos_c"][:, 0], dtype=float)
    y = np.array(simout.sensordata["pos_c"][:, 1], dtype=float)
    omegas = np.array(simout.sensordata["gyro_c"][:, 2], dtype=float) #rotation along z-axis
    locxs = np.array(simout.sensordata["local_x_axis"], dtype=float)
    
    phis = []
    q_des = []
    
    for t, locx, omega  in zip(simout.times, locxs, omegas):
        phi = calculate_phi_from_loc_x_axis(locx)*180/np.pi
        phis.append(phi)

        r_des = unpack_traj(trajectory(t, init_pos))[0]
        r_des[2] = np.arctan2(np.sin(r_des[2]), np.cos(r_des[2]))*180/np.pi
        q_des.append(r_des)

    q = np.vstack([x,y,phis]).T

    plot_tracking(simout, q, q_des)

    
    # simout.plot(fin_dur, ['Скорость центра робота [м/с]', 'Скорость вращения робота [рад/с]','Единичный вектор оси Ox робота' ], [['v_x','v_y','v_z'], ['omega_x','omega_y','omega_z'], ['x','y','z'] ])

