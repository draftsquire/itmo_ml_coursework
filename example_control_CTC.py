import numpy as np
import matplotlib.pyplot as plt
import mujoco  # Add back mujoco import
from typing import List, Tuple

from mecanum_gen import generate_scene_square, obstacle_positions
from mecanum_sim import SensorOutput, SimHandler
from control import Summator3, tau_body2wheel, ctrl_bodyCTC
from traj import point_traj, unpack_traj, rotate_traj_2, z_traj, rotate_traj, round_shaped_traj, round_shaped_traj_2
from path_planner import PathPlanner
from utils import plot_tracking



if __name__ == '__main__':
    # Import obstacle positions from mecanum_gen
    from mecanum_gen import obstacle_positions


    # Starting position
    init_pos = [-3.5, -3.5, 0]

    # Generate trajectory function from path with slower speed for safety
    trajectory_f = round_shaped_traj_2
    spec,_ = generate_scene_square(init_pos, n_rollers=8, torq_max=.4)
    
    # Add back sensor definitions
    spec.add_sensor(name='pos_c', type=mujoco.mjtSensor.mjSENS_FRAMEPOS, 
                   objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='gyro_c', type=mujoco.mjtSensor.mjSENS_GYRO, 
                   objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='local_x_axis', type=mujoco.mjtSensor.mjSENS_FRAMEXAXIS, 
                   objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_VELOCIMETER, 
                   objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    
    # Compile model before generating XML
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

    summator = Summator3()
    # run MuJoCo simulation
    fin_dur = simh.simulate(is_slowed=0, control_func=ctrl_bodyCTC, control_func_args=(trajectory_f, init_pos, summator))

    # Plotting graphs
    x = np.array(simout.sensordata["pos_c"][:, 0], dtype=float)
    y = np.array(simout.sensordata["pos_c"][:, 1], dtype=float)
    locxs = np.array(simout.sensordata["local_x_axis"], dtype=float)
    phis = np.arctan2(locxs[:,1], locxs[:,0])*180/np.pi

    q_des = []
    for t in simout.times:
        r_des = unpack_traj(trajectory_f(t, init_pos))[0]
        r_des[2] = np.arctan2(np.sin(r_des[2]), np.cos(r_des[2]))*180/np.pi
        q_des.append(r_des)

    q = np.vstack([x,y,phis]).T

    plot_tracking(simout, q, q_des)
    