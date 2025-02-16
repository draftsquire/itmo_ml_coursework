import numpy as np
import matplotlib.pyplot as plt
import mujoco  # Add back mujoco import
from typing import List, Tuple

from mecanum_gen import generate_scene_square, obstacle_positions_square
from mecanum_sim import SensorOutput, SimHandler
from control import Summator3, tau_body2wheel, ctrl_bodyCTC
from traj import point_traj, unpack_traj
from path_planner import PathPlanner
from utils import plot_tracking


def generate_trajectory_from_path(path: List[Tuple[float, float]], 
                                init_pos: List[float],
                                speed: float = 1.0) -> callable:
    """
    Convert path to trajectory function
    
    Args:
        path: List of waypoints [(x, y)]
        init_pos: Initial position [x, y, theta]
        speed: Desired speed
        
    Returns:
        Trajectory function t -> (pos, vel, acc)
    """
    path = np.array(path)
    total_distance = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    duration = total_distance / speed
    
    def trajectory(t: float, init_pos: List[float]) -> np.ndarray:
        if t >= duration:
            return np.array([path[-1, 0], path[-1, 1], 0.0])  # x, y, theta
            
        # Calculate progress along path
        s = (t / duration) * total_distance
        
        # Find segment
        distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        segment_idx = np.searchsorted(distances, s)
        
        if segment_idx == 0:
            pos = path[0]
        else:
            # Interpolate within segment
            s_prev = distances[segment_idx - 1]
            alpha = (s - s_prev) / (distances[segment_idx] - s_prev)
            pos = path[segment_idx - 1] + alpha * (path[segment_idx] - path[segment_idx - 1])
            
        # Calculate desired orientation (tangent to path)
        if segment_idx < len(path) - 1:
            theta = np.arctan2(path[segment_idx + 1, 1] - path[segment_idx, 1],
                             path[segment_idx + 1, 0] - path[segment_idx, 0])
        else:
            theta = np.arctan2(path[-1, 1] - path[-2, 1],
                             path[-1, 0] - path[-2, 0])
            
        return np.array([pos[0], pos[1], theta])
    
    return trajectory


if __name__ == '__main__':
    # Import obstacle positions from mecanum_gen
    from mecanum_gen import obstacle_positions_square
    
    target_objest = 'boxx8.stl'

    # Convert obstacle positions to list format for path planner
    obstacles = []
    
    # Add boxes
    for i in range(9):  # boxx0 to boxx8
        name = f'boxx{i}.stl'
        if name != target_objest and name in obstacle_positions_square:
            pos = obstacle_positions_square[name]
            obstacles.append((pos[0], pos[1], 0.4))  # Use actual box size
    
    # Add borders
    for i in range(4):  # border0 to border3
        name = f'border{i}.stl'
        if name in obstacle_positions_square:
            pos = obstacle_positions_square[name]
            obstacles.append((pos[0], pos[1], 0.2))  # Use actual border width

    # Define environment bounds based on the actual room size
    bounds = (-4, 4, -4, 4)  # (min_x, max_x, min_y, max_y)
    
    # Starting position
    init_pos = [3.5, 3.5, 0]  
    
    # Set goal position
    goal_pos = [-3.5, -3.5]  # Take only x,y coordinates

    # Define robot dimensions (for mecanum robot)
    robot_footprint = [
        (-0.2, -0.15),  # back-left
        (0.2, -0.15),   # back-right
        (0.2, 0.15),    # front-right
        (-0.2, 0.15),   # front-left
    ]

    # Create path planner with robot size
    planner = PathPlanner(
        start=(init_pos[0], init_pos[1]),
        goal=goal_pos,
        obstacles=obstacles,
        bounds=bounds,
        algorithm='astar',  # or 'rrt'
        grid_resolution=0.1,  # for A*
        rrt_step_size=0.5,   # for RRT
        max_iterations=5000,
        robot_radius=0.2,  # Default circular approximation
        robot_footprint=robot_footprint  # Actual robot shape
    )
    
    # Plan path
    path = planner.plan()
    if path is None:
        print("No path found!")
        exit()
        
    # Visualize path
    planner.visualize(path)
    
    # Generate trajectory from path with slower speed for safety
    trajectory = generate_trajectory_from_path(path, init_pos, speed=0.2)

    spec, _ = generate_scene_square(init_pos, n_rollers=8, torq_max=.4)
    
    # Add sensors
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

    simtime = 120

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
    fin_dur = simh.simulate(is_slowed=0, control_func=ctrl_bodyCTC, control_func_args=(trajectory, init_pos, summator))

    # Plotting graphs
    x = np.array(simout.sensordata["pos_c"][:, 0], dtype=float)
    y = np.array(simout.sensordata["pos_c"][:, 1], dtype=float)
    locxs = np.array(simout.sensordata["local_x_axis"], dtype=float)
    phis = np.arctan2(locxs[:,1], locxs[:,0])*180/np.pi

    q_des = []
    for t in simout.times:
        r_des = unpack_traj(trajectory(t, init_pos))[0]
        r_des[2] = np.arctan2(np.sin(r_des[2]), np.cos(r_des[2]))*180/np.pi
        q_des.append(r_des)

    q = np.vstack([x,y,phis]).T

    plot_tracking(simout, q, q_des)
    