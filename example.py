import numpy as np
import mujoco

from mecanum_gen import generate_scene
from mecanum_sim import SensorOutput, SimHandler


if __name__ == '__main__':
    spec = generate_scene([5, 0, 0])
    spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name='gyro_c', type=mujoco.mjtSensor.mjSENS_GYRO, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)

    spec.compile()
    model_xml = spec.to_xml()

    simtime = 100

    # prepare data logger
    simout = SensorOutput(sensor_names=[sen.name for sen in spec.sensors],
                          sensor_dims=[3,3])
    # prepare sim params
    simh = SimHandler(model_xml, None, simlength=simtime, simout=simout)  

    # define control function
    def ctrl_f(t, data):
        # we can access mjData here
        print(data.qpos[:4])
        # return np.array([np.sin(6*t),-np.sin(6*t),np.sin(6*t),-np.sin(6*t)])*0.2
        return np.array([0,0,0,0])

    # run MuJoCo simulation
    fin_dur = simh.simulate(is_slowed=True, control_func=ctrl_f)

    simout.plot(fin_dur, ['Скорость центра робота [м/с]','dasdasd'], [['v_x','v_y','v_z'],['dick','cock','penis']])

    # print out xml
    print(model_xml)
