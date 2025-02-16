import itertools
from os.path import join

import numpy as np
import mujoco

# Add at the top of the file, outside any function
obstacle_positions = {
    'board1.stl': [2.71921921, 0.28092855, 0],
    'board2.stl': [2.83014, -0.551561, 0],
    'box1.stl': [-0.32663, -0.550714, 0],
    'box2.stl': [-1.49374, -0.568944, 0],
    'chair1.stl': [1.20863, -1.15756, 0],
    'chair2.stl': [1.20863, -1.15756, 0],
    'chair3.stl': [-2.38319, -1.13964, 0],
    'chair4.stl': [-2.38319, -1.13964, 0],
    'filter.stl': [-1.01023, -0.036624, 0],
    'puff.stl': [0.736282, -0.39378, 0],
    'puff2.stl': [-3.36267, -1.36048, 0],
    'shelf1.stl': [-0.784509, 0.853262, 0],
    'shelf2.stl': [0.394559, -1.1298, 0],
    'sofa.stl': [-0.673365, -1.36861, 0],
    'table1.stl': [1.69882, 0.627825, 0],
    'table2.stl': [2.73016, 0.711562, 0],
    'table_r1.stl': [0.73241, 0.87131, 0],
    'table_r2.stl': [0.73241, 0.87131, 0],
    'trash1.stl': [2.28045, 0.025732, 0],
    'trash2.stl': [2.29587, -0.525378, 0],
    'trash3.stl': [1.84011, -0.859605, 0],
    'wall1.stl': [-0.501274, -1.81892, 0],
    'wall2.stl': [-0.501274, 1.19183, 0],
}

boxx_shift = 2.26543
obstacle_positions_square = {
    'border0.stl': [0, 4, 0],
    'border1.stl': [0, -4, 0],
    'border2.stl': [4, 0, 0],
    'border3.stl': [-4, 0, 0],
    'boxx0.stl': [-boxx_shift, -boxx_shift, 0],
    'boxx1.stl': [boxx_shift, 0, 0],
    'boxx2.stl': [0, 0, 0],
    'boxx3.stl': [-boxx_shift, 0, 0],
    'boxx4.stl': [boxx_shift, boxx_shift, 0],
    'boxx5.stl': [0, boxx_shift, 0],
    'boxx6.stl': [-boxx_shift, boxx_shift, 0],
    'boxx7.stl': [boxx_shift, -boxx_shift, 0],
    'boxx8.stl': [0, -boxx_shift, 0],
}

def disable_parts_contact(spec, parts_lists):
    exclusion_set = set()
    p_sets = []
    for parts_list in parts_lists:
        cur = set(parts_list)
        p_sets.append(cur)

    for parts_set in p_sets:
        local_combs = list(itertools.combinations(parts_set, 2))
        for comb in local_combs:
            exclusion_set.add(comb)

    for e in sorted(exclusion_set, key=lambda x: x[0].name + x[1].name):
        b1 = e[0]
        b2 = e[1]
        spec.add_exclude(name=f'exclude_{b1.name}_{b2.name}',
                               bodyname1=b1.name, bodyname2=b2.name)

def create_wheel(wtype, R, hub_thickness, n_roller, roller_angle=np.pi/4, hub_name='hub'):
        wpos = [0,0,0]

        hub_r = .9*R
        roller_r = .8*R

        roller_damping = 0#.0001
        hub_mass = 0.02
        hub_damping = 0#.001

        step = (2*np.pi) / n_roller #TODO check if nroller 16 works, it seemed to not
        chord = 2 * (R - roller_r)*np.sin(step/2) #distance between neighbor pins
        psi = roller_angle
        # for desired roller angle this must hold: chord/h == np.tan(psi)
        h = chord/np.tan(psi)

        wspec = mujoco.MjSpec()

        hub = wspec.worldbody.add_body(name=hub_name, pos=wpos, 
                                    #    mass=hub_mass, ipos=[0,0,0], 
                                    #    iquat=[0.707107, 0, 0, 0.707107], 
                                    #    inertia=[0.0524193, 0.0303095, 0.0303095]
                                       )
        hub.add_joint(name=hub_name, axis=[0,1,0], damping=hub_damping)
        hub.add_geom(size=[hub_r,hub_thickness/2,0], quat=[0.707107, 0.707107, 0, 0], 
                     type=mujoco.mjtGeom.mjGEOM_CYLINDER, group=1,
                     contype=1, conaffinity=0,
                    #  rgba=[0.2, 0.2, 0.2, 0.5]
                     )

        for i in range(n_roller):
            roller_name = 'roller_' + str(i)
            joint_name = 'slip_' + str(i)

            pin_1 = np.array([(R - roller_r)*np.cos(step*i), -h/2, (R - roller_r)*np.sin(step*i)])

            if wtype == 0:
                if i == n_roller-1:
                    pin_2 = np.array([(R - roller_r)*np.cos(step*0), h/2, (R - roller_r)*np.sin(step*0)])
                else:
                    pin_2 = np.array([(R - roller_r)*np.cos(step*(i+1)), h/2, (R - roller_r)*np.sin(step*(i+1))])
            else:
                if i == 0:
                    pin_2 = np.array([(R - roller_r)*np.cos(step*(n_roller-1)), h/2, (R - roller_r)*np.sin(step*(n_roller-1))])
                else:
                    pin_2 = np.array([(R - roller_r)*np.cos(step*(i-1)), h/2, (R - roller_r)*np.sin(step*(i-1))])
            axis = pin_2 - pin_1
            pos = pin_1 + axis/2

            roller = hub.add_body(name=roller_name, pos=pos,
                                  ipos=[0,0,0], #iquat=[0.711549, 0.711549, 0, 0], 
                                  inertia=[.00001, .00001, .00001], mass=.001)
            roller.add_joint(name=joint_name, axis=axis, 
                             damping=roller_damping, limited=False, actfrclimited=False)
            roller.add_geom(size=[roller_r,0,0], quat=[1, 0, 0, 0], group=1
                            # contype=1, conaffinity=0, 
                            # rgba=[0.2, 0.2, 0.2, 1]
                            )
        return hub, wspec

def generate_scene(init_pos=[0,0,0], n_rollers=8, torq_max=0):
    """
    Generate scene with mecanum platform.

    Args:
        init_pos: [x, y, z] coordinates for initial robot position. 
    Returns:
        Mujoco mjSpec object

    """
    spec = mujoco.MjSpec()

    # spec.option.timestep = 0.05
    # getattr(spec.visual, 'global').azimuth = 45
    # getattr(spec.visual, 'global').elevation = 0
    spec.visual.scale.jointlength = 1.6
    spec.visual.scale.jointwidth = 0.12

    # does not work in mujoco 3.2.6, fixed in an unreleased version
    # j_color = [47 / 255, 142 / 255, 189 / 255, 1]
    # mjcf_model.visual.rgba.joint = j_color
    # spec.visual.rgba.constraint = j_color

    spec.stat.extent = 0.6
    spec.stat.center = [0,0,.3]

    spec.add_texture(name="//unnamed_texture_0", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1, 1, 1], rgb2=[1, 1, 1], 
                     width=512, height=3072)
    tex = spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4], 
                     rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name='groundplane', texrepeat=[5, 5], reflectance=.2, texuniform=True).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

    spec.worldbody.add_light(name="//unnamed_light_0", directional=True, castshadow=False, pos=[0, 0, 3], dir=[0, 0.8, -1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, condim=1, size=[0, 0, 0.125], material="groundplane")#, contype="1" conaffinity="0")

    mesh_filenames = ['board1.stl', 'board2.stl', 'box1.stl', 'box2.stl', 'chair1.stl', 
                 'chair2.stl', 'chair3.stl', 'chair4.stl', 'filter.stl', 'puff.stl', 
                 'puff2.stl', 'shelf1.stl', 'shelf2.stl', 'sofa.stl', 'table1.stl', 
                 'table2.stl', 'table_r1.stl', 'table_r2.stl', 'trash1.stl', 
                 'trash2.stl', 'trash3.stl', 'wall1.stl', 'wall2.stl']
    mesh_path = "meshes"
    for fn in mesh_filenames:
        mesh_name = fn[:-4] # cutoff the file extension
        spec.add_mesh(file=join(mesh_path,fn))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name)

    wheel_filename = 'mecanum.stl'
    mecanum1_visual = spec.add_mesh(name='mesh1', file=join(mesh_path,wheel_filename), scale=[.78,.78,.78])
    mecanum2_visual = spec.add_mesh(name='mesh2', file=join(mesh_path,wheel_filename), scale=[.78,.78,-.78])

    l, w, h = .4, .2, .05
    wR = 0.04
    hub_thickness = wR
    # n_roll = 8

    box = spec.worldbody.add_body(name="box", pos=[init_pos[0],init_pos[1], 0 + wR+h/2])
    box.add_joint(name="slidex", axis=[1,0,0], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="slidey", axis=[0,1,0], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="slidez", axis=[0,0,1], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="phi", axis=[0,0,1])
    # box.add_freejoint()
    box_color = [.2,.4,.2,1]
    box.add_geom(size=[w/2,l/2,h/2], type=mujoco.mjtGeom.mjGEOM_BOX, rgba=box_color)
    box.add_site(name='box_center')

    dx = w/2 + hub_thickness/2
    dy = .8*l/2
    dz = -h/2

    site1 = box.add_site(pos=[dx,dy,dz], euler=[0,0,-90]) # front right
    site2 = box.add_site(pos=[-dx,dy,dz], euler=[0,0,-90]) # front left
    site3 = box.add_site(pos=[-dx,-dy,dz], euler=[0,0,-90]) # rear left
    site4 = box.add_site(pos=[dx,-dy,dz], euler=[0,0,-90]) # rear right

    hub_name = 'hub'
    wheel_body1, _ = create_wheel(0, wR, hub_thickness, n_rollers, hub_name=hub_name)
    wheel_body2, _ = create_wheel(1, wR, hub_thickness, n_rollers, hub_name=hub_name)
    w1 = site1.attach(wheel_body1, 'w1-', '')
    w2 = site2.attach(wheel_body2, 'w2-', '')
    w3 = site3.attach(wheel_body1, 'w3-', '')
    w4 = site4.attach(wheel_body2, 'w4-', '')

    spec.compiler.inertiagrouprange = [0,1]

    wheel_color = [.2,.2,.2,1]

    w1.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w2.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w3.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w4.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)

    disable_parts_contact(spec, [(box, w1, *w1.bodies),
                                 (box, w2, *w2.bodies),
                                 (box, w3, *w3.bodies),
                                 (box, w4, *w4.bodies),
                                 (w1, spec.worldbody),
                                 (w2, spec.worldbody),
                                 (w3, spec.worldbody),
                                 (w4, spec.worldbody)])
    # Collision is enabled only for pairs world-rollers(spheres) and world-chassie(box). 
    # Hub and visual wheel are disabled

    input_saturation = [-abs(torq_max),abs(torq_max)] # Nm
    for i in range(4):
        spec.add_actuator(name=f'torque{i+1}', target=f'w{i+1}-'+hub_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          ctrllimited=bool(abs(torq_max)), ctrlrange=input_saturation
                          )

    return spec

def generate_scene_square_random(init_pos=[0,0,0], n_rollers=8, torq_max=0):
    """
    Generate scene with mecanum platform.

    Args:
        init_pos: [x, y, z] coordinates for initial robot position. 
    Returns:
        Mujoco mjSpec object

    """
    spec = mujoco.MjSpec()

    # spec.option.timestep = 0.05
    # getattr(spec.visual, 'global').azimuth = 45
    # getattr(spec.visual, 'global').elevation = 0
    spec.visual.scale.jointlength = 1.6
    spec.visual.scale.jointwidth = 0.12

    # does not work in mujoco 3.2.6, fixed in an unreleased version
    # j_color = [47 / 255, 142 / 255, 189 / 255, 1]
    # mjcf_model.visual.rgba.joint = j_color
    # spec.visual.rgba.constraint = j_color

    spec.stat.extent = 0.6
    spec.stat.center = [0,0,.3]

    spec.add_texture(name="//unnamed_texture_0", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1, 1, 1], rgb2=[1, 1, 1], 
                     width=512, height=3072)
    tex = spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4], 
                     rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name='groundplane', texrepeat=[5, 5], reflectance=.2, texuniform=True).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

    spec.worldbody.add_light(name="//unnamed_light_0", directional=True, castshadow=False, pos=[0, 0, 3], dir=[0, 0.8, -1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, condim=1, size=[0, 0, 0.125], material="groundplane")#, contype="1" conaffinity="0")

    mesh_filenames = ['board1.stl', 'board2.stl', 'box1.stl', 'box2.stl', 'chair1.stl', 
                 'chair2.stl', 'chair3.stl', 'chair4.stl', 'filter.stl', 'puff.stl', 
                 'puff2.stl', 'shelf1.stl', 'shelf2.stl', 'sofa.stl', 'table1.stl', 
                 'table2.stl', 'table_r1.stl', 'table_r2.stl', 'trash1.stl', 
                 'trash2.stl', 'trash3.stl', 'wall1.stl', 'wall2.stl']
    mesh_path = "meshes"

    selected = [2,3,9,12,16, 17, 18,19,20]*2
    np.random.seed(2)
    # rand_pos = (np.random.rand(len(selected),2)-.5)*4
    x_range = np.arange(-4,4,.5)
    # y_range = np.arange(-2,2,.5)
    rand_pos_idx = np.random.randint(0,len(x_range),(len(selected),2))
    final_positions = {}

    def dict_append(final_positions, fn, fin_pos):
        if fn not in final_positions:
            final_positions[fn] = fin_pos
        elif type(final_positions[fn]) == tuple:
            final_positions[fn] = (*final_positions[fn], fin_pos)
        else:
            final_positions[fn] = (final_positions[fn], fin_pos)

    # print('rand',pos_grid)
    for i, fn in enumerate(np.array(mesh_filenames)[selected]):
        mesh_name = f'mesh_{i+1}'#fn[:-4] # cutoff the file extension
        spec.add_mesh(file=join(mesh_path,fn), name=mesh_name)
        shift_to_0 = -np.array(obstacle_positions[fn])
        pos_gridded = np.array([x_range[rand_pos_idx[i,0]], x_range[rand_pos_idx[i,1]], 0])
        
        fin_pos = pos_gridded+shift_to_0
        final_positions[mesh_name] = fin_pos

        # dict_append(final_positions, fn, fin_pos)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, pos=fin_pos)

    spec.add_mesh(file=join(mesh_path,'wall1.stl'),name='wall1')
    spec.add_mesh(file=join(mesh_path,'wall1.stl'),name='wall2')
    spec.add_mesh(file=join(mesh_path,'wall1.stl'),name='wall3')
    spec.add_mesh(file=join(mesh_path,'wall1.stl'),name='wall4')
    shift_to_0 = -np.array(obstacle_positions['wall1.stl'])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='wall1', pos= np.array([0,-4,0])+shift_to_0+np.array([-.5,0,0]))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='wall2', pos= np.array([0,4,0])+shift_to_0+np.array([-.5,0,0]))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='wall3', pos= np.array([-4,0,0])+shift_to_0+np.array([-2.3,-1.75,0]), euler=[0,0,90])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='wall4', pos= np.array([4,0,0])+shift_to_0+np.array([-2.3,-1.75,0]), euler=[0,0,90])
    final_positions['wall1'] = np.array([0,-4,0])+shift_to_0+np.array([-.5,0,0])
    final_positions['wall2'] = np.array([0,4,0])+shift_to_0+np.array([-.5,0,0])
    final_positions['wall3'] = np.array([-4,0,0])+shift_to_0+np.array([-2.3,-1.75,0])
    final_positions['wall4'] = np.array([4,0,0])+shift_to_0+np.array([-2.3,-1.75,0])
    # dict_append(final_positions, 'wall1.stl', np.array([0,-4,0])+shift_to_0+np.array([-.5,0,0]))
    # dict_append(final_positions, 'wall1.stl', np.array([0,4,0])+shift_to_0+np.array([-.5,0,0]))
    # dict_append(final_positions, 'wall1.stl', np.array([-4,0,0])+shift_to_0+np.array([-2.3,-1.75,0]))
    # dict_append(final_positions, 'wall1.stl', np.array([4,0,0])+shift_to_0+np.array([-2.3,-1.75,0]))

    shift_to_0 = -np.array(obstacle_positions['puff.stl'])

    # spec.add_mesh(file=join(mesh_path,'puff.stl'),name='puff1')
    # fin_pos = shift_to_0+np.array([-2,-2,0])
    # spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='puff1', pos= fin_pos)
    # final_positions['puff1'] = fin_pos

    spec.add_mesh(file=join(mesh_path,'puff.stl'),name='puff2')
    fin_pos = shift_to_0+np.array([-1.1,-2.7,0])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='puff2', pos= fin_pos)
    final_positions['puff2'] = fin_pos

    wheel_filename = 'mecanum.stl'
    mecanum1_visual = spec.add_mesh(name='mesh1', file=join(mesh_path,wheel_filename), scale=[.78,.78,.78])
    mecanum2_visual = spec.add_mesh(name='mesh2', file=join(mesh_path,wheel_filename), scale=[.78,.78,-.78])

    l, w, h = .4, .2, .05
    wR = 0.04
    hub_thickness = wR
    # n_roll = 8

    box = spec.worldbody.add_body(name="box", pos=[init_pos[0],init_pos[1], 0 + wR+h/2])
    box.add_joint(name="slidex", axis=[1,0,0], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="slidey", axis=[0,1,0], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="slidez", axis=[0,0,1], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="phi", axis=[0,0,1])
    # box.add_freejoint()
    box_color = [.2,.4,.2,1]
    box.add_geom(size=[w/2,l/2,h/2], type=mujoco.mjtGeom.mjGEOM_BOX, rgba=box_color)
    box.add_site(name='box_center')

    dx = w/2 + hub_thickness/2
    dy = .8*l/2
    dz = -h/2

    site1 = box.add_site(pos=[dx,dy,dz], euler=[0,0,-90]) # front right
    site2 = box.add_site(pos=[-dx,dy,dz], euler=[0,0,-90]) # front left
    site3 = box.add_site(pos=[-dx,-dy,dz], euler=[0,0,-90]) # rear left
    site4 = box.add_site(pos=[dx,-dy,dz], euler=[0,0,-90]) # rear right

    hub_name = 'hub'
    wheel_body1, _ = create_wheel(0, wR, hub_thickness, n_rollers, hub_name=hub_name)
    wheel_body2, _ = create_wheel(1, wR, hub_thickness, n_rollers, hub_name=hub_name)
    w1 = site1.attach(wheel_body1, 'w1-', '')
    w2 = site2.attach(wheel_body2, 'w2-', '')
    w3 = site3.attach(wheel_body1, 'w3-', '')
    w4 = site4.attach(wheel_body2, 'w4-', '')

    spec.compiler.inertiagrouprange = [0,1]

    wheel_color = [.2,.2,.2,1]

    w1.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w2.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w3.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w4.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)

    disable_parts_contact(spec, [(box, w1, *w1.bodies),
                                 (box, w2, *w2.bodies),
                                 (box, w3, *w3.bodies),
                                 (box, w4, *w4.bodies),
                                 (w1, spec.worldbody),
                                 (w2, spec.worldbody),
                                 (w3, spec.worldbody),
                                 (w4, spec.worldbody)])
    # Collision is enabled only for pairs world-rollers(spheres) and world-chassie(box). 
    # Hub and visual wheel are disabled

    input_saturation = [-abs(torq_max),abs(torq_max)] # Nm
    for i in range(4):
        spec.add_actuator(name=f'torque{i+1}', target=f'w{i+1}-'+hub_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          ctrllimited=bool(abs(torq_max)), ctrlrange=input_saturation
                          )

    return spec, final_positions

def generate_scene_square(init_pos=[0,0,0], n_rollers=8, torq_max=0):
    """
    Generate scene with mecanum platform.

    Args:
        init_pos: [x, y, z] coordinates for initial robot position. 
    Returns:
        Mujoco mjSpec object

    """
    spec = mujoco.MjSpec()

    # spec.option.timestep = 0.05
    # getattr(spec.visual, 'global').azimuth = 45
    # getattr(spec.visual, 'global').elevation = 0
    spec.visual.scale.jointlength = 1.6
    spec.visual.scale.jointwidth = 0.12

    # does not work in mujoco 3.2.6, fixed in an unreleased version
    # j_color = [47 / 255, 142 / 255, 189 / 255, 1]
    # mjcf_model.visual.rgba.joint = j_color
    # spec.visual.rgba.constraint = j_color

    spec.stat.extent = 0.6
    spec.stat.center = [0,0,.3]

    spec.add_texture(name="//unnamed_texture_0", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1, 1, 1], rgb2=[1, 1, 1], 
                     width=512, height=3072)
    tex = spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4], 
                     rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name='groundplane', texrepeat=[5, 5], reflectance=.2, texuniform=True).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

    spec.worldbody.add_light(name="//unnamed_light_0", directional=True, castshadow=False, pos=[0, 0, 3], dir=[0, 0.8, -1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, condim=1, size=[0, 0, 0.125], material="groundplane")#, contype="1" conaffinity="0")

    mesh_filenames = [*[f'boxx{i}.stl' for i in range(9)], *[f'border{i}.stl' for i in range(4)]]
    
    mesh_path = "meshes"

    final_positions = {}

    for i, fn in enumerate(np.array(mesh_filenames)):
        mesh_name = fn[:-4] # cutoff the file extension
        spec.add_mesh(file=join(mesh_path,fn), name=mesh_name)
        # shift_to_0 = -np.array(obstacle_positions_square[fn])
        # fin_pos = shift_to_0
        # final_positions[fn] = fin_pos

        # dict_append(final_positions, fn, fin_pos)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name)


    wheel_filename = 'mecanum.stl'
    mecanum1_visual = spec.add_mesh(name='mesh1', file=join(mesh_path,wheel_filename), scale=[.78,.78,.78])
    mecanum2_visual = spec.add_mesh(name='mesh2', file=join(mesh_path,wheel_filename), scale=[.78,.78,-.78])

    l, w, h = .4, .2, .05
    wR = 0.04
    hub_thickness = wR
    # n_roll = 8

    box = spec.worldbody.add_body(name="box", pos=[init_pos[0],init_pos[1], 0 + wR+h/2])
    box.add_joint(name="slidex", axis=[1,0,0], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="slidey", axis=[0,1,0], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="slidez", axis=[0,0,1], type=mujoco.mjtJoint.mjJNT_SLIDE)
    box.add_joint(name="phi", axis=[0,0,1])
    # box.add_freejoint()
    box_color = [.2,.4,.2,1]
    box.add_geom(size=[w/2,l/2,h/2], type=mujoco.mjtGeom.mjGEOM_BOX, rgba=box_color)
    box.add_site(name='box_center')

    dx = w/2 + hub_thickness/2
    dy = .8*l/2
    dz = -h/2

    site1 = box.add_site(pos=[dx,dy,dz], euler=[0,0,-90]) # front right
    site2 = box.add_site(pos=[-dx,dy,dz], euler=[0,0,-90]) # front left
    site3 = box.add_site(pos=[-dx,-dy,dz], euler=[0,0,-90]) # rear left
    site4 = box.add_site(pos=[dx,-dy,dz], euler=[0,0,-90]) # rear right

    hub_name = 'hub'
    wheel_body1, _ = create_wheel(0, wR, hub_thickness, n_rollers, hub_name=hub_name)
    wheel_body2, _ = create_wheel(1, wR, hub_thickness, n_rollers, hub_name=hub_name)
    w1 = site1.attach(wheel_body1, 'w1-', '')
    w2 = site2.attach(wheel_body2, 'w2-', '')
    w3 = site3.attach(wheel_body1, 'w3-', '')
    w4 = site4.attach(wheel_body2, 'w4-', '')

    spec.compiler.inertiagrouprange = [0,1]

    wheel_color = [.2,.2,.2,1]

    w1.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w2.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w3.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)
    w4.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2, rgba=wheel_color)

    disable_parts_contact(spec, [(box, w1, *w1.bodies),
                                 (box, w2, *w2.bodies),
                                 (box, w3, *w3.bodies),
                                 (box, w4, *w4.bodies),
                                 (w1, spec.worldbody),
                                 (w2, spec.worldbody),
                                 (w3, spec.worldbody),
                                 (w4, spec.worldbody)])
    # Collision is enabled only for pairs world-rollers(spheres) and world-chassie(box). 
    # Hub and visual wheel are disabled

    input_saturation = [-abs(torq_max),abs(torq_max)] # Nm
    for i in range(4):
        spec.add_actuator(name=f'torque{i+1}', target=f'w{i+1}-'+hub_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          ctrllimited=bool(abs(torq_max)), ctrlrange=input_saturation
                          )

    return spec, obstacle_positions_square
