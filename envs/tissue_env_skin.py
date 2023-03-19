import random
import math

from objects.skin_tissue import SkinTissue

from .base_env import *
from components.utils import Parameters
from components.utils import Parameters, read_ints, create_springs
from components.header import add_scene_header
import matplotlib.pyplot as plt


import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import Sofa.SofaGL

display_size = (1280, 720)

def init_display(node):
    pygame.display.init()
    pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    Sofa.SofaGL.glewInit()
    Sofa.Simulation.initVisual(node)
    Sofa.Simulation.initTextures(node)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 200.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def simple_render(rootNode):
    """
     Get the OpenGL Context to render an image (snapshot) of the simulation state
     """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cameraMVM = rootNode.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    Sofa.SofaGL.draw(rootNode)

    pygame.display.get_surface().fill((0,0,0))
    pygame.display.flip()

def full_render(root_node, observation_points, target_points, grasping_points=None):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cameraMVM = root_node.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    Sofa.SofaGL.draw(root_node)

    fixed_points = np.array([[-10, -10, 0], [-10, 10, 0], [10, 10, 0], [10, -10, 0]])

    glEnable(GL_POINT_SMOOTH)
    glPointSize(3)
    glBegin(GL_POINTS)
    glColor3d(1, 1, 1)
    for p in fixed_points:
        glVertex3d(p[0], p[1], p[2])
    glEnd()

    glPointSize(15)
    glBegin(GL_POINTS)
    glColor3d(0, 0, 0)
    for p in observation_points:
        glVertex3d(p[0], p[1], p[2])
    glColor3d(0, 1, 0)
    for p in target_points:
        glVertex3d(p[0], p[1], p[2])
    glEnd()
    
    glPointSize(20)
    glBegin(GL_POINTS)
    if grasping_points is not None:
        glColor3d(0.58824, 0.19608, 0.66667)
        for p in grasping_points:
            glVertex3d(p[0], p[1], p[2])
    glEnd()
    
    pygame.display.get_surface().fill((0, 0, 0))
    pygame.display.flip()


class SkinTissueEnv(SofaEnv):
    def __init__(self, params: Parameters = None, obs_sequence_length: int = 10, render_mode: str = "human") -> None:
        action_space_type = "discrete"
        action_size = [4, 4]
        obs_size = 4
        self.params = params
        super().__init__(action_space_type, action_size, obs_size, obs_sequence_length, render_mode)
        self._init_env()

    def _init_env(self):
        # Environment variables
        self.observed_point_position = None
        self.target_point_position = None
        self.initial_distance = None
        self.current_distance_to_target = None
        self.last_point_position = None
        self.attach_position_1 = None
        self.attach_position_2 = None

        # render related
        self.r_observed_position = None
        self.r_desired_position = None
        self.r_fig = None

    def reset(self) -> np.ndarray:
        super().reset()

        Sofa.Simulation.unload(self.sf_root)
        add_scene_header( self.sf_root,
                            gravity=self.params.gravity,
                            dt=self.params.dt,
                        )
        self._create_scene([141], [61])
        Sofa.Simulation.init(self.sf_root)
        if self.render_mode=="human":
            init_display(self.sf_root)

        self._step_simulation(2)

        self.observed_point_position = self.tissue_state.position.value[self.observed_point_idx]
        self.attach_position_1 = self.master_points_1.position.value
        self.attach_position_2 = self.master_points_2.position.value

        # random target position
        self.target_point_position = self._generate_random_target_point()
        current_position = self.observed_point_position[:,0:2].flatten()
        target_position = self.target_point_position[:,0:2].flatten()
        self.current_distance_to_target = np.linalg.norm(current_position - target_position)
        self.initial_distance = self.current_distance_to_target
        self.last_distance_to_target = self.current_distance_to_target
        self.last_point_position = self.observed_point_position.copy()

        # print(self.current_distance_to_target)

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        _action = action.copy()
        def _get_direction(a):
            if a==0:
                direction = [0.2, 0]
            elif a==1:
                direction = [-0.2, 0]
            elif a==2:
                direction = [0, 0.2]
            elif a==3:
                direction = [0, -0.2]
            else:
                raise ValueError
            return direction

        direction = np.zeros((2,2))
        direction[0] = _get_direction(_action[0])
        direction[1] = _get_direction(_action[1])

        current_dummy_position = self.master_points_1.position.value.copy()
        for i in range(np.size(current_dummy_position, 0)):
            current_dummy_position[i,:] += np.array([direction[0][0],direction[0][1],0])
        if current_dummy_position[0,0] > -15 and current_dummy_position[0,0] < -5 and current_dummy_position[0,1] > -5 and current_dummy_position[0,1] < 5:
            self.master_points_1.position = current_dummy_position.tolist()

        current_dummy_position = self.master_points_2.position.value.copy()
        for i in range(np.size(current_dummy_position, 0)):
            current_dummy_position[i,:] += np.array([direction[1][0],direction[1][1],0])
        if current_dummy_position[0,0] > 5 and current_dummy_position[0,0] < 15 and current_dummy_position[0,1] > -5 and current_dummy_position[0,1] < 5:
            self.master_points_2.position = current_dummy_position.tolist()

        self._step_simulation(5)
        self._update()
        self.count_finish += 1

        # reward
        reward_position = 50 * (1 - np.sqrt(self.current_distance_to_target/self.initial_distance))

        correct_stop = False
        reward_correct_stop = 0
        reward = reward_position + reward_correct_stop

        info = {}

        if self.count_finish > 99 or correct_stop:
            # print("end_ep")
            done = True
            obs = self.reset()
            info["TimeLimit.truncated"] = True
        else:
            done = False
            obs = self._get_obs()
            self.last_point_position = self.observed_point_position.copy()

        # print(reward)
        self.render(self.render_mode)
        return obs, reward, done, info

    def render(self, render_mode):
        if render_mode=="human":
            Sofa.Simulation.updateVisual(self.sf_root)
            # simple_render(self.sf_root)
            target_mod = self.target_point_position.copy()
            target_mod[:,2] = self.observed_point_position[:,2]
            attach_points = np.array([self.attach_position_1, self.attach_position_2]).squeeze()
            full_render(self.sf_root, self.observed_point_position, target_mod, attach_points)
        elif render_mode=="pyplot":
            if self.r_fig is None:
                plt.ion()
                self.r_fig = plt.figure()
                plt.grid()
                self.r_ax = self.r_fig.add_subplot(111)
                self.r_ax.set_xlim([-15, 15])
                self.r_ax.set_ylim([-15, 15])
                self.r_current_point, = self.r_ax.plot(self.observed_point_position[:,0], self.observed_point_position[:,1], "k*")
                self.r_desired_point, = self.r_ax.plot(self.target_point_position[:,0], self.target_point_position[:,1], "g*")

                self.r_attach_point_1, = self.r_ax.plot(self.attach_position_1[0,0], self.attach_position_1[0,1], "r*")
                self.r_attach_point_2, = self.r_ax.plot(self.attach_position_2[0,0], self.attach_position_2[0,1], "b*")
            else:
                self.r_current_point.set_xdata(self.observed_point_position[:,0])
                self.r_current_point.set_ydata(self.observed_point_position[:,1])
                self.r_desired_point.set_xdata(self.target_point_position[:,0])
                self.r_desired_point.set_ydata(self.target_point_position[:,1])

                self.r_attach_point_1.set_xdata(self.attach_position_1[0,0])
                self.r_attach_point_1.set_ydata(self.attach_position_1[0,1])

                self.r_attach_point_2.set_xdata(self.attach_position_2[0,0])
                self.r_attach_point_2.set_ydata(self.attach_position_2[0,1])

                self.r_fig.canvas.draw()
                self.r_fig.canvas.flush_events()

    def _create_scene(self, attach_indices_1, attach_indices_2):
        transform = np.identity(4)

        self.dummy_node_1 = self.sf_root.addChild('DummyNode1')
        self.master_points_1 = self.dummy_node_1.addObject('MechanicalObject',name='Dummy',template='Vec3d',showObject='1',showObjectScale='5',listening='1')
        self.master_points_1.size = len(attach_indices_1)
        self.dummy_node_1.init()

        self.dummy_node_2 = self.sf_root.addChild('DummyNode2')
        self.master_points_2 = self.dummy_node_2.addObject('MechanicalObject',name='Dummy',template='Vec3d',showObject='1',showObjectScale='5',listening='1')
        self.master_points_2.size = len(attach_indices_2)
        self.dummy_node_2.init()

        self.tissue = self.sf_root.addObject(
                                SkinTissue( self.sf_root,
                                        volume_filename=self.params.volume_file,
                                        init_tf=transform,
                                        density=self.params.rho,
                                        material=self.params.material,
                                        E=self.params.E,
                                        nu=self.params.nu,
                                        fixed_indices=None
                                      )
                                )

        self.sf_tissue_node = self.sf_root.getChild("Tissue")

        self.mech_surface = self.sf_tissue_node.getObject("tissue_state")

        # Define stiff springs
        ks = 10000
        kd = 0
        restLength = 0
        springs_1 = create_springs(attach_indices_1,ks,kd,restLength)
        self.stiff_springs = self.sf_tissue_node.addObject('StiffSpringForceField',template="Vec3d",spring=springs_1,name="ExternalSprings1",object1="@DummyNode1/Dummy",object2="@Tissue/tissue_state",listening='1')
        springs_2 = create_springs(attach_indices_2,ks,kd,restLength)
        self.stiff_springs = self.sf_tissue_node.addObject('StiffSpringForceField',template="Vec3d",spring=springs_2,name="ExternalSprings2",object1="@DummyNode2/Dummy",object2="@Tissue/tissue_state",listening='1')

        # Initialize dummy nodes positions
        self.contact_points_position_1 = [self.mech_surface.position[i] for i in attach_indices_1]
        self.master_points_1.position = self.contact_points_position_1
        self.contact_points_position_2 = [self.mech_surface.position[i] for i in attach_indices_2]
        self.master_points_2.position = self.contact_points_position_2

        # Initialize observed internal point
        self.tissue_topology = self.sf_tissue_node.getObject("topology")
        self.tissue_state = self.sf_tissue_node.getObject("tissue_state")
        self.tissue_nodes = self.tissue_topology.position.value
        self.valid_observation_indices = self._find_valid_indices()
        self.observed_point_idx = self._generate_random_observed_points()

    def _update(self):
        self.observed_point_position = self.tissue_state.position.value[self.observed_point_idx]
        self.attach_position_1 = self.master_points_1.position.value
        self.attach_position_2 = self.master_points_2.position.value
        
        # round distance
        current_position = self.observed_point_position[:,0:2].flatten()
        current_position = np.around(current_position, decimals=3)
        target_position = self.target_point_position[:,0:2].flatten()
        self.current_distance_to_target = np.linalg.norm(current_position - target_position)

    def _generate_random_target_point(self):
        def _gen_theta():
            theta = self.nprng.uniform(-np.pi, np.pi, size=(2,))
            sign = np.cos(theta)
            if sign[0] >= 0 and sign[1] <= 0:
                return _gen_theta()
            return theta
        theta = _gen_theta()
        # random distance
        random_distance_1 = 0.8 + self.rng.random() * 0.0
        random_distance_2 = 0.8 + self.rng.random() * 0.0
        displacement_distance_1 = [random_distance_1*np.cos(theta[0]), random_distance_1*np.sin(theta[0])]
        displacement_distance_2 = [random_distance_2*np.cos(theta[1]), random_distance_2*np.sin(theta[1])]
        target = self.observed_point_position.copy()
        target[0] += np.array([displacement_distance_1[0], displacement_distance_1[1], 0])
        target[1] += np.array([displacement_distance_2[0], displacement_distance_2[1], 0])
        return np.around(target, decimals=1)

    def _find_valid_indices(self):
        # find indices of points in a circle on the surface
        rest_node_position = self.tissue_topology.position.value
        index_plane = np.flatnonzero(rest_node_position[:,2] <= 1e-4)
        plane_node_position_2d = rest_node_position[index_plane,:][:, 0:2]
        # square_sum = np.sum(plane_node_position_2d * plane_node_position_2d, axis=1)
        index_circle = np.flatnonzero(np.logical_and( np.abs(plane_node_position_2d[:,0]) < 5, np.abs(plane_node_position_2d[:,1]) < 2.5 ))
        return index_plane[index_circle]

    def _generate_random_observed_points(self):
        indices = self.nprng.choice(self.valid_observation_indices, size=(2,), replace=False)
        if (np.abs(self.tissue_topology.position.value[indices][0,0] - self.tissue_topology.position.value[indices][1,0]) < 6 or 
            np.abs(self.tissue_topology.position.value[indices][0,1] - self.tissue_topology.position.value[indices][1,1]) > 3):
            return self._generate_random_observed_points()

        if self.tissue_topology.position.value[indices][0,0] > self.tissue_topology.position.value[indices][1,0]:
            indices = np.array([indices[1], indices[0]])

        return indices

    def _get_obs(self):
        position_current = np.around(self.observed_point_position[:,0:2], decimals=3)
        position_target = self.target_point_position[:,0:2]
        position_attach = np.array([self.attach_position_1[:,0:2], self.attach_position_2[:,0:2]]).squeeze()
        position_attach = np.around(position_attach, decimals=3)
        obs = np.array([position_current - position_target]).flatten()
        return self._pad_obs([obs])

    def _step_simulation(self, t):
        slow_steps = int(t)
        for _ in range(slow_steps):
            Sofa.Simulation.animate(self.sf_root, self.sf_root.dt.value)


class SkinTissueEnvDiscrete(SkinTissueEnv):
    def __init__(self, params: Parameters = None, obs_sequence_length: int = 10, render_mode: str = "human") -> None:
        action_space_type = "single_discrete"
        action_size = 16
        obs_size = 4
        self.params = params
        self.action_referrence = np.array(
            [
                [0,0],
                [0,1],
                [0,2],
                [0,3],
                [1,0],
                [1,1],
                [1,2],
                [1,3],
                [2,0],
                [2,1],
                [2,2],
                [2,3],
                [3,0],
                [3,1],
                [3,2],
                [3,3],
            ]
        )
        super(SkinTissueEnv, self).__init__(action_space_type, action_size, obs_size, obs_sequence_length, render_mode)
        self._init_env()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        _action = action.copy() # discrete action
        _action_out = self.action_referrence[_action]
        return super().step(_action_out)


class SkinTissueEnvContinuous(SkinTissueEnv):
    def __init__(self, params: Parameters = None, obs_sequence_length: int = 10, render_mode: str = "human") -> None:
        action_space_type = "continuous"
        action_size = (4,)
        obs_size = 4
        self.params = params
        super(SkinTissueEnv, self).__init__(action_space_type, action_size, obs_size, obs_sequence_length, render_mode)
        self._init_env()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        _action = action.copy()
        _action[0] = round(_action[0], 1)
        _action[1] = round(_action[1], 1)
        _action[2] = round(_action[2], 1)
        _action[3] = round(_action[3], 1)
        
        direction = np.zeros((2,2))
        direction[0] = [_action[0] * 0.2, _action[1] * 0.2]
        direction[1] = [_action[2] * 0.2, _action[3] * 0.2]

        current_dummy_position = self.master_points_1.position.value.copy()
        for i in range(np.size(current_dummy_position, 0)):
            current_dummy_position[i,:] += np.array([direction[0][0],direction[0][1],0])
        if current_dummy_position[0,0] > -15 and current_dummy_position[0,0] < -5 and current_dummy_position[0,1] > -5 and current_dummy_position[0,1] < 5:
            self.master_points_1.position = current_dummy_position.tolist()

        current_dummy_position = self.master_points_2.position.value.copy()
        for i in range(np.size(current_dummy_position, 0)):
            current_dummy_position[i,:] += np.array([direction[1][0],direction[1][1],0])
        if current_dummy_position[0,0] > 5 and current_dummy_position[0,0] < 15 and current_dummy_position[0,1] > -5 and current_dummy_position[0,1] < 5:
            self.master_points_2.position = current_dummy_position.tolist()

        self._step_simulation(5)
        self._update()
        self.count_finish += 1

        # reward
        reward_position = 20 * (1 - np.sqrt(self.current_distance_to_target/self.initial_distance))
        correct_stop = False
        reward_correct_stop = 0
        reward = reward_position + reward_correct_stop

        if self.count_finish > 99 or correct_stop:
            # print("end_ep")
            done = True
            obs = self.reset()
            info = {'TimeLimit.truncated': True}
        else:
            done = False
            obs = self._get_obs()
            self.last_point_position = self.observed_point_position.copy()
            info = {}

        # print(reward)
        self.render(self.render_mode)
        return obs, reward, done, info


class SkinTissueEnvContinuousRandom(SkinTissueEnvContinuous):
    def __init__(self, params: Parameters = None, obs_sequence_length: int = 10, render_mode: str = "human", randomize=False) -> None:
        self.attach_position_list_1 = range(130, 151)
        self.attach_position_list_2 = range(52, 73)
        self.randomize = randomize
        self.simulation_step = 15

        super().__init__(params, obs_sequence_length, render_mode)
    
    def reset(self, restore=False, grasping_points=[141, 61], idx=None, target=None, attach_1=None, attach_2=None) -> np.ndarray:
        super().reset()

        Sofa.Simulation.unload(self.sf_root)
        add_scene_header( self.sf_root,
                            gravity=self.params.gravity,
                            dt=self.params.dt,
                        )

        if restore:
            return self.restore(idx, target, attach_1, attach_2)
        else:
            if self.randomize:
                attach_1 = self.rng.sample(self.attach_position_list_1, 1)
                attach_2 = self.rng.sample(self.attach_position_list_2, 1)
                youngs_modulus = self.rng.sample(range(240, 480), 1) # 0.6 - 1.2 MPa
            else:
                attach_1 = [grasping_points[0]]
                attach_2 = [grasping_points[1]]
                youngs_modulus = self.params.E
            self._create_scene(attach_1, attach_2, youngs_modulus)
            Sofa.Simulation.init(self.sf_root)
            if self.render_mode=="human":
                init_display(self.sf_root)

            self._step_simulation(2)

            self.observed_point_position = self.tissue_state.position.value[self.observed_point_idx]
            self.attach_position_1 = self.master_points_1.position.value
            self.attach_position_2 = self.master_points_2.position.value

            # random target position
            self.target_point_position = self._generate_random_target_point()
            current_position = self.observed_point_position[:,0:2].flatten()
            target_position = self.target_point_position[:,0:2].flatten()
            self.current_distance_to_target = np.linalg.norm(current_position - target_position)
            self.initial_distance = self.current_distance_to_target
            self.last_distance_to_target = self.current_distance_to_target
            self.last_point_position = self.observed_point_position.copy()

            return self._get_obs()

    def _create_scene(self, attach_indices_1, attach_indices_2, youngs_modulus=2000):
        transform = np.identity(4)

        self.dummy_node_1 = self.sf_root.addChild('DummyNode1')
        self.master_points_1 = self.dummy_node_1.addObject('MechanicalObject',name='Dummy',template='Vec3d',showObject='0',showObjectScale='5',listening='1')
        self.master_points_1.size = len(attach_indices_1)
        self.dummy_node_1.init()

        self.dummy_node_2 = self.sf_root.addChild('DummyNode2')
        self.master_points_2 = self.dummy_node_2.addObject('MechanicalObject',name='Dummy',template='Vec3d',showObject='0',showObjectScale='5',listening='1')
        self.master_points_2.size = len(attach_indices_2)
        self.dummy_node_2.init()

        self.tissue = self.sf_root.addObject(
                                SkinTissue( self.sf_root,
                                        volume_filename=self.params.volume_file,
                                        init_tf=transform,
                                        density=self.params.rho,
                                        material=self.params.material,
                                        E=youngs_modulus,
                                        nu=self.params.nu,
                                        fixed_indices=None
                                      )
                                )

        self.sf_tissue_node = self.sf_root.getChild("Tissue")

        self.mech_surface = self.sf_tissue_node.getObject("tissue_state")

        # Define stiff springs
        ks = 10000
        kd = 0
        restLength = 0
        springs_1 = create_springs(attach_indices_1,ks,kd,restLength)
        self.stiff_springs = self.sf_tissue_node.addObject('StiffSpringForceField',template="Vec3d",spring=springs_1,name="ExternalSprings1",object1="@DummyNode1/Dummy",object2="@Tissue/tissue_state",listening='1')
        springs_2 = create_springs(attach_indices_2,ks,kd,restLength)
        self.stiff_springs = self.sf_tissue_node.addObject('StiffSpringForceField',template="Vec3d",spring=springs_2,name="ExternalSprings2",object1="@DummyNode2/Dummy",object2="@Tissue/tissue_state",listening='1')

        # Initialize dummy nodes positions
        self.contact_points_position_1 = [self.mech_surface.position[i] for i in attach_indices_1]
        self.master_points_1.position = self.contact_points_position_1
        self.contact_points_position_2 = [self.mech_surface.position[i] for i in attach_indices_2]
        self.master_points_2.position = self.contact_points_position_2

        # Initialize observed internal point
        self.tissue_topology = self.sf_tissue_node.getObject("topology")
        self.tissue_state = self.sf_tissue_node.getObject("tissue_state")
        self.tissue_nodes = self.tissue_topology.position.value
        self.valid_observation_indices = self._find_valid_indices()
        self.observed_point_idx = self._generate_random_observed_points()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        _action = action.copy()
        _action[0] = round(_action[0], 1)
        _action[1] = round(_action[1], 1)
        _action[2] = round(_action[2], 1)
        _action[3] = round(_action[3], 1)
        
        direction = np.zeros((2,2))
        direction[0] = [_action[0] * 0.4, _action[1] * 0.4]
        direction[1] = [_action[2] * 0.4, _action[3] * 0.4]

        current_dummy_position = self.master_points_1.position.value.copy()
        for i in range(np.size(current_dummy_position, 0)):
            current_dummy_position[i,:] += np.array([direction[0][0],direction[0][1],0])
        self.master_points_1.position = current_dummy_position.tolist()

        current_dummy_position = self.master_points_2.position.value.copy()
        for i in range(np.size(current_dummy_position, 0)):
            current_dummy_position[i,:] += np.array([direction[1][0],direction[1][1],0])
        self.master_points_2.position = current_dummy_position.tolist()

        self._step_simulation(self.simulation_step)
        self._update()
        self.count_finish += 1

        # reward
        reward_position = 20 * (1 - np.sqrt(self.current_distance_to_target/self.initial_distance))
        
        correct_stop = False
        reward_correct_stop = 0
        reward = reward_position + reward_correct_stop

        if self.count_finish > 99 or correct_stop:
            done = True
            obs = self.reset()
            info = {'TimeLimit.truncated': True}
        else:
            done = False
            obs = self._get_obs()
            self.last_point_position = self.observed_point_position.copy()
            info = {}

        # print(reward)
        self.render(self.render_mode)
        return obs, reward, done, info

    def get_points(self):
        return self.observed_point_idx, self.target_point_position

    def set_points(self, idx, target):
        self.observed_point_idx = idx
        self.target_point_position = target

    def restore(self, idx, target, attach_1, attach_2):
        super().reset()

        Sofa.Simulation.unload(self.sf_root)
        add_scene_header( self.sf_root,
                            gravity=self.params.gravity,
                            dt=self.params.dt,
                        )
                        
        self.set_points(idx, target)
        youngs_modulus = self.params.E
        self._create_scene(attach_1, attach_2, youngs_modulus)
        self.set_points(idx, target)
        Sofa.Simulation.init(self.sf_root)
        if self.render_mode=="human":
            init_display(self.sf_root)

        self._step_simulation(2)

        self.observed_point_position = self.tissue_state.position.value[self.observed_point_idx]
        self.attach_position_1 = self.master_points_1.position.value
        self.attach_position_2 = self.master_points_2.position.value

        # target position
        current_position = self.observed_point_position[:,0:2].flatten()
        target_position = self.target_point_position[:,0:2].flatten()
        self.current_distance_to_target = np.linalg.norm(current_position - target_position)
        self.initial_distance = self.current_distance_to_target
        self.last_distance_to_target = self.current_distance_to_target
        self.last_point_position = self.observed_point_position.copy()

        return self._get_obs()

    def get_grasping_points(self):
        return np.array([self.attach_position_1[:,0:2], self.attach_position_2[:,0:2]]).squeeze()

    def get_node_points(self):
        return self.tissue_state.position.value[:,0:2]

    def set_simulation_step(self, t):
        self.simulation_step = t
