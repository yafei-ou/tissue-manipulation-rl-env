import Sofa.Core
import Sofa.SofaDeformable

from typing import Optional
import numpy as np

from components.solver import SolverType, ConstraintCorrectionType, add_solver
from components.tetrahedral import Topology, add_collision_models, add_loader, add_triangle_forcefield, add_topology, add_visual_models
from components.utils import create_springs, read_ints


import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

DISPLAY_SIZE = (800, 600)

def init_display(node):
    pygame.display.init()
    pygame.display.set_mode(DISPLAY_SIZE, pygame.DOUBLEBUF | pygame.OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    Sofa.SofaGL.glewInit()
    Sofa.Simulation.initVisual(node)
    Sofa.Simulation.initTextures(node)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (DISPLAY_SIZE[0] / DISPLAY_SIZE[1]), 0.1, 50.0)

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
    gluPerspective(45, (DISPLAY_SIZE[0] / DISPLAY_SIZE[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cameraMVM = rootNode.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    Sofa.SofaGL.draw(rootNode)

    pygame.display.get_surface().fill((0, 0, 0))
    pygame.display.flip()


class SkinTissue(Sofa.Core.Controller):
    def __init__(
        self,
        root_node: Sofa.Core.Node,
        volume_filename: str,
        init_tf: np.ndarray = np.identity(4),
        density: float = 1000,
        material: str = "Corotated",
        E: float = 3000,
        nu: float = 0.45,
        fixed_indices: Optional[list] = None,
        node_name: str = "Tissue",
        ):
        """ 
        Args:
            root_node (Sofa.Core.Node): root node of the simulation.
            volume_filename (str): name of the tetrahedral volume file.
            init_tf (np.ndarray): initial 4x4 transformation to be applied to the model.
            density (float): object density.
            material (str): type of material.
            E (float): elastic modulus of the object.
            nu (float): poisson ratio.
            fixed_indices (list): indices of the volume mesh which are constrained in all directions.
            node_name (str): name of the created node.
            
        """
        Sofa.Core.Controller.__init__(self)

        tissue_node = root_node.addChild(node_name)

        # Add mechanical model
        volume_loader = add_loader( parent_node=tissue_node,
                                    filename=volume_filename,
                                    name='tissue_volume_loader',
                                    transformation=init_tf
                                    )

        self.topology = add_topology( parent_node=tissue_node, 
                                      mesh_loader=volume_loader,
                                      topology=Topology.TRIANGLE
                                    )

        self.state = tissue_node.addObject('MechanicalObject',
                                name='tissue_state',
                                template='Vec3d',
                                showObject=0,
                                listening=1,
                                src='@tissue_volume_loader'
                                )

        tissue_node.addObject('UniformMass', name='tissue_mass', totalMass=density)
                
        add_triangle_forcefield( parent_node=tissue_node,
                                    material = material,
                                    E = E,
                                    nu = nu
                                    )

        add_solver( parent_node=tissue_node, 
                    solver_type=SolverType.CG,
                    rayleigh_stiffness = 0.1,
                    rayleigh_mass = 0.1,
                    linear_solver_iterations = 50,
                    add_constraint_correction=True, 
                    constraint_correction=ConstraintCorrectionType.LINEAR
                    )

        # Use fixed points are in the center circular region of the tissue
        if fixed_indices is None:
            self.fixed_indices_str = "0 1 2 3"
            tissue_node.addObject('FixedConstraint', name='fixedConstraint', indices=self.fixed_indices_str)
        else:
            pass

        # Add visual models
        visual_node = tissue_node.addChild(f"Visual{node_name}")
        
        add_visual_models( parent_node = visual_node,
                            color = [1, 0.533, 0.471, 0.99] # 255, 136, 120
                            )
        visual_node.addObject('IdentityMapping', name='VisualMapping', input="@../tissue_state", output="@Visual")
        
        self.previous_position = self.topology.position.value
        self.node = tissue_node
        self.is_stable = True


    def _find_fixed_points(self, whole_plane=False):
        rest_node_position = self.topology.position.value
        index_plane = np.flatnonzero(rest_node_position[:,1] <= 1e-4)
        if whole_plane:
            return index_plane
        plane_node_position_2d = rest_node_position[index_plane,:][:, [0,2]]
        square_sum = np.sum(plane_node_position_2d * plane_node_position_2d, axis=1)
        index_circle = np.flatnonzero(square_sum < 400)
        return index_plane[index_circle]
