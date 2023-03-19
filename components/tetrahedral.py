import Sofa.Core

from typing import Union
from enum import Enum
from pathlib import Path

LOADER_INFOS = {
    ".obj": "MeshObjLoader",
    ".stl": "MeshSTLLoader",
    ".vtk": "MeshVTKLoader",
    ".msh": "MeshGmshLoader",
    ".gidmsh": "GIDMeshLoader",
}

MATERIALS =  [
    "Corotated",
    "NeoHookean",
    "StVenantKirchhoff"
]

class Topology(Enum):
    TRIANGLE    = "Triangle"
    TETRAHEDRON = "Tetrahedron"

def add_loader(
    parent_node: Sofa.Core.Node, 
    filename: Union[Path, str],
    name: str = "loader",
    transformation: list = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1]
    )-> Sofa.Core.Object:

    filepath = Path(filename)
    assert filepath.absolute().is_file(), f"Could not find file {filepath.absolute()}"
    filetype = filepath.suffix
    assert filetype in LOADER_INFOS, f"No loader found for filetype {filetype}"

    loader = parent_node.addObject(LOADER_INFOS[filetype], 
                            filename=str(filepath),
                            name=name,
                            transformation=transformation
                            )
    return loader

def add_topology(
    parent_node: Sofa.Core.Node, 
    mesh_loader: Sofa.Core.Object,
    topology: Topology
    ) -> Sofa.Core.Object:

    topology_container = parent_node.addObject(f"{topology.value}SetTopologyContainer", 
                                                name="topology", 
                                                src=mesh_loader.getLinkPath()
                                                )
    return topology_container

def add_tetrahedral_forcefield(
    parent_node: Sofa.Core.Node, 
    material: str = MATERIALS[1],
    E: float = 2850,
    nu: float = 0.45
    ) -> Sofa.Core.Object:

    assert material in MATERIALS, f"Invalid choice for material type {material}"

    if material == "Corotated":
        femFF = parent_node.addObject('TetrahedronFEMForceField',
                                            youngModulus=E,
                                            poissonRatio=nu,
                                            name='FEM',
                                            method='large'
                                            )
        return femFF
                                            
    elif material == "StVenantKirchhoff":
        llambda = (E*nu)/((1+nu)*(1-2*nu))
        mu = E/(2+2*nu)
        paramSet = [mu,llambda]
    elif material == "NeoHookean":
        mu = E/(2+2*nu)
        K = E/(3*(1-2*nu))
        paramSet = [mu,K]

    femFF = parent_node.addObject('TetrahedronHyperelasticityFEMForceField',
                                    ParameterSet=paramSet, 
                                    materialName=material,
                                    name='FEM'
                                    )
    return femFF

def add_triangle_forcefield(
    parent_node: Sofa.Core.Node, 
    material: str = MATERIALS[1],
    E: float = 2850,
    nu: float = 0.45
    ) -> Sofa.Core.Object:
    femFF = parent_node.addObject('TriangularFEMForceField',
                                        youngModulus=E,
                                        poissonRatio=nu,
                                        name='FEM',
                                        method='large'
                                        )
    parent_node.addObject('TriangularBendingSprings',
                            name="FEM-Bend",
                            stiffness="10",
                            template="Vec3d",
                            damping="1.0",
                        )
    return femFF

def add_collision_models(
    parent_node: Sofa.Core.Node, 
    contact_stiffness: float = 10,
    moving: int = 1,
    simulated: int = 1,
    color: list = [1, 0, 1, 1],
    triangles: bool = True,
    lines: bool = True,
    points: bool = True,
    spheres: bool = False,
    radius: float = 0.01,
    ):
    if spheres:
        parent_node.addObject('SphereCollisionModel', 
                                    radius=radius,
                                    moving=moving, 
                                    simulated=simulated, 
                                    contactStiffness=contact_stiffness, 
                                    color=color)
    else:
        if triangles:        
            parent_node.addObject('TriangleCollisionModel', 
                                        moving=moving, 
                                        simulated=simulated, 
                                        contactStiffness=contact_stiffness, 
                                        color=color)
        if lines:
            parent_node.addObject('LineCollisionModel', 
                                        moving=moving, 
                                        simulated=simulated, 
                                        contactStiffness=contact_stiffness, 
                                        color=color)
        if points:
            parent_node.addObject('PointCollisionModel', 
                                        moving=moving, 
                                        simulated=simulated, 
                                        contactStiffness=contact_stiffness, 
                                        color=color)

def add_visual_models(
    parent_node: Sofa.Core.Node, 
    color: list = [1, 0, 1, 1],
    ):
    parent_node.addObject('VisualStyle', displayFlags='showVisual')
    parent_node.addObject('OglModel', color=color, name="Visual")
