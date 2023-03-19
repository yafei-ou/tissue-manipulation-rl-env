import Sofa.Core
from enum import Enum

class SolverType(Enum):
    CG         = "CGLinearSolver"
    SOFASPARSE = "SparseLDLSolver"

class ConstraintCorrectionType(Enum):
    UNCOUPLED   = "UncoupledConstraintCorrection"
    LINEAR      = "LinearSolverConstraintCorrection"
    PRECOMPUTED = "PrecomputedConstraintCorrection"
    GENERIC     = "GenericConstraintCorrection"

def add_solver(
            parent_node: Sofa.Core.Node,
            solver_type: SolverType = SolverType.CG,
            solver_name: str = "Solver",
            rayleigh_stiffness: float = 0.1,
            rayleigh_mass: float = 0.1,
            linear_solver_iterations: int = 25,
            linear_solver_threshold: float = 1e-16,
            linear_solver_tolerance: float = 1e-8,
            add_constraint_correction: bool = False,
            constraint_correction: ConstraintCorrectionType = ConstraintCorrectionType.LINEAR
            ):
        
    ode_solver = parent_node.addObject("EulerImplicitSolver", rayleighMass=rayleigh_mass, rayleighStiffness=rayleigh_stiffness)

    # Linear solver
    if solver_type == SolverType.CG:
        parent_node.addObject(solver_type.value, name=solver_name, iterations=linear_solver_iterations, threshold=linear_solver_threshold, tolerance=linear_solver_tolerance)
    elif solver_type == SolverType.SOFASPARSE:
        parent_node.addObject(solver_type.value, name=solver_name, template="CompressedRowSparseMatrixMat3x3d")
    else:
        raise NotImplementedError(f"No implementation for solver type {solver_type}.")

    if add_constraint_correction:
        parent_node.addObject( constraint_correction.value )
