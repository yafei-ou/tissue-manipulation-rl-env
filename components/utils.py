import yaml
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R

class Parameters:
    def __init__(self, filename):
        with open(filename, 'r') as stream:
            entries = yaml.load(stream, Loader=yaml.SafeLoader)
        self.__dict__.update(entries)

class Analysis(Enum):
    LM          = "LM"
    #PRESCRDISPL = "PrescrDispl"
    #PENALTY     = "Penalty"

def matrix2xyzquat(
    matrix: np.ndarray, # 4x4 transformation matrix
    offset: np.ndarray = np.zeros(3) # offset to add to the initial position, in the not-transformed system
    ):

    # Extract translation
    xyz = np.asarray([matrix[0,3], matrix[1,3], matrix[2,3]])
    
    # Extract rotation
    rot_matrix = matrix[:3,:3]
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()

    # If offset is not zero, apply it
    if not np.array_equal(offset, np.zeros(3)):
        offset_transformed = offset.dot(rot_matrix.T)
        xyz += offset_transformed
        
    # Create translation + quaternion
    xyzquat = np.append(xyz, quat)
    return xyzquat

def create_springs(slaveIndices,ks,kd,restLength):
        dim = len(slaveIndices)
        master = (np.arange(dim))
        slave = (np.asarray(slaveIndices))
        ks = (np.repeat(ks,dim))
        kd = (np.repeat(kd,dim))
        L = (np.repeat(restLength,dim))
        springs = np.hstack((master,slave,ks,kd,L))
        springs = np.reshape(springs,(5,dim)).T
        return springs

def read_ints(filename):
        """
        Reads a series of numbers from a .txt file filename and returns a list of integers values.
        """
        my_list = []
        with open(filename, "r") as f:
                for line in f:
                        for i in line.split():
                                if i.isdigit():
                                        my_list.append(int(i))
        return my_list

def is_stable(displacement):
    """ 
    Analyzes the provided displacement and tells if the displacement is associated with 
    a stable deformation (i.e., lower than high_thresh).

    Arguments
    -----------
    displacement : array_like
        Nx3 array with x,y,z displacements of N points.
    
    Returns
    -----------
    bool
        False if there is at least one displacement with NaN value
    """

    displ_norm = np.linalg.norm(displacement, axis=1)
    max_displ_norm = np.amax(displ_norm)

    if np.isnan(max_displ_norm):
        return False
    else:
        return True
