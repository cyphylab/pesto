import numpy as np
from scipy.spatial.transform import Rotation


def apply_rotvec(rotvec, vecs):
    """This is a debug function. It's not the fastest implementation. Only for 3D data"""
    angle = np.linalg.norm(rotvec)
    rotvec = rotvec/angle
    unrotated_vecs = np.einsum('i,ji->j', rotvec, vecs)[:,None] * rotvec[None,:]
    rotated_vecs = vecs - unrotated_vecs
    normal_vecs = np.cross(rotvec, rotated_vecs)
    outvecs = unrotated_vecs + np.cos(angle)*rotated_vecs + np.sin(angle)*normal_vecs
    return outvecs


def rotation_norm(R):
    """Returns the minimum rotation angle around all possible 3D rotation axes that achieves the rotation R."""
    if R.shape[-1] == 3:
        return np.linalg.norm(Rotation.from_matrix(R).as_rotvec(), axis=1)
    elif R.shape[-1] == 2:
        return np.arccos(np.trace(R, axis1=1, axis2=2)/2)
    else:
        raise Exception('R.shape must be (N,2,2) or (N,3,3)')
    

def twist_norm(T):
    """Returns the norm of the translation component and the rotation norm of the rotation component."""
    return np.linalg.norm(T[:,:-1,-1], axis=1), rotation_norm(T[:,:-1,:-1])


def twist_mean(Ts):
    T_mean = np.zeros((Ts.shape[1], Ts.shape[2]))
    T_mean[-1,-1] = 1.0
    T_mean[:-1, -1] = np.mean(Ts[:,:-1,-1], axis=0)

    if Ts.shape[1] == Ts.shape[2] == 4:
        rotvecs = Rotation.from_matrix(Ts[:,:-1,:-1]).as_rotvec()
        rotvec_mean = np.mean(rotvecs, axis=0)
        T_mean[:-1,:-1] = Rotation.from_rotvec(rotvec_mean).as_matrix()
    elif Ts.shape[1] == Ts.shape[2] == 3:
        pointing_vectors = Ts[:,:-1,0]
        mean_pointing_vector = np.mean(pointing_vectors, axis=0)
        mean_pointing_vector /= np.linalg.norm(mean_pointing_vector)
        T_mean[:-1,0] = mean_pointing_vector
        T_mean[:-1,1] = np.array([-mean_pointing_vector[1], mean_pointing_vector[0]])
    else:
        raise Exception('Ts.shape must be (N,3,3) or (N,4,4)')
    
    return T_mean
