import argparse
import sys
import pandas as pd
import math
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation
from pesto3d import (
        run_pesto,
)


GAP_THRESHOLD = 0.001
SHAPE_DISTANCE_THRESHOLD = 0.03 # 0.03 normally
SHAPE_DISTANCE_THRESHOLD = 0.1 # 0.03 normally
TWIST_DISTANCE_THRESHOLD = (0.07, 7/180*np.pi)
VISIBILITY_THRESHOLD = 0.02
DEG_THRESHOLD = 1


def make_transformation_matrix(R, t):
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,3] = t
    T[3,3] = 1
    return T

def rotation_norm(R):
    """Returns the minimum rotation angle around all possible 3D rotation axes that achieves the rotation R."""
    if R.shape[-1] == 3:
        return np.linalg.norm(Rotation.from_matrix(R).as_rotvec(), axis=1)
    elif R.shape[-1] == 2:
        return np.abs(np.arctan2(R[:,1,0], R[:,0,0]))
    else:
        raise Exception('R.shape must be (N,2,2) or (N,3,3)')

def twist_norm(T_true, T_hat, T):
    """Returns the norm of the translation component and the rotation norm of the rotation component."""
    return np.array([np.linalg.norm(T_true[:-1,-1] - T_hat[:-1,-1]).item(), np.rad2deg(rotation_norm(T[:,:-1,:-1])).item()])


def random_rotation():
    u = np.random.rand(3)
    quat = np.array([
        np.sqrt(1-u[0]) * np.sin(2*np.pi * u[1]),
        np.sqrt(1-u[0]) * np.cos(2*np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2*np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2*np.pi * u[2]),
    ])
    return Rotation.from_quat(quat).as_matrix()


def random_translation():
    t = (np.random.rand(3) * 2 -1) * 0.5
    t /= np.linalg.norm(t)
    return t


def random_transformation():
    R = random_rotation()
    t = random_translation()
    return make_transformation_matrix(R, t)


NOISE_BOUND = 0.0005

np.random.seed(0)
point_file = 'example_data/bun_zipper_res3.ply'

src_cloud = o3d.io.read_point_cloud(point_file)
src = np.asarray(src_cloud.points)
radius = np.max(np.linalg.norm(src, axis=1))
cloud1 = src / radius
src_noise = cloud1 + (np.random.rand(len(src), 3) - 0.5) * 2 * NOISE_BOUND
cloud2 = np.array(src_noise)
            
T_true = random_transformation()
cloud2_hat = (T_true[:-1,:-1] @ cloud2.T).T + T_true[:-1,-1]

T_true = np.linalg.inv(T_true)

status, match_set, common_area1, common_area2, tA, tB = run_pesto(
        cloud1, cloud2_hat, GAP_THRESHOLD, SHAPE_DISTANCE_THRESHOLD,
        VISIBILITY_THRESHOLD, DEG_THRESHOLD
)
if status != 0:
    print("Oops! Something went wrong")
T_pesto = match_set.twist_set

err_pesto = twist_norm(T_true, T_pesto, np.expand_dims(np.linalg.solve(T_true, T_pesto), axis=0))
print("Error", err_pesto)
