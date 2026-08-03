import numpy as np
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from itertools import combinations
from math import factorial
import time
from typing import NamedTuple

from collections import namedtuple

from triangle_utils import (
        get_rotation_translation,
        get_centroid,
        get_relative_vector,
)
from rotation_utils import (
    apply_rotvec,
    rotation_norm,
    twist_norm,
    twist_mean,
)

from tqdm import tqdm

MatchSet = namedtuple("MatchSet", "twist_set index cost") 


class PerformanceAnalyzer:
    def __init__(self) -> None:
        self.samples = {}
        self.state = {}

    def start(self, label):
        self.state[label] = time.perf_counter_ns()

    def stop(self, label):
        now = time.perf_counter_ns()
        if label in self.samples:
            self.samples[label].append(now - self.state[label])
        else:
            self.samples[label] = [now - self.state[label]]

    def report(self):
        print('-------------------------------')
        print('Performance report:')
        for key, val in self.samples.items():
            print(f'{key}: mean: {np.mean(val)}, std: {np.std(val)}, ({np.mean(val)/np.mean(self.samples["total"]):.2%})')


def wrap_angle(angle, range=(-np.pi, np.pi)):
    return range[0] + np.mod(angle-range[0], range[1]-range[0])


def tessellation_edges(points, simplices):
    """This is significantly faster than the previous version, and filters for duplicate edges."""
    simplices_sorted = np.sort(simplices, axis=1)
    column_index_pairs = list(combinations(range(simplices.shape[1]), 2))
    lines = [np.stack((simplices_sorted[:,a], simplices_sorted[:,b]), axis=1) for a, b in column_index_pairs]
    lines = np.concatenate(lines, axis=0)
    lines = np.unique(lines, axis=0)
    lines = points[lines]
    return lines


def simplex_edge_lengths(points, simplices, sort=True):
    """The returned set of edge lengths (one set for each simplex) is a unique
    descriptor of the simplex up to a rigid transformation (congruence)."""
    pairs = np.array(list(combinations(range(simplices.shape[1]), 2)))
    edge_diffs = points[simplices[:,pairs[:,0]]] - points[simplices[:,pairs[:,1]]]
    edge_lengths = np.linalg.norm(edge_diffs, axis=2)
    if sort:
        edge_lengths = np.sort(edge_lengths, axis=1)
    
    return edge_lengths


def simplex_volumes(points, simplices):
    origins = points[simplices[:,0]]
    verts = points[simplices[:,1:]] - origins[:, None,:]
    volumes = np.absolute(np.linalg.det(verts))/factorial(points.shape[1])
    return volumes


def direction_signs(directions, points):
    """This is a helper function used to construct a unique frame attached to a simplex."""
    dot_products = np.einsum('ijk,ilk->ijl', directions, points)
    hi = np.max(dot_products, axis=2)
    lo = np.min(dot_products, axis=2)
    signs = (hi >= np.abs(lo))*2.0 - 1.0
    return signs


def simplex_frames(points, simplices, twist_matrix_output=False):
    """Computes a frame of reference that is unique for any simplex with no symmetries."""
    centroids = np.mean(points[simplices], axis=1)
    vert_diffs = points[simplices] - centroids[:,None,:]
    covariances = np.einsum('ijk,ijl->ikl', vert_diffs, vert_diffs)
    U, S, _ = np.linalg.svd(covariances)

    # Now need to ensure I pick the right directions
    U[:,:,0] *= np.linalg.det(U)[:,None]
    U = np.swapaxes(U, 1, 2) # Direction convention is sets of stacks of row vectors
    signs = direction_signs(U, vert_diffs)
    U[:,:-1,:] *= signs[:,:-1,None]
    last_swap = (np.sum(signs[:,:-1] < 0, axis=1) % 2) == 0
    last_swap = (last_swap > 0) * 2.0 - 1
    U[:,-1,:] *= last_swap[:,None]
    U = np.swapaxes(U, 1, 2) # Go back to stacks of rotation matrices

    if twist_matrix_output:
        twists = np.empty((len(simplices), points.shape[1]+1, points.shape[1]+1))
        twists[:,:-1:,:-1] = U
        twists[:,:-1,-1] = centroids
        twists[:,-1,:-1] = 0.0
        twists[:,-1,-1] = 1.0
        return twists
    else:
        return centroids, U


# def rotation_distance(R1, R2):
#     R_diff = np.swapaxes(R1, 1, 2) @ R2
#     rotvecs = Rotation.from_matrix(R_diff).as_rotvec()
#     angles = np.linalg.norm(rotvecs)
#     return angles


    

def filter_simplex_sizes(points, simplices, size_threshold, max_simplex_num=None):
    volumes = simplex_volumes(points, simplices)
    volume_sort_indices = np.argsort(volumes)[::-1]
    sorted_volumes = volumes[volume_sort_indices]
    mask = sorted_volumes >= size_threshold
    filtered_volume_indices = volume_sort_indices[mask]

    if max_simplex_num is None:
        return simplices[filtered_volume_indices]
    else:
        return simplices[filtered_volume_indices[:max_simplex_num]]


def match_simplex_shapes(edge_lengths_1, edge_lengths_2, shape_distance_threshold, sort_matches=False, normalize=False):
    tree_1, tree_2 = KDTree(edge_lengths_1), KDTree(edge_lengths_2)
    results = tree_1.query_ball_tree(tree_2, shape_distance_threshold)
    matches = np.array([(i, j) for i in range(len(tree_1.data)) for j in results[i]])

    if len(matches) <= 0:
        return matches, np.array([])
    
    distances = np.linalg.norm(edge_lengths_1[matches[:,0]] - edge_lengths_2[matches[:,1]], axis=1)

    if normalize:
        distances /= (np.sum(edge_lengths_1[matches[:,0]], axis=1) + np.sum(edge_lengths_2[matches[:,1]], axis=1))/2

    if sort_matches:
        sorted_indices = np.argsort(distances)
        matches = matches[sorted_indices]
        distances = distances[sorted_indices]

    return matches, distances


def approx_so3_distance(r1, r2):
    return 1 - np.dot(r1.as_quat(),  r2.as_quat()) **2


def transform_metric(T1, T2):
    rotation_distance = approx_so3_distance(
            Rotation.from_matrix(T1[:-1,:-1]), Rotation.from_matrix(T2[:-1,:-1])
    )
    translation_distance = np.linalg.norm(T1[:-1,-1]- T2[:-1,-1])
    return rotation_distance + translation_distance


def group_matches_by_geo_consistency(twists1, twists2, matches, twist_distance_threshold):
    if len(matches) <= 0:
        return [], []
    
    twists1_inv = np.linalg.inv(twists1)
    twists = twists2[matches[:,1]] @ twists1_inv[matches[:,0]]
    match_sets = []
    twist_sets = []
    touched = set() # Do not consider matches that have already been grouped
    for i, (a, b), T1 in zip(range(len(matches)), matches, twists):
        if (a,b) in touched:
            continue
        match_sets.append([(a,b)])
        twist_sets.append([T1])
        touched.add((a,b))
        for (c, d), T2 in zip(matches[i+1:], twists[i+1:]):
            pos_dist, rot_dist = twist_norm(np.expand_dims(np.linalg.solve(T1, T2), axis=0))
            if pos_dist < twist_distance_threshold[0] and rot_dist < twist_distance_threshold[1]:
                match_sets[-1].append((c,d))
                twist_sets[-1].append(T2)
                touched.add((c,d))

    return match_sets, twist_sets


def extract_largest_match_set(match_sets, mode='count', volumes1=None, volumes2=None):
    """Placeholder for now. Need to replace with volume criterion."""
    if len(match_sets) <= 0:
        return np.array([]), None
    
    if mode == 'count':
        set_lengths = [len(match_set) for match_set in match_sets]
        largest_set_index = np.argmax(set_lengths)
        return np.array(match_sets[largest_set_index]), largest_set_index
    
    elif mode == 'volume':
        set_volumes = [[volumes1[a] + volumes2[b] for (a, b) in match_set] for match_set in match_sets]
        set_volumes = [np.sum(volumes) for volumes in set_volumes]
        largest_set_index = np.argmax(set_volumes)
        return np.array(match_sets[largest_set_index]), largest_set_index

    else:
        raise Exception('Mode must be "count" or "volume".')




class ProcessedPoints:
    def __init__(self, points) -> None:
        self.tess = Delaunay(points)
        self.edge_lengths = simplex_edge_lengths(self.tess.points, self.tess.simplices, sort=True)
        self.volumes = simplex_volumes(self.tess.points, self.tess.simplices)
        self.twists = simplex_frames(self.tess.points, self.tess.simplices, twist_matrix_output=True)
        self.lc = Line3DCollection(tessellation_edges(self.tess.points, self.tess.simplices), linewidths=0.5)


def preprocess_points(points, radius, flatness_threshold):
    tree = KDTree(points)
    results = tree.query_ball_tree(tree, radius)
    keep = np.full(len(results), True)

    for i in range(len(results)):
        neighbors = tree.data[results[i]]
        neighbors -= np.mean(neighbors, axis=0)
        covariance = np.einsum('jk,jl->kl', neighbors, neighbors) / len(neighbors)
        _, S, _ = np.linalg.svd(covariance)
        if np.min(S) < flatness_threshold:
            keep[i] = False

    new_points = points[keep]
    return new_points


def ICP_cost_full(cloud_a, cloud_b, threshold=np.inf):
   if type(cloud_a) is KDTree:
      tree_a = cloud_a
   else:
      tree_a = KDTree(cloud_a)

   if type(cloud_b) is KDTree:
      tree_b = cloud_b
   else:
      tree_b = KDTree(cloud_b)

   d, i = tree_b.query(tree_a.data)
   cost = np.sqrt(np.mean(d[d < threshold]**2))
   return cost, d


def tessellation_lines(points, simplices):
    segments = tessellation_edges(points, simplices)
    lines = LineCollection(segments, linewidths=0.5)
    return lines


class ProcessedPoints(NamedTuple):
    tessellation: Delaunay
    points: np.ndarray
    simplices: np.ndarray
    twists: np.ndarray
    edges: np.ndarray


def process_points(points, simplex_size_thr, simplex_max_num):
    tess = Delaunay(points)
    simplices = filter_simplex_sizes(tess.points, tess.simplices, simplex_size_thr, simplex_max_num)
    twists = simplex_frames(tess.points, simplices, twist_matrix_output=True)
    edges = simplex_edge_lengths(tess.points, simplices, sort=True)

    return ProcessedPoints(tess, points, simplices, twists, edges)


class SimplexMatchParams(NamedTuple):
    shape_dist_thr: float
    normalize_match_shape_dist: bool
    twist_dist_thr: float
    mode: str = 'volume'


class SimplexMatchResult(NamedTuple):
    largest_match_set: np.ndarray
    largest_match_set_index: int
    twist_sets: np.ndarray
    match_distances: np.ndarray
    

def find_matching_simplices(
    processed_src: ProcessedPoints,
    processed_dst: ProcessedPoints,
    params: SimplexMatchParams
) -> SimplexMatchResult:
    
    matches, match_distances = match_simplex_shapes(processed_src.edges, processed_dst.edges, params.shape_dist_thr, True, params.normalize_match_shape_dist)
    match_sets, twist_sets = group_matches_by_geo_consistency(processed_src.twists, processed_dst.twists, matches, params.twist_dist_thr)

    if params.mode == 'count':
        largest_match_set, largest_match_set_index = extract_largest_match_set(match_sets, params.mode)
    elif params.mode == 'volume':
        volumes1 = simplex_volumes(processed_src.points, processed_src.simplices)
        volumes2 = simplex_volumes(processed_dst.points, processed_dst.simplices)
        largest_match_set, largest_match_set_index = extract_largest_match_set(match_sets, params.mode, volumes1, volumes2)
    else:
        raise Exception('Argument "mode" must be "count" or "volume".')
    
    return SimplexMatchResult(largest_match_set, largest_match_set_index, twist_sets, match_distances)


class TransformResult(NamedTuple):
    twist: np.ndarray
    R: np.ndarray
    p: np.ndarray
    icp_cost: float
    icp_cost_70: float
    icp_distribution: np.ndarray
    rejected: bool


def construct_transform_from_matches(points_src, points_dst, simplex_match_result, icp_70_thr, simplex_min_num):
    if len(simplex_match_result.largest_match_set) >= simplex_min_num:
        T_mean = twist_mean(np.array(simplex_match_result.twist_sets[simplex_match_result.largest_match_set_index]))
        R_hat, p_hat = T_mean[:-1,:-1], T_mean[:-1,-1]

        T_icp = np.linalg.inv(T_mean)
        R_icp, p_icp = T_icp[:-1,:-1], T_icp[:-1,-1]
        cost, icp_dists = ICP_cost_full(points_src, np.einsum('ij,kj->ki', R_icp, points_dst) + p_icp)
        icp_70 = np.percentile(icp_dists, 70)
        reject = True if icp_70 > icp_70_thr else False

        return TransformResult(T_mean, R_hat, p_hat, cost, icp_70, icp_dists, reject)

    else:
        return None
