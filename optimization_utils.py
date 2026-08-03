import numpy as np

from collections import (
        defaultdict,
        namedtuple,
)
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from objective import (
        calculate_objective_3d,
        check_pasta,
)
from triangle_utils import (
        get_rotation_translation,
)

GAMMA = 0.5

MatchSet = namedtuple("MatchSet", "twist_set index cost") 


def apply_transformation(points, T):
    return (T[:-1,:-1] @ points.T).T + T[:-1,-1]


def approx_so3_distance(r1, r2):
    """Return a distance measurement between SO(3) rotations r1 and r2"""
    return 1 - np.dot(r1.as_quat(),  r2.as_quat()) **2


def calculate_match_set(
        cloud1, cloud2, triangles_1, triangles_2, shape_distance_threshold, vis_thresh, deg_thresh
):
    matches, twists = group_matches_triangle_vector(
            triangles_1, triangles_2, shape_distance_threshold, cloud1, cloud2
    )
    #print("# of matches:", len(matches))
    if matches:
        match_set, mask, b_indices = extract_best_match_set(cloud1, cloud2, matches, twists, vis_thresh, deg_thresh)
        i, j, *_ = matches[match_set.index]
        #print("vector", np.linalg.norm(triangles_1.triangle_vectors[i] - triangles_2.triangle_vectors[j]))
        return 0, match_set, mask, b_indices, i, j
    else:
        return 1, None, None, None, None


def extract_best_match_set(
        cloud1, cloud2,
        matches, twists, threshold=0.02, deg_thresh=3
):
    if len(matches) <= 0:
        return MatchSet([], None, np.inf), [], []

    def call_objective(cloud1, cloud2, T, norm):
        #print("NORM", norm)
        R, p = T[:-1,:-1], T[:-1, -1]
        cloud2_hat = (R @ cloud2.T).T + p
        cost, k1, k2, mask, b_indices = calculate_objective_3d(
                cloud1, cloud2_hat, threshold, norm, deg_thresh
        )
        return cost, mask, b_indices

    idx, cost, mask, b_indices = pick_optimum(cloud1, cloud2, np.array(twists), call_objective, matches)

    return MatchSet(twists[idx], idx, cost), mask, b_indices


def group_matches_triangle_vector(
        triangle_1,
        triangle_2,
        shape_distance_threshold,
        pointsA, 
        pointsB,
):
    """Match triangles and return transformations for each pairs"""
    tree1, tree2 = KDTree(triangle_1.triangle_vectors), KDTree(triangle_2.triangle_vectors)
    results = tree1.query_ball_tree(tree2, shape_distance_threshold)

    tree1 = KDTree(triangle_1.triangle_vectors)
    dist, idx = tree1.query(triangle_2.triangle_vectors, k=3)
    top_k = np.argsort(dist.flatten())[:30]
    #print(top_k, len(dist))
    #print(dist.flatten()[top_k])
    #print("# of results", len(results))

    matches = []
    twist_sets = []
    for i in top_k:
        j = i // 3
        k = i % 3
        index = idx[j][k]
        if dist[j][k] == np.inf:
            continue
        tA = triangle_1[index]
        tB = triangle_2[j]
        rotation_matrix = get_rotation_translation(tA, tB)
        rotation_matrix = np.linalg.inv(rotation_matrix)
        twist_sets.append(rotation_matrix)
        cen, covar = check_pasta(
                pointsA, pointsB, tA, tB, rotation_matrix, 0.1
        )
        #print(cen, covar, cen + 10 * covar)
        #print(rotation_matrix)
        ca = np.mean(tA, axis=0)
        cb = np.mean(tB, axis=0)
        matches.append((index, j, (cen, covar), dist[j][k]))

    return matches, twist_sets


def transform_metric(T1, T2):
    """Returns distance between twist T1 and T2"""
    rotation_distance = approx_so3_distance(
            Rotation.from_matrix(T1[:-1,:-1]), Rotation.from_matrix(T2[:-1,:-1])
    )
    translation_distance = np.linalg.norm(T1[:-1,-1]- T2[:-1,-1])
    return rotation_distance + translation_distance


def transform_metric_list(T1, T):
    approx_dist = (3 - np.trace(T1[:-1,:-1].T @ T[:,:-1,:-1], axis1=1, axis2=2))/4
    translation_distance = np.linalg.norm(T1[:-1,-1] - T[:,:-1,-1], axis=1)
    return approx_dist + translation_distance


def pick_optimum(cloud1, cloud2, transforms, cost_fn, matches):
    """A version of trust region optimization for discrete set"""
    #print("# of matches", len(transforms))
    curr = 0
    touched = np.zeros(len(transforms))
    touched = np.zeros(5)
    #curr_cost = cost_fn(cloud1, cloud2, transforms[curr])
    touched[curr] = 1
    if len(transforms) == 1:
        return curr, curr_cost
    trust_region = 0.25 # np.deg2rad(30) + 0.25 #np.nanmean(dists)

    #print("costs")
    costs = []
    cAs, cBs = [], []
    kls = []
    dists = []
    for i, T in enumerate(transforms):
        #print(T)
        #print(cost_fn(cloud1, cloud2, T))
        #print(matches[i][-2:])
        #costs.append(cost_fn(cloud1, cloud2, T))
        cen, covar = matches[i][-2]
        dist = matches[i][-1]
        kls.append(cen ** 2 + 50 * covar + 10 * dist)
        dists.append(matches[i][-1])
    #top_k = np.argsort([-x[-1] for x in matches])[:5]


    kls = np.array(kls)

    sorting = sorted(np.arange(len(kls)), key=lambda x: kls[x])
    #print(sorting)
    top_k = sorting[:10]

    transforms = transforms[top_k]
    costs = []
    masks = []
    indices = []
    for i, T in enumerate(transforms):
        #print(T)
        cost, mask, idx = cost_fn(cloud1, cloud2, T, np.linalg.norm(T[:-1,-1]))
        costs.append(cost)
        masks.append(mask)
        indices.append(idx)
        #print(costs[-1], matches[top_k[i]][-2], matches[top_k[i]][-1], kls[top_k][i])
        #print(costs[-1], *matches[top_k[i]][-2], dists[top_k[i]], kls[top_k][i])
    #input()
    idx = np.argmax(costs)
    return top_k[idx], costs[idx], masks[idx], indices[idx]
