import numpy as np
from scipy.spatial import KDTree

from triangle_utils import (
        get_rotation_translation,
)


def calculate_objective(cloud1, cloud2, threshold):
    """Return objective for cloud1 and cloud2"""
    mask1, mask2, objective = check_visibility(cloud1, cloud2, threshold)
    area = calculate_keep_perimeter(mask1, cloud1)
    return -0.0001 * objective + area, objective, area


def calculate_keep_perimeter(keep_mask, cloud):
    distances = cloud_distance_between_points(cloud)
    if np.all(keep_mask):
        return np.sum(distances)
    int_keep_mask = keep_mask.astype(int)
    dist_mask = (int_keep_mask - np.roll(int_keep_mask, -1, axis=0)) == 1
    dist_mask = (~keep_mask) | dist_mask
    return np.sum(distances[~dist_mask])


def check_visibility(pointsA, pointsB, thresh):
    """Check visibility"""
    kdtree = KDTree(pointsB)
    dist, idx = kdtree.query(pointsA, distance_upper_bound=thresh)
    maskA = dist < thresh 
    maskB = np.zeros(len(pointsB))
    m = idx < len(pointsB)
    maskB[idx[m]] = 1
    objective = np.sum(dist[maskA])
    return maskA, maskB.astype(bool), objective


def cloud_distance_between_points(cloud):
    return np.linalg.norm(cloud - np.roll(cloud, -1, axis=0), axis=1)


def cost_fn(cloud1, cloud2, T, threshold):
    cloud2_hat = (T[:-1,:-1] @ cloud2.T).T + T[:-1,-1]
    return calculate_objective(cloud1, cloud2_hat, threshold)


def get_match2d(
        cloud1, cloud2,
        triangle_1,
        triangle_2,
        threshold
):
    tree1 = KDTree(triangle_1.triangle_vectors)
    dist, idx = tree1.query(triangle_2.triangle_vectors, k=3)
    top_k = np.argsort(dist.flatten())[:30]

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
                cloud1, cloud2, tA, tB, rotation_matrix, 0.1
        )
        ca = np.mean(tA, axis=0)
        cb = np.mean(tB, axis=0)
        matches.append((index, j, (cen, covar), dist[j][k]))

    transforms = np.array(twist_sets)
    costs = []
    cAs, cBs = [], []
    kls = []
    dists = []
    for i, T in enumerate(transforms):
        cen, covar = matches[i][-2]
        dist = matches[i][-1]
        kls.append(cen ** 2 + 50 * covar + 10 * dist)
        dists.append(matches[i][-1])
    kls = np.array(kls)
    sorting = sorted(np.arange(len(kls)), key=lambda x: kls[x])
    top_k = sorting[:10]
    transforms = transforms[top_k]
    costs = []
    for i, T in enumerate(transforms):
        cost, _, _ = cost_fn(cloud1, cloud2, T, threshold)
        #print(cost)
        costs.append(cost)
    idx = np.argmax(costs)
    return matches[top_k[idx]], transforms[idx], costs[idx]


def get_match2d_debug(
        cloud1, cloud2,
        triangle_1,
        triangle_2,
        threshold
):
    tree1 = KDTree(triangle_1.triangle_vectors)
    dist, idx = tree1.query(triangle_2.triangle_vectors, k=3)
    top_k = np.argsort(dist.flatten())[:30]

    matches = []
    twist_sets = []
    indices = []
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

        T = rotation_matrix
        cloud2_hat = (T[:-1,:-1] @ cloud2.T).T + T[:-1,-1]
        mask1, mask2, _ = check_visibility(cloud1, cloud2_hat, threshold)
        cen, covar = check_pasta(
                cloud1[mask1], cloud2[mask2], tA, tB, rotation_matrix, 0.1
        )
        ca = np.mean(tA, axis=0)
        cb = np.mean(tB, axis=0)
        matches.append((index, j, (cen, covar), dist[j][k], T, tA, tB, triangle_1.triangle_vectors[index], triangle_2.triangle_vectors[j]))

    transforms = np.array(twist_sets)
    costs = []
    cAs, cBs = [], []
    kls = []
    dists = []
    for i, T in enumerate(transforms):
        cen, covar = matches[i][-2]
        dist = matches[i][-1]
        kls.append(cen ** 2 + 50 * covar + 10 * dist)
        dists.append(matches[i][-1])
    kls = np.array(kls)
    sorting = sorted(np.arange(len(kls)), key=lambda x: kls[x])
    top_k = sorting[:10]
    transforms = transforms[top_k]
    costs = []
    for i, T in enumerate(transforms):
        cost, _, _ = cost_fn(cloud1, cloud2, T, threshold)
        #print(cost)
        costs.append(cost)
    idx = np.argmax(costs)
    return matches[top_k[idx]], transforms[idx], costs[idx], matches[top_k]


def just_triangle_matching(
        cloud1, cloud2,
        triangle_1,
        triangle_2,
        threshold,
        past
):
    tree1 = KDTree(triangle_1.triangle_vectors)
    tree2 = KDTree(triangle_2.triangle_vectors)

    twist_sets = []
    matches = []

    count = 0
    for p in past:
        tA = p[-4]
        tB = p[-3]
        vA = p[-2]
        vB = p[-1]
        T = p[-5]
        d = p[-6]
        dist, idx1 = tree1.query(vA)
        diff1 = np.linalg.norm(tA - triangle_1[idx1])
        dist, idx2 = tree2.query(vA)
        diff2 = np.linalg.norm(tB - triangle_2[idx2])
        if diff1 < 0.1 and diff2 < 0.1:
            twist_sets.append(T)
            matches.append((idx1, idx2, (-1,-1), d, T, tA, tB, vA, vB))
            count += 1
    dist, idx = tree1.query(triangle_2.triangle_vectors, k=3)
    top_k = np.argsort(dist.flatten())[:21]

    indices = []
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

        T = rotation_matrix
        cloud2_hat = (T[:-1,:-1] @ cloud2.T).T + T[:-1,-1]
        mask1, mask2, _ = check_visibility(cloud1, cloud2_hat, threshold)
        #cen, covar = check_pasta(
        #        cloud1[mask1], cloud2[mask2], tA, tB, rotation_matrix, 0.1
        #)
        #ca = np.mean(tA, axis=0)
        #cb = np.mean(tB, axis=0)
        matches.append((index, j, (-1, -1), dist[j][k], T, tA, tB, triangle_1.triangle_vectors[index], triangle_2.triangle_vectors[j]))
    return matches, count


def get_match2d_debug_hierarchy(
        cloud1, cloud2,
        triangle_1,
        triangle_2,
        threshold,
        past
):
    tree1 = KDTree(triangle_1.triangle_vectors)
    tree2 = KDTree(triangle_2.triangle_vectors)

    twist_sets = []
    matches = []

    count = 0
    for p in past:
        tA = p[-4]
        tB = p[-3]
        vA = p[-2]
        vB = p[-1]
        T = p[-5]
        d = p[-6]
        dist, idx1 = tree1.query(vA)
        diff1 = np.linalg.norm(tA - triangle_1[idx1])
        dist, idx2 = tree2.query(vA)
        diff2 = np.linalg.norm(tB - triangle_2[idx2])
        if diff1 < 0.1 and diff2 < 0.1:
            twist_sets.append(T)
            matches.append((idx1, idx2, (-1,-1), d, T, tA, tB, vA, vB))
            count += 1

    dist, idx = tree1.query(triangle_2.triangle_vectors, k=3)
    top_k = np.argsort(dist.flatten())[:20]
    indices = []
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

        T = rotation_matrix
        cloud2_hat = (T[:-1,:-1] @ cloud2.T).T + T[:-1,-1]
        mask1, mask2, _ = check_visibility(cloud1, cloud2_hat, threshold)
        cen, covar = check_pasta(
                cloud1[mask1], cloud2[mask2], tA, tB, rotation_matrix, 0.1
        )
        ca = np.mean(tA, axis=0)
        cb = np.mean(tB, axis=0)
        matches.append((index, j, (cen, covar), dist[j][k], T, tA, tB, triangle_1.triangle_vectors[index], triangle_2.triangle_vectors[j]))

    transforms = np.array(twist_sets)
    costs = []
    cAs, cBs = [], []
    kls = []
    dists = []
    for i, T in enumerate(transforms):
        cen, covar = matches[i][2]
        dist = matches[i][3]
        kls.append(cen ** 2 + 50 * covar + 10 * dist)
        dists.append(matches[i][3])
    kls = np.array(kls)
    sorting = sorted(np.arange(len(kls)), key=lambda x: kls[x])
    top_k = sorting[:10]
    transforms = transforms[top_k]
    costs = []
    for i, T in enumerate(transforms):
        cost, _, _ = cost_fn(cloud1, cloud2, T, threshold)
        #print(cost)
        costs.append(cost)
    idx = np.argmax(costs)
    #print(top_k[idx])
    #print(top_k, idx)
    return matches[top_k[idx]], transforms[idx], costs[idx], [matches[i] for i in top_k], count, top_k[idx] < count


def check_pasta(pointsA, pointsB, triangleA, triangleB, T, thresh):
    tca = triangleA.mean(axis=0)
    tcb = triangleB.mean(axis=0)
    radius = max(np.concatenate([np.linalg.norm(triangleA - tca, axis=1), np.linalg.norm(triangleB- tcb, axis=1)])) + 0.05 # usually 0.05
    maska = np.linalg.norm(pointsA - tca, axis=1) < radius
    maskb = np.linalg.norm(pointsB - tcb, axis=1) < radius
    
    ca = pointsA[maska].mean(axis=0).reshape(-1, 1)
    cb = pointsB[maskb].mean(axis=0).reshape(-1,1)

    temp = pointsA[maska].T - ca
    sA = (temp @ temp.T) / sum(maska)
    temp = pointsB[maskb].T - cb
    sB = (temp @ temp.T) / sum(maskb)

    centroid_diff = np.linalg.norm(ca.flatten() - ((T[:-1,:-1] @ cb) + T[:-1,-1:]).flatten())
    covar_diff = np.linalg.norm(sA - T[:-1,:-1] @ sB @ T[:-1,:-1].T)
    return centroid_diff, covar_diff
