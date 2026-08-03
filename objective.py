import numpy as np

from scipy.spatial import KDTree


def calculate_objective_3d(cloud1, cloud2, threshold, norm, deg_threshold=3):
    """Return objective for cloud1 and cloud2"""
    mask1, objective, b_indices = check_visibility(cloud1, cloud2, threshold)
    area = calculate_surface_area(mask1, cloud1, np.zeros(3), deg_threshold)
    #print(objective, area, norm, sum(mask1))
    return -0.0001 * objective + area, objective, area, mask1, b_indices
    #return -0.0001 * objective * sum(mask1)/len(cloud1) + area, objective, area
    #return -0.0001 * objective * norm + area, objective, area


def calculate_surface_area(keep_mask, cloud, pose, deg_threshold=3):
    """Return surface area for cloud given keep mask"""
    common_points = cloud[keep_mask] - pose
    norm_common_points = common_points / np.linalg.norm(common_points, axis=1).reshape(-1, 1)
    tree = KDTree(norm_common_points)
    _, results = tree.query(norm_common_points, k=9, distance_upper_bound=np.sin(np.deg2rad(deg_threshold)), workers=3)
    total = 0 
    triangles = 0
    good_mask = results <= np.arange(results.shape[0]).reshape(-1, 1)
    for i in range(1, 9):
        for j in range(i+1, 9):
            curr_mask = good_mask[:,i] & good_mask[:,j]
            temp = results[curr_mask]
            diff = np.linalg.norm(common_points[temp[:,i]] - common_points[temp[:,j]], axis=1)
            mask = diff <= np.sin(np.deg2rad(deg_threshold))
            temp = temp[mask]
            side1 = common_points[temp[:,0]] - common_points[temp[:,j]]
            side2 = common_points[temp[:,0]] - common_points[temp[:,i]]
            areas = np.linalg.norm(np.cross(side1, side2), axis=1)
            total += np.sum(areas)/2
    return total


def check_visibility(pointsA, pointsB, thresh):
    """Check visibility"""
    kdtree = KDTree(pointsB)
    dist, idxs = kdtree.query(pointsA)
    maskA = dist < thresh
    objective = np.sum(dist[maskA])
    b_indices = idxs.flatten()[maskA]
    return maskA, objective, b_indices 


def count_common_area(pointsA, pointsB, thresh):
    """Check visibility"""
    kdtree = KDTree(pointsB)
    dist, _ = kdtree.query(pointsA, workers=3)
    maskA = dist <= thresh
    return sum(maskA)


def kl_divergence(pA, pB):
    if pA.shape[0] < 3 or pB.shape[0] < 3:
        return np.nan
        mA = np.mean(pA, axis=0, keepdims=True)
        mB = np.mean(pB, axis=0, keepdims=True)
        return np.linalg.norm(mA - mB)
    mA = np.mean(pA, axis=0, keepdims=True)
    mB = np.mean(pB, axis=0, keepdims=True)
    temp = pA - mA
    cA = temp.T @ temp
    temp = pB - mB
    cB = temp.T @ temp
    if np.linalg.det(cB) == 0:
        return np.nan

    icB = np.linalg.inv(cB)
    kl = (mA - mB) @ icB @ (mA-mB).T + np.trace(icB @ cA) - np.log(np.linalg.det(cA) / np.linalg.det(cB)) - 3
    return kl.item() / 2


def kl_divergence_histogram(pA, pB):
    p, edges = np.histogramdd(pA.T, bins=5, range=[[-.1,.1],[-.1,.1],[-.1,.1]])
    q, edges = np.histogramdd(pB.T, bins=5, range=[[-.1,.1],[-.1,.1],[-.1,.1]])
    p = p.flatten() / pA.shape[0]
    q = q.flatten() / pB.shape[0]
    m = (p + q) / 2
    temp1 = np.nansum(p * np.log(p/m))
    temp2 = np.nansum(q * np.log(q/m))

    return (temp1 + temp2) / 2


def check_neighborhood(pointsA, pointsB, triangleA, triangleB, thresh):
    icp_total = 0
    points_total = 0
    kls = []
    for i in range(4):
        maska = np.linalg.norm(pointsA - triangleA[i], axis=1) < thresh
        maskb = np.linalg.norm(pointsB - triangleB[i], axis=1) < thresh
        tree = KDTree(pointsA[maska])
        dist, _ = tree.query(pointsB[maskb], distance_upper_bound=0.02)
        icp_total += np.sum(dist[~np.isinf(dist)])
        points_total += len(dist) - np.sum(np.isinf(dist))
        kls.append(kl_divergence(pointsA[maska], pointsB[maskb]))
    return icp_total/points_total, points_total, kls


def check_histogram(pointsA, pointsB, triangleA, triangleB, thresh):
    kls = 0
    for i in range(4):
        maska = np.linalg.norm(pointsA - triangleA[i], axis=1) < thresh
        maskb = np.linalg.norm(pointsB - triangleB[i], axis=1) < thresh
        kl = kl_divergence(pointsA[maska], pointsB[maskb])
        rank = np.clip((np.ceil(np.log10(kl))), a_min=-2, a_max=5)
        if np.isnan(rank):
            kls += 5
        else:
            kls += rank
    return kls/4


def check_histogram2(pointsA, pointsB, triangleA, triangleB, thresh):
    kls = 0
    for i in range(4):
        maska = np.linalg.norm(pointsA - triangleA[i], axis=1) < thresh
        maskb = np.linalg.norm(pointsB - triangleB[i], axis=1) < thresh
        kl = kl_divergence(pointsA[maska], pointsB[maskb])
        kls += kl
        #rank = np.clip((np.ceil(np.log10(kl))), a_min=-2, a_max=5)
        #if np.isnan(rank):
        #    kls += 5
        #else:
        #    kls += rank
        #print(kl)
    return kls/4


def check_pasta(pointsA, pointsB, triangleA, triangleB, T, thresh):
    """
    maska = 0
    maskb = 0
    for i in range(4):
        maska += np.linalg.norm(pointsA - triangleA[i], axis=1) < thresh
        maskb += np.linalg.norm(pointsB - triangleB[i], axis=1) < thresh
    maska = maska>0
    maskb = maskb>0
    """
    tca = triangleA.mean(axis=0)
    tcb = triangleB.mean(axis=0)
    radius = max(np.concatenate([np.linalg.norm(triangleA - tca, axis=1), np.linalg.norm(triangleB- tcb, axis=1)])) + 0.005 # usually 0.05
    maska = np.linalg.norm(pointsA - tca, axis=1) < radius
    maskb = np.linalg.norm(pointsB - tcb, axis=1) < radius
    
    ca = pointsA[maska].mean(axis=0).reshape(3, 1)
    cb = pointsB[maskb].mean(axis=0).reshape(3,1)
    #print("mean", ca.flatten(), (T[:-1,:-1] @ cb).T + T[:-1,-1])
    #print("mean", ca.flatten() - ((T[:-1,:-1] @ cb).T + T[:-1,-1]).flatten())

    temp = pointsA[maska].T - ca
    sA = (temp @ temp.T) / sum(maska)
    temp = pointsB[maskb].T - cb
    sB = (temp @ temp.T) / sum(maskb)
    #print(sA, sB)
    #print(ca, cb)

    centroid_diff = np.linalg.norm(ca.flatten() - ((T[:-1,:-1] @ cb) + T[:-1,-1:]).flatten())
    #print(ca.flatten(), ((T[:-1,:-1] @ cb) + T[:-1,-1:]).flatten())
    covar_diff = np.linalg.norm(sA - T[:-1,:-1] @ sB @ T[:-1,:-1].T)
    #print(sA, T[:-1,:-1] @ sB @ T[:-1,:-1].T)
    return centroid_diff, covar_diff


def check_pasta2(pointsA, pointsB, triangleA, triangleB, T, thresh):
    maska = 0
    maskb = 0
    for i in range(4):
        maska += np.linalg.norm(pointsA - triangleA[i], axis=1) < thresh
        maskb += np.linalg.norm(pointsB - triangleB[i], axis=1) < thresh
    maska = maska>0
    maskb = maskb>0
    
    ca = pointsA[maska].mean(axis=0).reshape(3, 1)
    cb = pointsB[maskb].mean(axis=0).reshape(3,1)

    temp = pointsA[maska].T - ca
    sA = (temp @ temp.T) / sum(maska)
    temp = pointsB[maskb].T - cb
    sB = (temp @ temp.T) / sum(maskb)
    #print(sA, sB)
    #print(ca, cb)

    centroid_diff = np.linalg.norm(ca - (T[:-1,:-1] @ cb) + T[:-1,-1:])
    #print(ca.flatten(), ((T[:-1,:-1] @ cb) + T[:-1,-1:]).flatten())
    covar_diff = np.linalg.norm(sA - T[:-1,:-1] @ sB @ T[:-1,:-1].T)
    #print(sA, T[:-1,:-1] @ sB @ T[:-1,:-1].T)
    return centroid_diff, covar_diff


def indicator_function(dists, thresh):
    """Indicator function is 1 when dist is less than thresh"""
    temp = np.exp(-100 * (dists-thresh))
    t = (-1/(temp+1) + 1) > 0.5
    return t
