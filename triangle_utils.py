import numpy as np

from scipy.spatial import Delaunay


SIGNED_AREA_SCALE = 1
TRIANGLE_VECTOR_WEIGHT = np.array([1,2,2,2])
TRIANGLE_VECTOR_WEIGHT = 1#np.array([1,1,1,1])

class TrianglePoints:
    def __init__(self, points, gap_threshold=0.5):
        self.points = points
        self.dt = Delaunay(points)
        self.triangle_vectors, self.simplices, self.order = get_triangle_vectors(
                points, self.dt.simplices, gap_threshold
        )

    def __getitem__(self, index):
        return self.points[self.simplices[index][self.order[index]]]


def calculate_triangle_side_lengths(points, simplices):
    rolled = np.roll(simplices, 1, axis=1)
    return np.linalg.norm(points[simplices] - points[rolled], axis=2)


def get_centroid(triangle):
    return triangle.mean(axis=0)


def get_gap_ordering(points, simplices):
    side_lengths = calculate_triangle_side_lengths(points, simplices)
    gaps = abs(side_lengths - np.roll(side_lengths, -1, axis=1))
    max_gap = np.argmax(gaps, axis=1)
    min_gap = np.argmin(gaps, axis=1)
    middle_gap = 3 - max_gap - min_gap
    ordered_gaps = np.stack([max_gap, middle_gap, min_gap]).T
    return ordered_gaps


def get_signed_area(points, simplices, gaps, index):
    area_matrix = (points[simplices[index][gaps[index, 0]]] - points[simplices[index][gaps[index, 1:]]]).T
    signed_area = np.linalg.det(area_matrix)
    return signed_area 


def get_relative_vector(tA, tB):
    c = get_centroid(tA)
    diff = tA - c
    d1 = diff[0] / np.linalg.norm(diff[0])
    d2 = diff[1] - np.dot(d1, diff[1]) * d1
    d2 /= np.linalg.norm(d2)
    R = np.stack([d1,d2])

    rA = (R @ tA.T).T
    rB = (R @ tB.T).T

    return rA - rB


def get_rotation_translation(triangleA, triangleB):
    cA = get_centroid(triangleA)
    cB = get_centroid(triangleB)
    A = triangleA - cA
    B = triangleB - cB
    H = A.T @ B
    U, S, V = np.linalg.svd(H)
    R = V.T @ U.T
    t = cB.reshape(-1, 1) - R @ cA.reshape(-1, 1)

    return get_transformation_matrix(R, t)


def get_transformation_matrix(R, t):
    if R.shape[0] == 2:
        A = np.zeros((3,3))
    else:
        A = np.zeros((4,4))
    A[:-1,:-1] = R
    A[:-1,-1:] = t
    A[-1,-1] = 1
    return A


def get_triangle_areas(points, simplices, threshold=0.5):
    simplices = remove_bad_triangles(points, simplices, threshold=threshold)
    ordered_gaps = get_gap_ordering(points, simplices)
    areas = []
    for i in range(len(simplices)):
        area = get_signed_area(points, simplices, ordered_gaps, i)
        areas.append(area)
    return areas


def get_triangle_vectors(points, simplices, threshold=0.5):
    simplices = remove_bad_triangles(points, simplices, threshold=threshold)
    ordered_gaps = get_gap_ordering(points, simplices)
    mask = np.full((len(simplices)), False)
    triangle_vectors = np.zeros((len(simplices), 4))
    for i in range(len(simplices)):
        area = get_signed_area(points, simplices, ordered_gaps, i)
        abs_area = abs(area)
        sign = (area> 0) * 2 -1

        p12_side_length = np.linalg.norm(points[simplices[i][ordered_gaps[i,0]]] - points[simplices[i][ordered_gaps[i,1]]]) #/ abs_area
        p23_side_length = np.linalg.norm(points[simplices[i][ordered_gaps[i,1]]] - points[simplices[i][ordered_gaps[i,2]]]) #/ abs_area
        p13_side_length = np.linalg.norm(points[simplices[i][ordered_gaps[i,0]]] - points[simplices[i][ordered_gaps[i,2]]]) #/ abs_area
        max_side = max(p12_side_length, p23_side_length, p13_side_length)
        triangle_vectors[i] = [sign * max_side, p12_side_length, p23_side_length, p13_side_length] * TRIANGLE_VECTOR_WEIGHT
        mask[i] = abs_area >= 5e-3
    return triangle_vectors[mask], simplices[mask], ordered_gaps[mask]


def remove_bad_triangles(points, simplices, threshold=0.5):
    side_lengths = calculate_triangle_side_lengths(points, simplices)
    gaps = abs(side_lengths - np.roll(side_lengths, -1, axis=1))
    mask = np.logical_not(np.any(abs(gaps) < threshold, axis=1))
    return simplices[mask]
