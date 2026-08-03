import numpy as np

from scipy.spatial import Delaunay


class SimplexPoints:
    def __init__(self, points, gap_threshold=0.5):
        self.points = points
        self.dt = Delaunay(points)
        self.triangle_vectors, self.simplices, self.order = get_simplex_vectors(
                points, self.dt.simplices, gap_threshold
        )
        #print("SIMPLICES", len(self.dt.simplices), len(self.simplices))
        """
        self.areas = get_triangle_areas(
                points, self.dt.simplices, gap_threshold
        )
        """

    def __getitem__(self, index):
        #print(index, self.simplices[index], self.order[index])
        return self.points[self.simplices[index][self.order[index]]]


def calculate_simplex_side_lengths(points, simplices):
    p1 = points[simplices[:,[0,0,0,1,1,2]]]
    p2 = points[simplices[:,[1,2,3,2,3,3]]]
    diff = p1 - p2
    return np.linalg.norm(diff, axis=2)


def get_gap_ordering(points, simplices, sides, threshold):
    v1 = np.abs([sides[:,0] - sides[:,1], sides[:,0] - sides[:,2], sides[:,1] - sides[:,2]])
    one = np.min(v1, axis=0)
    v2 = np.abs([sides[:,0] - sides[:,3], sides[:,0] - sides[:,4], sides[:,3] - sides[:,4]])
    two = np.min(v2, axis=0)
    v3 = np.abs([sides[:,1] - sides[:,3], sides[:,1] - sides[:,5], sides[:,3] - sides[:,5]])
    three = np.min(v3, axis=0)
    v4 = np.abs([sides[:,2] - sides[:,4], sides[:,2] - sides[:,5], sides[:,4] - sides[:,5]])
    four = np.min(v4, axis=0)
    gaps = np.stack([one,two,three,four]).T
    good_simplices_mask = np.logical_not(np.any(gaps < threshold, axis=1))

    ordered_gaps = np.argsort(-gaps[good_simplices_mask], axis=1)
    return ordered_gaps, simplices[good_simplices_mask], good_simplices_mask


def get_signed_area(points, simplices, gaps, index):
    area_matrix = (points[simplices[index][gaps[index, 0]]] - points[simplices[index][gaps[index, 1:]]]).T
    signed_area = np.linalg.det(area_matrix)
    return signed_area 


def get_all_signed_areas(points, simplices, gaps):
    x = np.repeat(np.arange(simplices.shape[0]), 4)
    y = gaps.flatten()
    order = simplices[x,y].reshape(-1, 4)
    v1 = points[order[:,0]] - points[order[:,1]]
    v2 = points[order[:,0]] - points[order[:,2]]
    v3 = points[order[:,0]] - points[order[:,3]]
    m = np.swapaxes(np.stack([v1,v2,v3]), 0, 1)
    areas = np.linalg.det(m)
    mask = np.abs(areas) >= 1e-3
    #mask[i] = abs_area >= 1e-3
    sign = (areas > 0) * 2 -1
    return sign, mask


def get_simplex_areas(points, simplices, threshold=0.5):
    simplices = remove_bad_triangles(points, simplices, threshold=threshold)
    ordered_gaps = get_gap_ordering(points, simplices)
    areas = []
    for i in range(len(simplices)):
        area = get_signed_area(points, simplices, ordered_gaps, i)
        areas.append(area)
    return areas


def get_all_simplex_vectors(sides, ordered_gaps, signs):
    idx = np.array([
        [-1,0,1,2],
        [0,-1,3,4],
        [1,3,-1,5],
        [2,4,5,-1],
    ])
    x = np.repeat(np.arange(ordered_gaps.shape[0]), 6)
    y = np.tile([0,0,0,1,1,2], ordered_gaps.shape[0])
    z = np.tile([1,2,3,2,3,3], ordered_gaps.shape[0])
    order_x = ordered_gaps[x,y]
    order_y = ordered_gaps[x,z]
    i = idx[order_x,order_y]

    sides_vector = sides[x, i].reshape(-1, 6)
    max_sides = signs * np.max(sides, axis=1)
    sides_vector = np.concatenate([max_sides.reshape(-1,1), sides_vector], axis=1)
    return sides_vector
    
    
def get_simplex_vectors(points, simplices, threshold=0.5):
    idx = np.array([
        [-1,0,1,2],
        [0,-1,3,4],
        [1,3,-1,5],
        [2,4,5,-1],
    ])
    sides = calculate_simplex_side_lengths(points, simplices)
    ordered_gaps, simplices, mask2 = get_gap_ordering(points, simplices, sides, threshold)
    mask = np.full((len(simplices)), False)
    simplex_vectors = np.zeros((len(simplices), 7))
    sides = sides[mask2]
    signs, mask = get_all_signed_areas(points, simplices, ordered_gaps)
    simplex_vectors = get_all_simplex_vectors(sides, ordered_gaps, signs)
    return simplex_vectors[mask], simplices[mask], ordered_gaps[mask]


def get_triangles(points, simplices):
    normalized_points = points / np.linalg.norm(points, axis=1)
    simplices


def remove_bad_simplices(points, simplices, threshold=0.5):
    side_lengths = calculate_simplex_side_lengths(points, simplices)
    gaps = abs(side_lengths - np.roll(side_lengths, -1, axis=1))
    mask = np.logical_not(np.any(abs(gaps) < threshold, axis=1))
    return simplices[mask], mask
