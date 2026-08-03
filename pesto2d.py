import numpy as np

from collections import namedtuple

from rotation_utils import (
        twist_mean,
)
from triangle_utils import (
        TrianglePoints,
)
from utils2d import (
        get_match2d,
        get_match2d_debug,
        get_match2d_debug_hierarchy,
        just_triangle_matching
)

PASTAResult = namedtuple("PASTAResult", "status T match_set")


def scale_point_cloud(cloud, scale):
    mask = np.linalg.norm(cloud, axis=1) <= scale
    return cloud[mask] / scale 


def use_pasta_triangle(src, dst, **kwargs):
    # Setting constants
    gap_threshold = kwargs.pop('gap_threshold', 0.02)
    scaling_constant = kwargs.pop('scaling_constant', 5)
    shape_distance_threshold = kwargs.pop('shape_distance_threshold', 0.03)
    twist_distance_threshold = kwargs.pop(
            'twist_distance_threshold', (0.07, 7/180*np.pi)
    )
    cost_threshold = kwargs.pop('cost_threshold', 20)
    #print(src.shape, dst.shape)

    # scale clouds to keep cost similar magnitude for different environments
    if scaling_constant is None:
        scaling_constant = 1/np.max(np.linalg.norm(src, axis=1))
    #print("SCALING", scaling_constant)
    source = scale_point_cloud(src, scaling_constant)
    dest = scale_point_cloud(dst, scaling_constant)
    #print(source.shape, dest.shape)

    # Generate Triangle Vectors for both clouds
    triangle_points_src = TrianglePoints(source, gap_threshold=gap_threshold)
    triangle_points_dest = TrianglePoints(dest, gap_threshold=gap_threshold)
    simplices_src = triangle_points_src.simplices
    simplices_dest = triangle_points_dest.simplices

    match, T_hat, cost = get_match2d(source, dest, triangle_points_src, triangle_points_dest, 0.05)
    T_hat[:-1,-1] *= scaling_constant
    status = 0
    i, j = match[0], match[1]
    return PASTAResult(status, T_hat, match), [], [], triangle_points_src[i] * scaling_constant, triangle_points_dest[j] * scaling_constant
