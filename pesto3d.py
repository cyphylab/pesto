import numpy as np
from optimization_utils import calculate_match_set
from simplex_utils import SimplexPoints


def run_pesto(
        cloud0, cloud1, gap_threshold, shape_threshold, vis_threshold, deg_threshold
):
    simplex_points_ref = SimplexPoints(cloud0, gap_threshold=gap_threshold)
    tess_ref = simplex_points_ref.dt
    points_ref = simplex_points_ref.points
    simplices_ref = simplex_points_ref.simplices

    simplex_points_targ = SimplexPoints(cloud1, gap_threshold=gap_threshold)
    tess_targ = simplex_points_targ.dt
    points_targ = simplex_points_targ.points
    simplices_targ = simplex_points_targ.simplices

    status, match_set, mask, b_indices, i, j = calculate_match_set(
            cloud0, cloud1, simplex_points_ref, 
            simplex_points_targ, shape_threshold, vis_threshold, deg_threshold
    )
    return status, match_set, cloud0[mask], cloud1[b_indices], simplex_points_ref[i], simplex_points_targ[j]
