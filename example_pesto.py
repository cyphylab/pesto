import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.patches import Arrow, Circle

from pasta_utils import (
        CORRECTION_TYPE,
        align_cloud,
        angle_to_rot_matrix,
        load_clouds,
        load_clouds_scale,
        load_poses,
        load_poses_scale,
        pasta2d_rototranslation,
        pasta2d_bound,
)
from pesto2d import use_pasta_triangle
from rotation_utils import (
        twist_mean,
        twist_norm,
)
from draw_simplex_lib import (
        ICP_cost_full,
        simplex_edge_lengths,
        simplex_frames,
        tessellation_edges,
)
from triangle_utils import (
        TrianglePoints,
)

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default=0)
args = parser.parse_args()

if args.i == 0:
    FILE_POSE = 'data/U_shaped_obstacles_1/optitrack values.csv'
    FILE_CLOUD = 'data/U_shaped_obstacles_1/point_cloud.csv'
    #FILE_POSE = 'data/L_shaped_obstacles_1/optitrack values.csv'
    #FILE_CLOUD = 'data/L_shaped_obstacles_1/point_cloud.csv'
    #folder = 'more_data/lab/commonarea3'
    #FILE_POSE = f'{folder}/optitrack values.csv'
    #FILE_CLOUD = f'{folder}/point_cloud.csv'
    OPTITRACK_CORRECTION_TYPE = 'ANGLE'
else:
    FILE_POSE = 'data/pasta_normal_shape_walking_around/optitrack values.csv'
    FILE_CLOUD = 'data/pasta_normal_shape_walking_around/point_cloud.csv'
    OPTITRACK_CORRECTION_TYPE = 'A'

PLAY = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

GAP_THRESHOLD = 0.02
SIZE_THRESHOLD = 0.2
MAX_SIMPLEX_NUM = 20
SHAPE_DISTANCE_THRESHOLD = 0.03
PLOT_SKIP = 2
RPLIDAR_HZ = 5.5
ARROW_SCALE = 0.25
MAX_SIMPLEX_NUM = 20
NORMALIZE_MATCH_SHAPE_DISTANCE = True
TWIST_DISTANCE_THRESHOLD = (0.07, 7/180*np.pi)
MAX_ICP_COST_70 = 0.2
PAUSE_CONDITION = {
    'err_pos': np.inf,
    'err_rot': np.inf
}

A, b = CORRECTION_TYPE[OPTITRACK_CORRECTION_TYPE]
RADIUS = 3.5
_, reg_clouds = load_clouds(FILE_CLOUD)
depths, clouds, scale_factor = load_clouds_scale(FILE_CLOUD, scale_factor=RADIUS)
#depths, clouds = load_clouds_radius_scale(FILE_CLOUD, RADIUS)
poses = load_poses(FILE_POSE, A, b)
scaled_poses = load_poses_scale(FILE_POSE, A, b, scale_factor=1/RADIUS)
poses = poses[::PLOT_SKIP]
scaled_poses = scaled_poses[::PLOT_SKIP]
clouds = clouds[::PLOT_SKIP]
reg_clouds = reg_clouds[::PLOT_SKIP]
scale_factor = RADIUS

raw_clouds = clouds
clouds = [align_cloud(cloud, pose) for cloud, pose in zip(clouds, scaled_poses)]
reg_aligned_clouds = [align_cloud(cloud, pose) for cloud, pose in zip(reg_clouds, poses)]

fig = plt.figure(figsize=(8,5))
ax_ref = fig.add_subplot(1,4,1)
ax_dst = fig.add_subplot(1,4,2)
ax_compare = fig.add_subplot(1,4,3)
ax_pos = fig.add_subplot(2,4,4)
ax_rot = fig.add_subplot(2,4,8)
plt.tight_layout()

t = np.arange(len(clouds)) / RPLIDAR_HZ
err_pos = np.full(len(clouds), np.nan)
err_rot = np.full(len(clouds), np.nan)
err_pos_pasta = np.full(len(clouds), np.nan)
err_rot_pasta = np.full(len(clouds), np.nan)
pasta_position = np.full((len(clouds), 2), np.nan)
true_position = np.full((len(clouds), 2), np.nan)
tt = np.full((len(clouds), 2), np.nan)

# Setting up reference frame objects
triangle_points_ref = TrianglePoints(raw_clouds[0], gap_threshold=GAP_THRESHOLD)
tess_src = triangle_points_ref.dt
points_src = triangle_points_ref.points
simplices_src = triangle_points_ref.simplices

# Setting up reference frame matplotlib drawing components
twists_ref = simplex_frames(points_src, simplices_src, twist_matrix_output=True)
twists_draw_src = simplex_frames(clouds[0], simplices_src, twist_matrix_output=True)
edges_ref = simplex_edge_lengths(points_src, simplices_src, sort=True)
segments_ref = tessellation_edges(clouds[0], tess_src.simplices)
lines_src = LineCollection(segments_ref, linewidths=0.5)

# Setting up initial Transformation matrix
R_src, p_src = angle_to_rot_matrix(poses[0,2]/180*np.pi), poses[0,:2]
T_pose_src = np.block([[R_src, np.expand_dims(p_src, axis=1)], [np.zeros((1,2)), np.ones((1,1))]])
print(T_pose_src)

plt.ion()
#for i in range(len(clouds)):
for i in range(60):
    print('--------------------------')
    print(f'Scan: {i}')

    # Calculate triangle vectors for current frame
    triangles = TrianglePoints(raw_clouds[i], gap_threshold=GAP_THRESHOLD)
    tess = triangles.dt
    points = triangles.points
    simplices_dst = triangles.simplices

    # Set up current frame matplotlib drawing components
    twists = simplex_frames(points, simplices_dst, twist_matrix_output=True)
    twists_draw_dst = simplex_frames(clouds[i], simplices_dst, twist_matrix_output=True)
    edges = simplex_edge_lengths(points, simplices_dst, sort=True)
    segments = tessellation_edges(clouds[i], tess.simplices)
    lines_dst = LineCollection(segments, linewidths=0.5)

    # Get True transformation matrix from ground truth
    R_dst, p_dst = angle_to_rot_matrix(poses[i,2]/180*np.pi), poses[i,:2]
    T_pose_dst = np.block([[R_dst, np.expand_dims(p_dst, axis=1)], [np.zeros((1,2)), np.ones((1,1))]])
    T_true = np.linalg.solve(T_pose_dst, T_pose_src)

    triangle_result, k1, k2, tA, tB = use_pasta_triangle(reg_clouds[0], reg_clouds[i], scaling_constant=3.5, cost_threshold=200)
    tA, tB = tA/3.5, tB/3.5
    match_set = triangle_result.match_set

    true_position[i] = p_dst.reshape(2)
    pos_bound, ang_bound = None, None
    T_hat = triangle_result.T
    T_hat = np.linalg.inv(T_hat)
    print(T_hat)
    print(T_true)

    T_norm = twist_norm(np.expand_dims(np.linalg.solve(T_true, T_hat), axis=0))

    print(T_pose_dst)
    print(T_hat @ T_pose_src)
    print(T_pose_src @ np.linalg.solve(T_pose_src, T_pose_dst))
    err_pos[i], err_rot[i] = T_norm[0][0], T_norm[1][0]

    if err_pos[i] < 0.20:
        print("BOUND", len(k1), len(k2))
        print(reg_clouds[0].shape, reg_clouds[i].shape)
        """
        pos_bound, ang_bound = pasta2d_bound(k1, k2, poses[0], poses[i])
        pasta_position[i] = (T_pose_src @ np.linalg.inv(T_hat))[:2,2]
        """
        pasta_position[i] = np.array([np.nan, np.nan])
    print("POS", pasta_position[i], true_position[i])
    print(f'Err Pos: {err_pos[i]:.3f}m | Err Rot: {(err_rot[i]*180/np.pi):.1f}deg')
    #print(f'PASTA cost: {triangle_result.match_set.cost}')

    # BEGIN RAW PASTA
    R_pasta, p_pasta, _ = pasta2d_rototranslation(raw_clouds[0], raw_clouds[i], dir_mode='icp', icp_threshold=0.7)
    T_pasta = np.block([[R_pasta, np.expand_dims(p_pasta, axis=1)], [np.zeros((1,2)), np.ones((1,1))]])
    T_pasta = np.linalg.inv(T_pasta)
    T_icp = np.linalg.inv(T_pasta)
    R_icp, p_icp = T_icp[:-1,:-1], T_icp[:-1,-1]
    p_icp = p_icp * scale_factor
    cost_pasta, icp_dists_pasta = ICP_cost_full(raw_clouds[0], np.einsum('ij,kj->ki', R_icp, raw_clouds[i]) + p_icp)
    icp_70_pasta = np.percentile(icp_dists_pasta, 70)
    print(f'PASTA>> ICP cost: {cost_pasta:.4f} | ICP 70%: {icp_70_pasta:.4f}')
    if icp_70_pasta > MAX_ICP_COST_70:
        print('ICP cost too high. Ignoring alignment.')
    else:
        T_pasta[:2,2] *= scale_factor 
        T_norm = twist_norm(np.expand_dims(np.linalg.solve(T_true, T_pasta), axis=0))
        err_pos_pasta[i], err_rot_pasta[i] = T_norm[0][0], T_norm[1][0]
    print(f'PASTA>> Err Pos: {err_pos_pasta[i]:.3f}m | Err Rot: {(err_rot_pasta[i]*180/np.pi):.1f}deg')
    # END RAW PASTA

    frames_pos_src = twists_draw_src[:,:-1,-1]
    frames_rot_src = twists_draw_src[:,:-1,:-1]
    frames_pos_dst = twists_draw_dst[:,:-1,-1]
    frames_rot_dst = twists_draw_dst[:,:-1,:-1]

    R_draw_dst, p_draw_dst = angle_to_rot_matrix(scaled_poses[i,2]/180*np.pi), scaled_poses[i,:2]
    R_draw_src, p_draw_src = angle_to_rot_matrix(scaled_poses[0,2]/180*np.pi), scaled_poses[0,:2]
    ax_ref.clear()
    ax_ref.set(xlim=[-0.85, 0.85], ylim=[-0.75, 0.75], xlabel='X position [m]', ylabel='Y position [m]')
    ax_ref.add_collection(lines_src)
    ax_ref.scatter(clouds[0][:,0], clouds[0][:,1], s=2, color='tab:red', zorder=20)
    ax_ref.add_patch(Arrow(p_draw_src[0], p_draw_src[1], R_draw_src[0,0]*ARROW_SCALE, R_draw_src[1,0]*ARROW_SCALE, color='tab:red', width=0.1, zorder=5)),
    ax_ref.add_patch(Arrow(p_draw_src[0], p_draw_src[1], R_draw_src[0,1]*ARROW_SCALE, R_draw_src[1,1]*ARROW_SCALE, color='tab:green',  width=0.1, zorder=5))
    ax_ref.scatter(p_draw_src[0], p_draw_src[1], color='black', s=16, zorder=30)
    try:
        #polys = clouds[0][simplices_src[match_set.twist_set[:,0]]]
        polys = [align_cloud(tA, scaled_poses[0])]
        for poly in polys:
            ax_ref.fill(poly[:,0], poly[:,1], alpha=0.5, color='tab:orange', edgecolor=None)
    except:
        print("YO1")
        pass
    """
    for dim_index, c in zip(range(2), ['tab:red', 'tab:green']):
        x, y = frames_pos_src[:,0], frames_pos_src[:,1]
        u, v = frames_rot_src[:,0,dim_index]*ARROW_SCALE, frames_rot_src[:,1,dim_index]*ARROW_SCALE
        ax_ref.quiver(x, y, u, v, color=c, width=0.003)
    """

    ax_dst.clear()
    ax_dst.set(xlim=[-0.85, 0.85], ylim=[-0.75, 0.75], xlabel='X position [m]', ylabel='Y position [m]')
    ax_dst.add_collection(lines_dst)
    ax_dst.scatter(clouds[i][:,0], clouds[i][:,1], s=2, color='tab:red', zorder=20)
    ax_dst.add_patch(Arrow(p_draw_dst[0], p_draw_dst[1], R_draw_dst[0,0]*ARROW_SCALE, R_draw_dst[1,0]*ARROW_SCALE, color='tab:red', width=0.1, zorder=5)),
    ax_dst.add_patch(Arrow(p_draw_dst[0], p_draw_dst[1], R_draw_dst[0,1]*ARROW_SCALE, R_draw_dst[1,1]*ARROW_SCALE, color='tab:green',  width=0.1, zorder=5))
    ax_dst.scatter(p_draw_dst[0], p_draw_dst[1], color='black', s=16, zorder=30)
    try:
        #polys = clouds[i][simplices_dst[match_set.twist_set[:,1]]]
        polys = [align_cloud(tB, scaled_poses[i])]
        for poly in polys:
            ax_dst.fill(poly[:,0], poly[:,1], alpha=0.5, color='tab:orange', edgecolor=None)
    except:
        print("YO2")
        pass
    """
    for dim_index, c in zip(range(2), ['tab:red', 'tab:green']):
        x, y = frames_pos_dst[:,0], frames_pos_dst[:,1]
        u, v = frames_rot_dst[:,0,dim_index]*ARROW_SCALE, frames_rot_dst[:,1,dim_index]*ARROW_SCALE
        ax_dst.quiver(x, y, u, v, color=c, width=0.003)
    """

    ax_ref.set_aspect(1)
    ax_dst.set_aspect(1)

    ax_compare.clear()
    ax_compare.set(xlim=[-3, 3], ylim=[-3, 3], xlabel='X position [m]', ylabel='Y position [m]')
    ax_compare.plot(true_position[:,0], true_position[:,1], color='tab:orange', label='True', alpha=0.8)
    ax_compare.scatter(pasta_position[:,0], pasta_position[:,1], s=1.5, color='tab:blue', label='Simplex Pasta')
    ax_compare.scatter(reg_aligned_clouds[0][:,0], reg_aligned_clouds[0][:,1], s=2, color='tab:red', zorder=20)
    ax_compare.scatter(reg_aligned_clouds[50][:,0], reg_aligned_clouds[50][:,1], s=2, color='tab:red', zorder=20)
    ax_compare.scatter(reg_aligned_clouds[30][:,0], reg_aligned_clouds[30][:,1], s=2, color='tab:red', zorder=20)
    if pos_bound is not None:
        circle = Circle((pasta_position[i,0], pasta_position[i,1]), radius=pos_bound, alpha=0.5)
        ax_compare.add_patch(circle)

    ax_pos.clear()
    ax_pos.grid()
    ax_pos.bar(t[:i+1], err_pos[:i+1], width=0.2, color='tab:blue', alpha=0.8, zorder=20, label='Simplex Pasta')
    ax_pos.bar(t[:i+1], err_pos_pasta[:i+1], width=0.2, color='tab:orange', alpha=1.0, zorder=10, label='Raw Pasta')
    ax_pos.set_ylim(0.0, 0.30)
    ax_pos.set_xlim(t[0], t[-1])
    ax_pos.legend()
    ax_pos.set_xlabel('Time [s]')
    ax_pos.set_ylabel('Position Error [m]')
    
    ax_rot.clear()
    ax_rot.grid()
    ax_rot.bar(t[:i+1], err_rot[:i+1]/np.pi*180, width=0.2, color='tab:blue', alpha=0.8, zorder=20, label='Simplex Pasta')
    ax_rot.bar(t[:i+1], err_rot_pasta[:i+1]/np.pi*180, width=0.2, color='tab:orange', alpha=1.0, zorder=10, label='Raw Pasta')
    ax_rot.set_ylim(0.0, 20.0)
    ax_rot.set_xlim(t[0], t[-1])
    ax_rot.legend()
    ax_rot.set_xlabel('Time [s]')
    ax_rot.set_ylabel('Rotation Error [deg]')

    fig.suptitle(f'Scan: {i} | Time: {i/RPLIDAR_HZ:.2f}s | Pose: ({poses[i][0]:.2f}m, {poses[i][1]:.2f}m, {poses[i][2]:.0f}deg)')

    plt.pause(0.4)
    if not PLAY:
        n = input()
        if 'c' in n:
            PLAY = True

    if (not np.isnan(err_pos[i]) and err_pos[i] > PAUSE_CONDITION['err_pos']) or (not np.isnan(err_rot[i]) and err_rot[i] > PAUSE_CONDITION['err_rot']):
        fake_input = input()
    

plt.ioff()
plt.show()
