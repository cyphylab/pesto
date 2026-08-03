import numpy as np
from math import factorial
from scipy.spatial import ConvexHull, HalfspaceIntersection, KDTree
from scipy.optimize import linprog


CORRECTION_TYPE = {
   'A': (np.array([[0,1,0],[-1,0,0],[0,0,1]]), np.zeros(3)),
   'B': (np.array([[0,-1,0],[-1,0,0],[0,0,1]]), np.array([0, 0, 180])),
   'C': (np.array([[0,-1,0],[-1,0,0],[0,0,1]]), np.zeros(3)),
   'ANGLE': (np.eye(3), np.array([0, 0, 180])),
   'DEFAULT':  (np.eye(3), np.zeros(3))
}

DATA_CORRECTIONS = {
   'coverage2_2022_09_04': CORRECTION_TYPE['B'],
   'coverage2_2022_09_08': CORRECTION_TYPE['B'],
   'coverage_2022_09_04': CORRECTION_TYPE['B'],
   'coverage_2022_09_08': CORRECTION_TYPE['B'],
   'coverage_obstacles2_2022_09_06': CORRECTION_TYPE['B'],
   'coverage_obstacles_2022_09_04': CORRECTION_TYPE['B'],
   'coverage_obstacles_2022_09_06': CORRECTION_TYPE['B'],
   'experiments_big_obstacles': CORRECTION_TYPE['A'],
   'experiments_big_obstacles_700_1': CORRECTION_TYPE['A'],
   'experiments_big_obstacles_700_2': CORRECTION_TYPE['A'],
   'experiments_no_obstacles': CORRECTION_TYPE['A'],
   'experiments_obstacles_700_1': CORRECTION_TYPE['A'],
   'experiments_obstacles_700_2': CORRECTION_TYPE['A'],
   'experiments_small_obstacles': CORRECTION_TYPE['A'],
   'many_obstacles_2022_09_12': CORRECTION_TYPE['B'],
   'moving_1': CORRECTION_TYPE['C'],
   'moving_2': CORRECTION_TYPE['C'],
   'obstacles_2022_09_09': CORRECTION_TYPE['B'],
   'pi4_irobot': CORRECTION_TYPE['B'],
   'pi4_irobot_04_18': CORRECTION_TYPE['B'],
   'pi4_irobot_04_18_bis': CORRECTION_TYPE['B'],
   'several_obstacles': CORRECTION_TYPE['C'],
   'several_obstacles_2': CORRECTION_TYPE['C'],
   'several_obstacles_3': CORRECTION_TYPE['C'],
   'L_shaped_obstacles_1': CORRECTION_TYPE['C'],
}


def wrap_angle(angle, interval=(-np.pi, np.pi)):
   return interval[0] + np.mod(angle + interval[0], interval[1] - interval[0])


def angle_to_rot_matrix(angle):
   return np.array([
      [np.cos(angle), -np.sin(angle)],
      [np.sin(angle), np.cos(angle)]
   ])


def rot_matrix_to_angle(R):
   R = np.clip(R, -1, 1)
   c, s = R[0,0], R[1,0]
   return np.arctan2(s, c)


def rototranslation_inverse(R, p):
   R_inv = np.transpose(R)
   p_inv = -R_inv @ p
   return R_inv, p_inv


def ICP_cost(cloud_a, cloud_b, threshold=np.inf):
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
   return cost


def depth_to_cartesian(depths, angle_range=(0.0, 2*np.pi)):
   n = depths.shape[0]

   angles = np.arange(depths.shape[1])*(angle_range[1]-angle_range[0])/depths.shape[1]
   rays = np.stack((np.cos(angles), np.sin(angles)), axis=1)

   clouds = []
   for i in range(n):
      cloud = rays * depths[i][:,None]
      cloud = cloud[depths[i] != np.inf, :]
      clouds.append(cloud)

   return clouds


def depth_to_cartesian_hallway(depths, angle_range=(0.0, 2*np.pi)):
   n = depths.shape[0]

   angles = np.arange(depths.shape[1])*(angle_range[1]-angle_range[0])/depths.shape[1]
   rays = np.stack((np.cos(angles), np.sin(angles)), axis=1)

   clouds = []
   for i in range(n):
      cloud = rays * depths[i][:,None]
      cloud = cloud[98:624]
      cloud = cloud[depths[i][98:624] != np.inf, :]
      clouds.append(cloud)

   return clouds


def load_clouds(file, angle_range=[0.0, 2*np.pi]):
   depths = np.loadtxt(file, delimiter=',', skiprows=1)[:,1:]
   clouds = depth_to_cartesian(depths, angle_range)
   return depths, clouds


def load_clouds_scale(file, scale_factor=1, angle_range=[0.0, 2*np.pi]):
   depths = np.loadtxt(file, delimiter=',', skiprows=1)[:,1:]
   scale_factor = np.nanmax(abs(depths[0][depths[0] != np.inf]))
   #depths *= 1/scale_factor
   clouds = depth_to_cartesian(depths, angle_range)
   clouds = [c /scale_factor for c in clouds]
   return depths, clouds, scale_factor


def load_clouds_scale_hallway(file, scale_factor=1, angle_range=[0.0, 2*np.pi]):
   depths = np.loadtxt(file, delimiter=',', skiprows=1)[:,1:]
   scale_factor = np.nanmax(abs(depths[0][depths[0] != np.inf]))
   print("SCALE_FACTOR", scale_factor)
   #depths *= 1/scale_factor
   clouds = depth_to_cartesian_hallway(depths, angle_range)
   #clouds = [c /scale_factor for c in clouds]
   return depths, clouds, scale_factor


def load_clouds_radius_scale(file, radius=1, angle_range=[0.0, 2*np.pi]):
   depths, clouds = load_clouds(file, angle_range)
   mask = depths > radius
   depths = depths / radius
   depths[mask] = np.inf
   clouds = depth_to_cartesian(depths, angle_range)
   return depths, clouds


def load_poses(file, A=np.eye(3), b=np.zeros(3), angle_range=(-180., 180.)):
   '''Performs affine correction to optitrack coords with angle wrapping'''
   poses = np.loadtxt(file, delimiter=',', skiprows=1, usecols=[1,2,3])
   poses = np.einsum('ij,kj->ki', A, poses) + b[None,:]
   poses[:,2] = wrap_angle(poses[:,2], angle_range)
   return poses


def load_poses_scale(file, A=np.eye(3), b=np.zeros(3), scale_factor=1, angle_range=(-180., 180.)):
   '''Performs affine correction to optitrack coords with angle wrapping'''
   poses = np.loadtxt(file, delimiter=',', skiprows=1, usecols=[1,2,3])
   poses = np.einsum('ij,kj->ki', A, poses) + b[None,:]
   poses[:,:2] *= scale_factor
   poses[:,2] = wrap_angle(poses[:,2], angle_range)
   return poses


def align_cloud(cloud, pose):
   translation = pose[:2]
   R = angle_to_rot_matrix(pose[2]/180*np.pi)
   return np.einsum('ij,kj->ki', R, cloud) + translation


def correct_optitrack_flip(poses, threshold=90.0):
   new_poses = np.copy(poses)
   for i in range(0,len(new_poses)-1):
      if abs(wrap_angle(new_poses[i+1,2] - new_poses[i,2], (-180., 180.))) > threshold:
         new_poses[i+1,2] = wrap_angle(new_poses[i+1,2] + 180., (-180., 180.))

   return new_poses


def shape_moments_2d(tris):
   # tris : resulting shape = (nb_simplices, 3, 2), for each simplex you have a 3 by 2 matrix containing the vertex + the next vertex + the hull centroid
   a = tris[:,1,:] - tris[:,0,:]  # roll(vertices) - vertices  (matrix of vectors)
   b = tris[:,2,:] - tris[:,0,:]  # vertices centroid - vertices  (matrix of vectors)

   areas = np.absolute(np.linalg.det(np.stack((a,b), axis=2)))  # here no need to divide by two since this is just the weighting for the average
   c = np.average(np.mean(tris, axis=1), axis=0, weights=areas)

   d = tris[:,0,:] - c[None,:]

   M = np.stack((d,a,b), axis=1)  # shape = (nb_vrtices, 3, 2)

   coeffs = np.array([
      [1,   1/3,  1/3],
      [1/3, 1/6,  1/12],
      [1/3, 1/12, 1/6]
   ])

   # To understand the following, check https://gharesifard.github.io/pdfs/IROS_2022_NOTE.pdf
   covs_11 = np.einsum("ijk,ijk->i", coeffs[None,:,:], np.einsum("ij,ik->ijk", M[:,:,0], M[:,:,0]))
   covs_12 = np.einsum("ijk,ijk->i", coeffs[None,:,:], np.einsum("ij,ik->ijk", M[:,:,0], M[:,:,1]))
   covs_21 = covs_12
   covs_22 = np.einsum("ijk,ijk->i", coeffs[None,:,:], np.einsum("ij,ik->ijk", M[:,:,1], M[:,:,1]))

   covs_1 = np.stack((covs_11,covs_12), axis=1)
   covs_2 = np.stack((covs_21,covs_22), axis=1)
   cov = np.stack((covs_1,covs_2), axis=2)
   cov = np.average(cov, axis=0, weights=areas)

   return c, cov


def directional_skewness(tris, p, w):
   def aux_n(x):
      if x == 0: return np.array([0,0])
      elif x == 1: return np.array([1,0])
      else: return np.array([0,1])

   coeff = np.empty((3,3,3))
   for i in range(3):
      for j in range(3):
         for k in range(3):
               n = aux_n(i) + aux_n(j) + aux_n(k)
               coeff[i,j,k] = factorial(n[0]) * factorial(n[1]) / factorial(2 + n[0] + n[1])
   coeff *= 2

   d = tris[:,0,:] - p[None,:]
   a = tris[:,1,:] - tris[:,0,:]
   b = tris[:,2,:] - tris[:,0,:]
   areas = np.absolute(np.linalg.det(np.stack((a,b), axis=2)))
   M = np.stack((d,a,b), axis=1)
   M = np.einsum("ijk,k->ij", M, w / np.linalg.norm(w))

   skews = np.einsum("ijkl,ijkl->i", coeff[None,:,:,:], np.einsum("ij,ik,il->ijkl", M, M, M))
   skewness = np.average(skews, weights=areas)

   return skewness


def uniformize(cloud, point_num):
   L = np.sum(np.linalg.norm(cloud-np.roll(cloud,1,axis=0), axis=1))
   n = cloud.shape[0]
   lpp = L/point_num

   new_cloud = np.empty((point_num,2))
   s = 0
   base_idx = 0
   vec_length = np.linalg.norm(cloud[(base_idx+1)%n]-cloud[base_idx])
   versor = (cloud[(base_idx+1)%n]-cloud[base_idx])/vec_length
   for i in range(new_cloud.shape[0]):
      new_cloud[i] = cloud[base_idx%n] + versor*s

      s += lpp
      while s >= vec_length:
         s -= vec_length
         base_idx += 1
         vec_length = np.linalg.norm(cloud[(base_idx+1)%n]-cloud[base_idx%n])
         versor = (cloud[(base_idx+1)%n]-cloud[base_idx%n])/vec_length
   
   return new_cloud


def rectify_basis(basis):
   if np.cross(basis[:,0], basis[:,1]) > 0:
      basis[:,0] = -basis[:,0]
   return basis


def choose_direction_max(mu, eigvec, cloud):
   dots = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,1])
   M = np.max(dots)
   m = np.min(dots)
   if abs(M) < abs(m):
      return rectify_basis(-eigvec)
   else:
      return rectify_basis(eigvec)


def choose_direction_skewness(mu, eigvec, tris):
   if directional_skewness(tris, mu, eigvec[:,1]) >= 0:
      return rectify_basis(eigvec)
   else:
      return rectify_basis(-eigvec)


def choose_direction_oracle(eigvec, oracle):
    if np.dot(eigvec[:,1], oracle[:,1]) >= 0:
        return rectify_basis(eigvec)
    else:
        return rectify_basis(-eigvec)


def cloud_moments(cloud):
   hull = ConvexHull(cloud)
   verts = cloud[hull.vertices,:]
   inner = np.mean(verts, axis=0)
   tris = np.stack((verts,np.roll(verts,1,axis=0),np.broadcast_to(inner, (verts.shape[0],2))), axis=1)

   mu, sigma = shape_moments_2d(tris)
   eigval, eigvec = np.linalg.eigh(sigma)

   return {
      'mu': mu,
      'sigma': sigma,
      'eigval': eigval,
      'eigvec': eigvec,
      'hull': hull,
      'vert': verts,
      'tris': tris,
      'cloud': cloud
   }


def pasta2d_rototranslation(cloud_A, cloud_B, dir_mode=None, **kwargs):
   A, B = [cloud_moments(cloud) for cloud in [cloud_A, cloud_B]]

   if dir_mode == 'max':
      A['eigvec'], B['eigvec'] = [choose_direction_max(x['mu'], x['eigvec'], x['cloud']) for x in [A, B]]
   elif dir_mode == 'skew':
      A['eigvec'], B['eigvec'] = [choose_direction_skewness(x['mu'], x['eigvec'], x['tris']) for x in [A, B]]
   elif dir_mode == 'icp':
      A['eigvec'], B['eigvec'] = [rectify_basis(x['eigvec']) for x in [A, B]]

   R = A['eigvec'] @ B['eigvec'].T
   p = A['mu'] - R @ B['mu']

   if dir_mode == 'icp':
      R_alt = -R
      p_alt = A['mu'] - R_alt @ B['mu']

      threshold = kwargs.pop('icp_threshold', np.inf)
      cost = ICP_cost(cloud_A, np.einsum('ij,kj->ki', R, cloud_B) + p, threshold)
      cost_alt = ICP_cost(cloud_A, np.einsum('ij,kj->ki', R_alt, cloud_B) + p_alt, threshold)
      if cost > cost_alt:
         R, p = R_alt, p_alt

   angle = rot_matrix_to_angle(R)
   if kwargs.pop('ret_icp_cost', False):
      return R, p, angle, min(cost, cost_alt)
   return R, p, angle


def linear_opt(equations, c):
   A, b = equations[:,:-1], -equations[:,-1]
   res = linprog(c, A, b)
   if res.status != 0:
      return None
   else:
      return res.x


def find_interior_point(equations):
   dim = equations.shape[1]-1
   x1 = linear_opt(equations, np.block([1,np.zeros(dim-1)]))
   x2 = linear_opt(equations, np.block([-1,np.zeros(dim-1)]))
   if x1 is not None and x2 is not None:
      return (x1+x2)/2
   else:
      return None


def intersect_hulls(hull_A, hull_B):
   equ = np.concatenate((hull_A.equations, hull_B.equations), axis=0)
   interior = find_interior_point(equ)
   if interior is not None:
      return ConvexHull(HalfspaceIntersection(equ, interior).intersections)
   else:
      return None


def rho_estimate(cloud):
   box_min = np.min(cloud, axis=0)
   box_max = np.max(cloud, axis=0)
   rho = np.linalg.norm(box_max - box_min)/2
   return rho


def bound(delta, rho, eigdelta_A, both=False):
   mean_perturb = 2*(1-delta)*rho
   eigval_perturb = (2*(1-delta)+4*(1-delta)**2)*rho**2
   eigdelta_B = eigval_perturb
   theta = np.arcsin(eigdelta_B/eigdelta_A)

   if not both:
      return theta
   else:
      return theta, mean_perturb


def bound_empirical(eigdelta_A, eigdelta_B):
   theta = np.arcsin(eigdelta_B/eigdelta_A)

   return theta


def pasta2d_bound(cloud_A, cloud_B, pose_A, pose_B):
   cloud_A = align_cloud(cloud_A, pose_A)
   cloud_B = align_cloud(cloud_B, pose_B)
   
   rho_A, rho_B = [rho_estimate(cloud) for cloud in [cloud_A, cloud_B]]
   hull_A, hull_B = [ConvexHull(cloud) for cloud in [cloud_A, cloud_B]]
   hull_intersect = intersect_hulls(hull_A, hull_B)

   # rho = max(rho_A, rho_B)
   rho = 4
   if hull_intersect is None:
      delta = 0
   else:
      delta = hull_intersect.volume / max(hull_A.volume, hull_B.volume)

   A = cloud_moments(cloud_A)
   eigdelta_A = np.abs(A['eigval'][1]-A['eigval'][0])
   ang_bound, e_p = bound(delta, rho, eigdelta_A, both=True)
   pos_bound = abs(np.sin(ang_bound))*np.linalg.norm(A['mu']) + e_p

   return pos_bound, ang_bound

