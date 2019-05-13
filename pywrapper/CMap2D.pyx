# distutils: language=c++

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
cimport cython
from math import sqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin

import os
from yaml import load
from matplotlib.pyplot import imread

import pose2d

cdef class CMap2D:
    cdef public np.float32_t[:,::1] occupancy_ # [:, ::1] means 2d c-contiguous
    cdef int occupancy_shape0
    cdef int occupancy_shape1
    cdef float resolution_
    cdef float thresh_occupied_
    cdef float thresh_free
    cdef float HUGE_
    cdef public np.float32_t[:] origin
    def __init__(self, folder=None, name=None):
        self.occupancy_ = np.ones((100, 100), dtype=np.float32) * 0.5
        self.occupancy_shape0 = 100
        self.occupancy_shape1 = 100
        self.resolution_ = 0.01
        self.origin = np.array([0., 0.], dtype=np.float32)
        self.thresh_occupied_ = 0.9
        self.thresh_free = 0.1
        self.HUGE_ = 1e10
        if folder is None or name is None:
            return
        # Load map from file
        folder = os.path.expanduser(folder)
        yaml_file = os.path.join(folder, name + ".yaml")
        print("Loading map definition from {}".format(yaml_file))
        with open(yaml_file) as stream:
            mapparams = load(stream)
        map_file = os.path.join(folder, mapparams["image"])
        print("Map definition found. Loading map from {}".format(map_file))
        mapimage = imread(map_file)
        mapimage = np.ascontiguousarray(
            1. - mapimage.T[:, ::-1] / 254.
        ).astype(np.float32)  # (0 to 1) 1 means 100% certain occupied
        self.occupancy_ = mapimage
        self.occupancy_shape0 = mapimage.shape[0]
        self.occupancy_shape1 = mapimage.shape[1]
        self.resolution_ = mapparams["resolution"]  # [meters] side of 1 grid square
        self.origin = np.array(mapparams["origin"][:2]).astype(np.float32)  # [meters] x y coordinates of point at i = j = 0
        if mapparams["origin"][2] != 0:
            raise ValueError(
                "Map files with a rotated frame (origin.theta != 0) are not"
                " supported. Setting the value to 0 in the MAP_NAME.yaml file is one way to"
                " resolve this."
            )
        self.thresh_occupied_ = mapparams["occupied_thresh"]
        self.thresh_free = mapparams["free_thresh"]
        self.HUGE_ = 100 * self.occupancy_shape0 * self.occupancy_shape1 # bigger than any possible distance in the map
        if self.resolution_ == 0:
            raise ValueError("resolution can not be 0")

    def resolution(self):
        res = float(self.resolution_)
        return res

    def thresh_occupied(self):
        res = float(self.thresh_occupied_)
        return res

    def as_occupied_points_ij(self):
        return np.ascontiguousarray(np.array(np.where(self.occupancy() > self.thresh_occupied())).T)

    cdef cas_tsdf(self, np.float32_t max_dist_m, np.int64_t[:,::1] occupied_points_ij, np.float32_t[:, ::1] min_distances):
        """ everything in ij units """
        cdef np.int64_t max_dist_ij = np.int64((max_dist_m / self.resolution_))
        cdef np.int64_t[:] point
        cdef np.int64_t pi
        cdef np.int64_t pj
        cdef np.float32_t norm
        cdef np.int64_t i
        cdef np.int64_t j 
        cdef np.int64_t iend
        cdef np.int64_t jend 

        for k in range(len(occupied_points_ij)):
            point = occupied_points_ij[k]
            pi = point[0]
            pj = point[1]
            i = max(pi - max_dist_ij, 0)
            iend = min(pi + max_dist_ij, min_distances.shape[0] - 1)
            j = max(pj - max_dist_ij, 0)
            jend = min(pj + max_dist_ij, min_distances.shape[1] - 1)
            while True:
                j = max(pj - max_dist_ij, 0)
                while True:
                    norm = sqrt((pi - i) ** 2 + (pj - j) ** 2)
                    if norm < min_distances[i, j]:
                        min_distances[i, j] = norm
                    j = j+1
                    if j >= jend: break
                i = i+1
                if i >= iend: break

    def as_tsdf(self, max_dist_m):
        if False:
            occupied_points_ij = self.as_occupied_points_ij()
            max_dist_ij = (max_dist_m / self.resolution_)
            min_distances_ij = np.ones((self.occupancy_shape0, self.occupancy_shape1), dtype=np.float32) * max_dist_ij
            self.cas_tsdf(max_dist_m, occupied_points_ij, min_distances_ij)
            # Change from i, j units to x, y units [meters]
            min_distances = min_distances_ij * self.resolution_
            # Switch sign for occupied and unkown points (*signed* distance field)
            min_distances[self.occupancy() > self.thresh_free] *= -1.
        # this is faster than the still poorly optimized cas_tsdf
        min_distances = self.as_sdf()
        min_distances[min_distances > max_dist_m] = max_dist_m
        min_distances[min_distances < -max_dist_m] = -max_dist_m
        return min_distances

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cxy_to_ij(self, np.float32_t[:,::1] xy, np.float32_t[:,::1] ij, bool clip_if_outside=True):
        if xy.shape[1] != 2:
            raise IndexError("xy should be of shape (n, 2)")
        for k in range(xy.shape[0]):
            ij[k, 0] = (xy[k, 0] - self.origin[0]) / self.resolution_
            ij[k, 1] = (xy[k, 1] - self.origin[1]) / self.resolution_
        if clip_if_outside:
            for k in range(xy.shape[0]):
                if ij[k, 0] >= self.occupancy_shape0:
                    ij[k, 0] = self.occupancy_shape0 - 1
                if ij[k, 1] >= self.occupancy_shape1:
                    ij[k, 1] = self.occupancy_shape1 - 1
                if ij[k, 0] < 0:
                    ij[k, 0] = 0
                if ij[k, 1] < 0:
                    ij[k, 1] = 0
        return ij

    def xy_to_ij(self, x, y=None, clip_if_outside=True):
        # if no y argument is given, assume x is a [...,2] array with xy in last dim
        """
        for each x y coordinate, return an i j cell index
        Examples
        --------
        >>> a = Map2D()
        >>> a.xy_to_ij(0.01, 0.02)
        (1, 2)
        >>> a.xy_to_ij([0.01, 0.02])
        array([1, 2])
        >>> a.xy_to_ij([[0.01, 0.02], [-0.01, 0.]])
        array([[1, 2],
               [0, 0]])
        """
        if y is None:
            return np.concatenate(
                self.xy_to_ij(
                    *np.split(np.array(x), 2, axis=-1), clip_if_outside=clip_if_outside
                ),
                axis=-1,
            )
        i = (x - self.origin[0]) / self.resolution_
        j = (y - self.origin[1]) / self.resolution_
        i = i.astype(int)
        j = j.astype(int)
        if clip_if_outside:
            i_gt = i >= self.occupancy_.shape[0]
            i_lt = i < 0
            j_gt = j >= self.occupancy_.shape[1]
            j_lt = j < 0
            if isinstance(i, np.ndarray):
                i[i_gt] = self.occupancy_.shape[0] - 1
                i[i_lt] = 0
                j[j_gt] = self.occupancy_.shape[1] - 1
                j[j_lt] = 0
            else:
                if i_gt:
                    i = self.occupancy_.shape[0] - 1
                if i_lt:
                    i = 0
                if j_gt:
                    j = self.occupancy_.shape[1] - 1
                if j_lt:
                    j = 0
        return i, j

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cij_to_xy(self, np.float32_t[:,::1] ij):
        xy = np.zeros([ij.shape[0], ij.shape[1]], dtype=np.float32)
        for k in range(ij.shape[0]):
            xy[k, 0] = ij[k, 0] * self.resolution_ + self.origin[0]
            xy[k, 1] = ij[k, 1] * self.resolution_ + self.origin[1]
        return xy

    def ij_to_xy(self, i, j=None):
        """
        Examples
        --------
        >>> a = Map2D()
        >>> a.ij_to_xy(1, 2)
        (0.01, 0.02)
        >>> a.ij_to_xy([1,2])
        array([0.01, 0.02])
        >>> a.ij_to_xy([[1,2], [-1, 0]])
        array([[ 0.01,  0.02],
               [-0.01,  0.  ]])
        """
        # if no j argument is given, assume i is a [...,2] array with ij in last dim
        if j is None:
            return np.concatenate(
                self.ij_to_xy(*np.split(np.array(i), 2, axis=-1)), axis=-1
            )
        x = i * self.resolution_ + self.origin[0]
        y = j * self.resolution_ + self.origin[1]
        return x, y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cis_inside_ij(self, np.float32_t[:,::1] ij):
        inside = np.ones([ij.shape[0],], dtype=np.bool)
        for k in range(ij.shape[0]):
            if ij[k, 0] >= self.occupancy_shape0:
                inside[k] = False
            if ij[k, 1] >= self.occupancy_shape1:
                inside[k] = False
            if ij[k, 0] < 0:
                inside[k] = False
            if ij[k, 1] < 0:
                inside[k] = False
        return inside

    def is_inside_ij(self, i, j=None):
        from functools import reduce
        """
        Examples
        --------
        >>> a = Map2D()
        >>> a.is_inside_ij(1, 2)
        True
        >>> a.is_inside_ij([1,2])
        array(True)
        >>> a.is_inside_ij([[1,2]])
        array([ True])
        >>> a.is_inside_ij([[1,a.occupancy_.shape[1]]])
        array([False])
        >>> a.is_inside_ij([[a.occupancy_.shape[0],2]])
        array([False])
        >>> a.is_inside_ij([[1,2], [-1, 0]])
        array([ True, False])
        """
        if j is None:
            return self.is_inside_ij(*np.split(np.array(i), 2, axis=-1))[..., 0]
        return reduce(
            np.logical_and,
            [i > 0, i < self.occupancy_.shape[0], j > 0, j < self.occupancy_.shape[1]],
        )

    def occupancy(self):
        occ = np.array(self.occupancy_)
        return occ

    def occupancy_T(self):
        occ_T = np.zeros((self.occupancy_shape1, self.occupancy_shape0), dtype=np.float32)
        for i in range(self.occupancy_shape1):
            for j in range(self.occupancy_shape0):
                occ_T[i, j] = self.occupancy_[j, i]
        return occ_T

    def as_sdf(self, raytracer=None):
        NUMBA = False
        RANGE_LIBC = True
        occupied_points_ij = np.array(self.as_occupied_points_ij())
        min_distances = np.ones(self.occupancy_.shape) * self.HUGE_
        if NUMBA:
#             compiled_sdf_math(occupied_points_ij, min_distances)
            pass
        if RANGE_LIBC:
            if raytracer is None:
                import range_libc

                pyomap = range_libc.PyOMap(self.occupancy_T() >= self.thresh_occupied_)
                rm = range_libc.PyRayMarching(pyomap, self.occupancy_shape0)
                min_distances = np.zeros((self.occupancy_shape0, self.occupancy_shape1), dtype=np.float32)
                rm.get_dist(min_distances)
            else:
                min_distances = raytracer.get_dist()
        # Change from i, j units to x, y units [meters]
        min_distances = min_distances * self.resolution_
        # Switch sign for occupied and unkown points (*signed* distance field)
        min_distances[self.occupancy() > self.thresh_free] *= -1.
        return min_distances

    cpdef as_coarse_map2d(self):
        coarse = CMap2D()
        if self.occupancy_shape0 % 2 != 0 or self.occupancy_shape1 % 2 != 0:
            raise IndexError("Shape needs to be divisible by 2 in order to make coarse map")
        coarse.occupancy_shape0 = self.occupancy_shape0 / 2
        coarse.occupancy_shape1 = self.occupancy_shape1 / 2
        coarse.occupancy_ = np.zeros((coarse.occupancy_shape0, coarse.occupancy_shape1), dtype=np.float32)
        for i in range(coarse.occupancy_shape0):
            for j in range(coarse.occupancy_shape1):
                coarse.occupancy_[i, j] = max(
                        self.occupancy_[i*2  , j*2  ],
                        self.occupancy_[i*2+1, j*2  ],
                        self.occupancy_[i*2  , j*2+1],
                        self.occupancy_[i*2+1, j*2+1],
                        )

        coarse.resolution_ = self.resolution_ * 2
        coarse.origin = np.array([0., 0.], dtype=np.float32)
        coarse.origin[0] = self.origin[0]
        coarse.origin[1] = self.origin[1]
        coarse.thresh_occupied_ = self.thresh_occupied_
        coarse.thresh_free = self.thresh_free
        coarse.HUGE_ = self.HUGE_
        return coarse


    def dijkstra(self, goal_ij, mask=None, extra_costs=None):
        kEdgeLength = 1 * self.resolution_  # meters
        # Initialize bool arrays
        open_ = np.ones((self.occupancy_shape0, self.occupancy_shape1), dtype=np.bool)
        not_in_to_visit = np.ones((self.occupancy_shape0, self.occupancy_shape1), dtype=np.bool)
        # Mask (close) unattainable nodes
        if mask is None:
            mask = self.occupancy() >= self.thresh_free
        open_[mask] = False
        # initialize extra costs
        if extra_costs is None:
            extra_costs = np.zeros((self.occupancy_shape0, self.occupancy_shape1))
        # initialize field to large value
        tentative = np.ones((self.occupancy_shape0, self.occupancy_shape1)) * self.HUGE_
        # Start at the goal location
        tentative[tuple(goal_ij)] = 0
        to_visit = [goal_ij]
        not_in_to_visit[tuple(goal_ij)] = False
        tentative[goal_ij[0], goal_ij[1]] = 0
        to_visit = [(goal_ij[0], goal_ij[1])]
        not_in_to_visit[goal_ij[0], goal_ij[1]] = False
        neighbor_offsets = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        len_i = tentative.shape[0]
        len_j = tentative.shape[1]
        while to_visit:
            # Make the current node that which has the smallest tentative values
            smallest_tentative_value = tentative[to_visit[0][0], to_visit[0][1]]
            smallest_tentative_id = 0
            for i in range(len(to_visit)):
                node_idx = to_visit[i]
                value = tentative[node_idx[0], node_idx[1]]
                if value < smallest_tentative_value:
                    smallest_tentative_value = value
                    smallest_tentative_id = i
            current = to_visit.pop(smallest_tentative_id)
            # Iterate over 4 neighbors
            for n in range(4):
                # Indices for the neighbours
                neighbor_idx = (
                    current[0] + neighbor_offsets[n][0],
                    current[1] + neighbor_offsets[n][1],
                )
                # Find which neighbors are open (exclude forbidden/explored areas of the grid)
                if neighbor_idx[0] < 0:
                    continue
                if neighbor_idx[0] >= len_i:
                    continue
                if neighbor_idx[1] < 0:
                    continue
                if neighbor_idx[1] >= len_j:
                    continue
                if not open_[neighbor_idx[0], neighbor_idx[1]]:
                    continue
                # costly regions are expensive to navigate through (costlier edges)
                # these extra costs have to be reciprocal in order for dijkstra to function
                # cost(a to b) == cost(b to a), hence the average between the node penalty values.
                # Find which neighbors are open (exclude forbidden/explored areas of the grid)
                edge_extra_costs = 0.5 * (
                    extra_costs[neighbor_idx[0], neighbor_idx[1]]
                    + extra_costs[current[0], current[1]]
                )
                new_cost = (
                    tentative[current[0], current[1]] + kEdgeLength + edge_extra_costs
                )
                if new_cost < tentative[neighbor_idx[0], neighbor_idx[1]]:
                    tentative[neighbor_idx[0], neighbor_idx[1]] = new_cost
                # Add neighbors to to_visit if not already present
                if not_in_to_visit[neighbor_idx[0], neighbor_idx[1]]:
                    to_visit.append(neighbor_idx)
                    not_in_to_visit[neighbor_idx[0], neighbor_idx[1]] = False
            # Close the current node
            open_[current[0], current[1]] = False
        return tentative

    def old_render_agents_in_lidar(self, ranges, angles, agents, lidar_ij):
        """ Takes a list of agents (shapes + position) and renders them into the occupancy grid """
        centers_i = [0]
        centers_j = [0]
        radii_ij = [np.inf]
        for agent in agents:
            if agent.type != "legs":
                raise NotImplementedError
            left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame = agent.get_legs_pose2d_in_map()
            llc_ij = self.xy_to_ij(left_leg_pose2d_in_map_frame[:2])
            rlc_ij = self.xy_to_ij(right_leg_pose2d_in_map_frame[:2])
            leg_radius_ij = agent.leg_radius / self.resolution_
            # circle centers in 'lidar' frame (frame centered at lidar pos, but not rotated,
            # as angles in array are already rotated according to sensor angle in map frame)
            centers_i.append(llc_ij[0] - lidar_ij[0])
            centers_j.append(llc_ij[1] - lidar_ij[1])
            radii_ij.append(leg_radius_ij)
            centers_i.append(rlc_ij[0] - lidar_ij[0])
            centers_j.append(rlc_ij[1] - lidar_ij[1])
            radii_ij.append(leg_radius_ij)
        # switch to polar coordinate to find intersection between each ray and agent (circles)
        angles = np.array(angles)
        ranges = np.array(ranges)
        radii_ij = np.array(radii_ij)
        centers_r_sq = np.array(centers_i)**2 + np.array(centers_j)**2
        centers_l = np.arctan2(centers_j, centers_i)
        # Circle in polar coord: r^2 - 2*r*r0*cos(phi-lambda) + r0^2 = R^2
        # Solve equation for r at angle phi in polar coordinates, of circle of center (r0, lambda)
        # and radius R. -> 2 solutions for r knowing r0, phi, lambda, R: 
        # r = r0*cos(phi-lambda) - sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # r = r0*cos(phi-lambda) + sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # solutions are real only if term inside sqrt is > 0
        first_term = np.sqrt(centers_r_sq) * np.cos(angles[:,None] - centers_l)
        sqrt_inner = centers_r_sq * np.cos(angles[:,None] - centers_l)**2 - centers_r_sq + radii_ij**2
        sqrt_inner[sqrt_inner < 0] = np.inf
        radii_solutions_a = first_term - np.sqrt(sqrt_inner)
        radii_solutions_b = first_term + np.sqrt(sqrt_inner)
        radii_solutions_a[radii_solutions_a < 0] = np.inf
        radii_solutions_b[radii_solutions_b < 0] = np.inf
        # range is the smallest range between original range, and intersection range with each agent
        all_sol = np.hstack([radii_solutions_a, radii_solutions_b])
        all_sol = all_sol * self.resolution_ # from ij coordinates back to xy
        all_sol[:,0] = ranges
        final_ranges = np.min(all_sol, axis=1)
        ranges[:] = final_ranges

    def render_agents_in_lidar(self, ranges, angles, agents, lidar_ij):
        if not self.crender_agents_in_lidar(ranges, angles.astype(np.float32), agents, lidar_ij.astype(np.float32)):
#             print("in rendering agents, object too close for efficient solution")
            self.old_render_agents_in_lidar(ranges, angles, agents, lidar_ij)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef crender_agents_in_lidar(self,
            np.ndarray[np.float32_t, ndim=1] ranges,
            np.ndarray[np.float32_t, ndim=1] angles,
            agents,
            np.ndarray[np.float32_t, ndim=1] lidar_ij,
            ):
        """ Takes a list of agents (shapes + position) and renders them into the occupancy grid
        assumes the angles are ordered from lowest to highest, spaced evenly (const increment)
        """
        cdef int n_centers = 2*len(agents)
        cdef np.float32_t[:] centers_i = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_j = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] radii_ij = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_r_sq = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_l = np.zeros((n_centers,), dtype=np.float32)
        # loop variables
        cdef CSimAgent cagent
        cdef np.float32_t[:, ::1] left_leg_pose2d_in_map_frame = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] right_leg_pose2d_in_map_frame = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] llc_ij = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] rlc_ij = np.zeros((1,3), dtype=np.float32)
        cdef int i1 = 0
        cdef int i2 = 0
        cdef np.float32_t leg_radius_ij
        for n in range(len(agents)):
            agent = agents[n]
            cagent = CSimAgent(agent.pose_2d_in_map_frame, agent.state)
            cagent.cget_legs_pose2d_in_map(left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame)
            self.cxy_to_ij(left_leg_pose2d_in_map_frame[:1,:2], llc_ij)
            self.cxy_to_ij(right_leg_pose2d_in_map_frame[:1, :2], rlc_ij)
            leg_radius_ij = cagent.leg_radius / self.resolution_
            # circle centers in 'lidar' frame (frame centered at lidar pos, but not rotated,
            # as angles in array are already rotated according to sensor angle in map frame)
            i1 = 2*n # even index, for left leg
            i2 = 2*n+1 # odd index for right leg
            centers_i[i1] = llc_ij[0, 0] - lidar_ij[0]
            centers_j[i1] = llc_ij[0, 1] - lidar_ij[1]
            radii_ij[i1] = leg_radius_ij
            centers_i[i2] = rlc_ij[0, 0] - lidar_ij[0]
            centers_j[i2] = rlc_ij[0, 1] - lidar_ij[1]
            radii_ij[i2] = leg_radius_ij
            # switch to polar coordinate to find intersection between each ray and agent (circles)
            centers_r_sq[i1] = centers_i[i1]**2 + centers_j[i1]**2
            centers_l[i1] = np.arctan2(centers_j[i1], centers_i[i1])
            centers_r_sq[i2] = centers_i[i2]**2 + centers_j[i2]**2
            centers_l[i2] = np.arctan2(centers_j[i2], centers_i[i2])
        # Circle in polar coord: r^2 - 2*r*r0*cos(phi-lambda) + r0^2 = R^2
        # Solve equation for r at angle phi in polar coordinates, of circle of center (r0, lambda)
        # and radius R. -> 2 solutions for r knowing r0, phi, lambda, R: 
        # r = r0*cos(phi-lambda) - sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # r = r0*cos(phi-lambda) + sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # solutions are real only if term inside sqrt is > 0
        for i in range(n_centers):
            # if an object is too close, this will not be efficient, tell the caller to switch to numpy
            if centers_r_sq[i] == 0 or centers_r_sq[i] < radii_ij[i]**2:
                return False
        # we can first check at what angles this holds.
        # there should be two extrema for the circle in phi, which are solutions for:
        # r0^2*cos^2(phi-lambda) - r0^2 + R^2 = 0 
        # the two solutions are:
        # phi = lambda + 2*pi*n +- arccos( +- sqrt(r0^2 - R^2) / r0 )
        # these exist only if r0 > R and r0 != 0
        cdef np.float32_t angle_min = angles[0]
        cdef np.float32_t angle_max = angles[len(angles)-1]
        cdef np.float32_t angle_inc = angles[1] - angle_min
        if angle_min >= angle_max:
            raise ValueError("angles expected to be ordered from min to max.")
        # angle_0_ref is a multiple of 2pi, the closest one smaller than angles[0]
        # assuming a scan covers less than full circle, all angles in the scan should lie 
        # between angle_0_ref and angle_0_ref + 2* 2pi (two full circles)
        cdef np.float32_t angle_0_ref = 2*np.pi * (angle_min // (2*np.pi))
        # loop variables
        cdef np.float32_t r0sq
        cdef np.float32_t r0
        cdef np.float32_t lmbda
        cdef np.float32_t R
        cdef np.float32_t phimin
        cdef np.float32_t phimax
        cdef np.float32_t phi
        cdef np.float32_t first_term
        cdef np.float32_t sqrt_inner
        cdef np.float32_t min_solution
        cdef np.float32_t possible_solution
        cdef np.float32_t possible_solution_m
        cdef int indexmin
        cdef int indexmax
        for i in range(n_centers):
            r0sq = centers_r_sq[i]
            r0 = np.sqrt(r0sq)
            lmbda = centers_l[i]
            R = radii_ij[i]
            phimin = lmbda - np.arccos( np.sqrt(r0sq - R**2) / r0 )
            phimax = lmbda + np.arccos( np.sqrt(r0sq - R**2) / r0 )
            #                    this is phi as an angle [0, 2pi]
            phimin = angle_0_ref + phimin % (np.pi * 2)
            phimax = angle_0_ref + phimax % (np.pi * 2)
            # try the second full circle if our agent is outside the scan
            if phimax < angle_min:
                phimin = phimin + np.pi * 2
                phimax = phimax + np.pi * 2
            # if still outside the scan, our agent is not visible
            if phimax < angle_min or phimin > angle_max:
                continue
            # find the index for the first visible circle point in the scan
            indexmin = int( ( max(phimin, angle_min) - angle_min ) // angle_inc )
            indexmax = int( ( min(phimax, angle_max) - angle_min ) // angle_inc )
            for idx in range(indexmin, indexmax+1):
                phi = angles[idx]
                first_term = r0 * np.cos(phi - lmbda)
                sqrt_inner = r0sq * np.cos(phi - lmbda)**2 - r0sq + R**2
                if sqrt_inner < 0:
                    # in this case that ray does not see the agent
                    continue
                min_solution = ranges[idx] # initialize with scan range
                possible_solution = first_term - np.sqrt(sqrt_inner) # in ij units
                possible_solution_m = possible_solution * self.resolution_ # in meters
                if possible_solution_m >= 0:
                    min_solution = min(min_solution, possible_solution_m)
                possible_solution = first_term + np.sqrt(sqrt_inner)
                possible_solution_m = possible_solution * self.resolution_
                if possible_solution_m >= 0:
                    min_solution = min(min_solution, possible_solution_m)
                ranges[idx] = min_solution
        return True

import pose2d

cdef class CSimAgent:
    cdef public np.float32_t[:] pose_2d_in_map_frame
    cdef public str type
    cdef public np.float32_t[:] state
    cdef public float leg_radius

    def __cinit__(self, pose, state):
        self.pose_2d_in_map_frame = pose
        self.type = "legs"
        self.state = state
        self.leg_radius = 0.03 # [m]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cget_legs_pose2d_in_map(self,
            np.float32_t[:, ::1] left_leg_pose2d_in_map_frame, 
            np.float32_t[:, ::1] right_leg_pose2d_in_map_frame):
        if self.type != "legs":
            raise NotImplementedError
        cdef np.float32_t[:] m_a_T = self.pose_2d_in_map_frame
        cdef np.float32_t leg_radius = self.leg_radius # [m]
        cdef np.float32_t leg_side_offset = 0.1 # [m]
        cdef np.float32_t leg_side_amplitude = 0.1 # [m] half amplitude
        cdef np.float32_t leg_front_amplitude = 0.3 # [m]
        # get position of each leg w.r.t agent (x is 'forward')
        # travel is a sine function relative to how fast the agent is moving in x y
        cdef np.float32_t front_travel =  leg_front_amplitude * ccos(
                self.state[0] * 2. / leg_front_amplitude # assuming dx = 2 dphi / A
                + self.state[2] # adds a little movement when rotating
                )
        cdef np.float32_t side_travel =  leg_side_amplitude * ccos(
                self.state[1] * 2. / leg_side_amplitude
                + self.state[2]
                )
        cdef np.float32_t[:, ::1] right_leg_pose2d_in_agent_frame = np.zeros((1,3), dtype=np.float32)
        right_leg_pose2d_in_agent_frame[0, 0] = front_travel
        right_leg_pose2d_in_agent_frame[0, 1] = side_travel + leg_side_offset
        right_leg_pose2d_in_agent_frame[0, 2] = 0
        cdef np.float32_t[:, ::1] left_leg_pose2d_in_agent_frame = np.zeros((1,3), dtype=np.float32) 
        right_leg_pose2d_in_agent_frame[0, 0] = -right_leg_pose2d_in_agent_frame[0, 0]
        right_leg_pose2d_in_agent_frame[0, 1] = -right_leg_pose2d_in_agent_frame[0, 1]
        right_leg_pose2d_in_agent_frame[0, 2] = -right_leg_pose2d_in_agent_frame[0, 2]
        capply_tf_to_pose(
                left_leg_pose2d_in_agent_frame, m_a_T, left_leg_pose2d_in_map_frame)
        capply_tf_to_pose(
                right_leg_pose2d_in_agent_frame, m_a_T, right_leg_pose2d_in_map_frame)

    def get_legs_pose2d_in_map(self):
        m_a_T = self.pose_2d_in_map_frame
        if self.type == "legs":
            leg_radius = self.leg_radius # [m]
            leg_side_offset = 0.1 # [m]
            leg_side_amplitude = 0.1 # [m] half amplitude
            leg_front_amplitude = 0.3 # [m]
            # get position of each leg w.r.t agent (x is 'forward')
            # travel is a sine function relative to how fast the agent is moving in x y
            front_travel =  leg_front_amplitude * np.cos(
                    self.state[0] * 2. / leg_front_amplitude # assuming dx = 2 dphi / A
                    + self.state[2] # adds a little movement when rotating
                    )
            side_travel =  leg_side_amplitude * np.cos(
                    self.state[1] * 2. / leg_side_amplitude
                    + self.state[2]
                    )
            right_leg_pose2d_in_agent_frame = np.array([
                front_travel,
                side_travel + leg_side_offset,
                0])
            left_leg_pose2d_in_agent_frame =  -right_leg_pose2d_in_agent_frame
            left_leg_pose2d_in_map_frame = pose2d.apply_tf_to_pose(
                    left_leg_pose2d_in_agent_frame, m_a_T)
            right_leg_pose2d_in_map_frame = pose2d.apply_tf_to_pose(
                    right_leg_pose2d_in_agent_frame, m_a_T)
            return left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame
        else:
            raise NotImplementedError

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef capply_tf_to_pose(np.float32_t[:, ::1] pose, np.float32_t[:] pose2d,
    np.float32_t[:, ::1] result):
    cdef np.float32_t th = pose2d[2]
    result[0, 0] = ccos(th) * pose[0, 0] - csin(th) * pose[0, 1] + pose2d[0]
    result[0, 1] = csin(th) * pose[0, 0] + ccos(th) * pose[0, 1] + pose2d[1]
    result[0, 2] = pose[0, 2] + th

