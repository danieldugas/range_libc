# distutils: language=c++

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
import cython

import os
from yaml import load
from matplotlib.pyplot import imread

cdef class CMap2D:
    cdef public np.float32_t[:,::1] occupancy_
    cdef int occupancy_shape0
    cdef int occupancy_shape1
    cdef float resolution
    cdef float thresh_occupied
    cdef float thresh_free
    cdef float HUGE_
    cdef public np.float32_t[:] origin
    def __cinit__(self, folder=None, name=None):
        self.occupancy_ = np.ones((100, 100), dtype=np.float32) * 0.5
        self.occupancy_shape0 = 100
        self.occupancy_shape1 = 100
        self.resolution = 0.01
        self.origin = np.array([0., 0.], dtype=np.float32)
        self.thresh_occupied = 0.9
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
        self.resolution = mapparams["resolution"]  # [meters] side of 1 grid square
        self.origin = np.array(mapparams["origin"][:2]).astype(np.float32)  # [meters] x y coordinates of point at i = j = 0
        if mapparams["origin"][2] != 0:
            raise ValueError(
                "Map files with a rotated frame (origin.theta != 0) are not"
                " supported. Setting the value to 0 in the MAP_NAME.yaml file is one way to"
                " resolve this."
            )
        self.thresh_occupied = mapparams["occupied_thresh"]
        self.thresh_free = mapparams["free_thresh"]
        self.HUGE_ = 100 * self.occupancy_shape0 * self.occupancy_shape1 # bigger than any possible distance in the map
        if self.resolution == 0:
            raise ValueError("resolution can not be 0")

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.nonecheck(False)
#     @cython.cdivision(True)
#    def xy_to_ij(self, np.ndarray[np.float32_t, ndim=2] xy, bool clip_if_outside=True):
    def xy_to_ij(self, np.float32_t[:,::1] xy, bool clip_if_outside=True):
        if xy.shape[1] != 2:
            raise IndexError("xy should be of shape (n, 2)")
        ij = np.zeros([xy.shape[0], xy.shape[1]], dtype=np.intc)
        for k in range(xy.shape[0]):
            ij[k, 0] = <int>((xy[k, 0] - self.origin[0]) / self.resolution)
            ij[k, 1] = <int>((xy[k, 1] - self.origin[1]) / self.resolution)
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

    def xx(self, np.float32_t[:,::1] xy):
        return xy
