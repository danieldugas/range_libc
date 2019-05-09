import range_libc
import numpy as np
import itertools, time

# -----------------
from map2d import Map2D, SimAgent, gridshow
from timeit import default_timer as timer
from pepper_2d_simulator import compiled_raytrace
from matplotlib import pyplot as plt
map_folder = "."
map_name = "office_full"
map2d = Map2D(map_folder, map_name)
print("Map '{}' loaded.".format(map_name))

testMap = range_libc.PyOMap(np.ascontiguousarray(map2d.occupancy.T >= map2d.thresh_occupied))
rm = range_libc.PyRayMarching(testMap, map2d.occupancy.shape[0])
grid = np.ascontiguousarray(np.zeros_like(map2d.occupancy, dtype=np.float32))
rm.get_dist(grid)

esdf = grid * map2d.resolution
esdf[map2d.occupancy > map2d.thresh_free] *= -1.
lidar_pos = np.array([1, 0, 0])
kLidarMinAngle = -2.35619449615
kLidarMaxAngle = 2.35572338104
kLidarAngleIncrement = 0.00581718236208
angles = np.arange(kLidarMinAngle, kLidarMaxAngle, kLidarAngleIncrement) + lidar_pos[2]
ranges = np.zeros_like(angles)
occupancy = map2d.occupancy
lidar_pos_ij = map2d.xy_to_ij(lidar_pos[:2])

compiled_raytrace(angles, lidar_pos_ij, map2d.occupancy, esdf,
        map2d.thresh_occupied, map2d.resolution, map2d.origin, ranges)
tic =timer()
compiled_raytrace(angles, lidar_pos_ij, map2d.occupancy, esdf,
        map2d.thresh_occupied, map2d.resolution, map2d.origin, ranges)
toc = timer()
print("raytrace: {}s".format(toc-tic))
testMap = range_libc.PyOMap(map2d.occupancy.T >= map2d.thresh_occupied)
rm = range_libc.PyRayMarching(testMap, map2d.occupancy.shape[0])
cddt = range_libc.PyCDDTCast(testMap, 1000, 108)
bl = range_libc.PyBresenhamsLine(testMap, 1000)
rmgpu = range_libc.PyRayMarchingGPU(testMap, 1024)
vals = np.zeros((len(angles), 3))
angles = np.arange(kLidarMinAngle, kLidarMaxAngle, kLidarAngleIncrement) + lidar_pos[2]
rm_ranges = np.zeros_like(ranges, dtype=np.float32)
rmgpu_ranges = np.zeros_like(ranges, dtype=np.float32)
vals[:, 0] = lidar_pos_ij[0]
vals[:, 1] = lidar_pos_ij[1]
vals[:, 2] = angles
vals = vals.astype(np.float32)
test_states = [None]*len(angles)
for i in range(len(angles)):
    test_states[i] = (vals[i,0],
                      vals[i,1],
                      vals[i,2])
obj = rm
tic =timer()
# obj.calc_range_many(vals, rm_ranges)
rm_ranges = np.array(list(map(lambda x: obj.calc_range(*x), test_states)))
toc = timer()
print("rm raytrace: {}s".format(toc-tic))
obj = rm
tic =timer()
vals = np.zeros((len(angles), 3))
rmnp_ranges = np.zeros_like(ranges, dtype=np.float32)
vals[:, 0] = lidar_pos_ij[0]
vals[:, 1] = lidar_pos_ij[1]
vals[:, 2] = angles
vals = vals.astype(np.float32)
rm.calc_range_many(vals, rmnp_ranges)
# rm_ranges = np.array(list(map(lambda x: obj.calc_range(*x), test_states)))
toc = timer()
print("rm GPU raytrace: {}s".format(toc-tic))
obj = rmgpu
tic =timer()
vals = np.zeros((len(angles), 3))
rmgpu_ranges = np.zeros_like(ranges, dtype=np.float32)
vals[:, 0] = lidar_pos_ij[0]
vals[:, 1] = lidar_pos_ij[1]
vals[:, 2] = angles
vals = vals.astype(np.float32)
obj.calc_range_many(vals, rmgpu_ranges)
# rm_ranges = np.array(list(map(lambda x: obj.calc_range(*x), test_states)))
toc = timer()
print("rm GPU raytrace: {}s".format(toc-tic))
other_agents = []
for i in range(16):
    # populate agents list
    agent = SimAgent()
    xyth = np.random.random((3,)).astype(np.float32)
    xyth[1] *= (testMap.width() - 2.0)
    xyth[0] *= (testMap.height() - 2.0)
    agent.pose_2d_in_map_frame = np.array([xyth[0], xyth[1], xyth[2]])
    agent.type = "legs"
    agent.state = np.array([0, 0, 0])
    other_agents.append(agent)
tic = timer()
rd_ranges = map2d.render_agents_in_lidar(ranges, angles, other_agents, lidar_pos_ij)
toc = timer()
print("render agents: {}s".format(toc-tic))
from matplotlib import pyplot as plt
plt.plot(ranges)
plt.plot(rm_ranges * map2d.resolution)
plt.plot(rmnp_ranges * map2d.resolution)
plt.plot(rmgpu_ranges * map2d.resolution)
plt.show()
