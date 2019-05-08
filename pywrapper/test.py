import range_libc
import numpy as np
import itertools, time
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt

####################################################################################################
#
#                                              WARNING
#  
#
#                    This file uses range_libc in it's native coordinate space.
#                      Use this method at your own peril since the coordinate
#                      conversions are nontrivial from ROS's coordinate space.
#                       Ignore this warning if you intend to use range_libc's 
#                                   left handed coordinate space
#
#
####################################################################################################

# print( range_libc.USE_CACHED_TRIG )
# print( range_libc.USE_CACHED_TRIG )
# print( range_libc.USE_ALTERNATE_MOD )
# print( range_libc.USE_CACHED_CONSTANTS )
# print( range_libc.USE_FAST_ROUND )
# print( range_libc.NO_INLINE )
# print( range_libc.USE_LRU_CACHE )
# print( range_libc.LRU_CACHE_SIZE )

# testMap = range_libc.PyOMap("../maps/basement_hallways_5cm.png",1)
testMap = range_libc.PyOMap("../maps/synthetic.map.png",1)
# testMap = range_libc.PyOMap("/home/racecar/racecar-ws/src/TA_examples/lab5/maps/basement.png",1)

if testMap.error():
	exit()
# testMap.save("./test.png")

num_vals = 100000

# vals = np.zeros((3,num_vals), dtype=np.float32)
# vals[0,:] = testMap.width()/2.0
# vals[1,:] = testMap.height()/2.0
# vals[2,:] = np.linspace(0,2.0*np.pi, num=num_vals)

# def make_scan(x,y,theta,n_ranges):
# 	MAX_SCAN_ANGLE = (0.75 * np.pi)
# 	bl = range_libc.PyBresenhamsLine(testMap, 300)
# 	# bl = range_libc.PyRayMarching(testMap, 500)
# 	queries = np.zeros((n_ranges,3),dtype=np.float32)
# 	ranges = np.zeros(n_ranges,dtype=np.float32)
# 	queries[:,0] = x
# 	queries[:,1] = y
# 	queries[:,2] = theta + np.linspace(-MAX_SCAN_ANGLE, MAX_SCAN_ANGLE, n_ranges)
# 	bl.calc_range_many(queries,ranges)
# 	bl.saveTrace("./test.png")

# make_scan(510,520,np.pi/2.0,61)

print( "Init: bl" )
bl = range_libc.PyBresenhamsLine(testMap, 500)
print( "Init: rm" )
# rm = range_libc.PyRayMarching(testMap, 500)
print( "Init: cddt" )
cddt = range_libc.PyCDDTCast(testMap, 500, 108)
cddt.prune()
print( "Init: glt" )
# glt = range_libc.PyGiantLUTCast(testMap, 500, 108)
# this is for testing the amount of raw functional call overhead, does not compute ranges
# null = range_libc.PyNull(testMap, 500, 108)

for x in xrange(0):
# for x in xrange(10):
	vals = np.random.random((3,num_vals)).astype(np.float32)
	vals[0,:] *= (testMap.width() - 2.0)
	vals[1,:] *= (testMap.height() - 2.0)
	vals[0,:] += 1.0
	vals[1,:] += 1.0
	vals[2,:] *= np.pi * 2.0
	ranges = np.zeros(num_vals, dtype=np.float32)

	test_states = [None]*num_vals
	for i in xrange(num_vals):
		test_states[i] = (vals[0,i], vals[1,i], vals[2,i])

	def bench(obj,name):
		print( "Running:", name )
		start = time.clock()
		obj.calc_range_many(vals, ranges)
		end = time.clock()
		dur_np = end - start
		print( ",,,"+name+" np: finished computing", ranges.shape[0], "ranges in", dur_np, "sec" )
		start = time.clock()
		ranges_slow = map(lambda x: obj.calc_range(*x), test_states)
		end = time.clock()
		dur = end - start

		diff = np.linalg.norm(ranges - np.array(ranges_slow))
		if diff > 0.001:
			print( ",,,"+"Numpy result different from slow result, investigation possibly required. norm:", diff )
		# print( "DIFF:", diff )

		print( ",,,"+name+": finished computing", ranges.shape[0], "ranges in", dur, "sec" )
		print( ",,,"+"Numpy speedup:", dur/dur_np )

	bench(bl, "bl")
	bench(rm, "rm")
	bench(cddt, "cddt")
	bench(glt, "glt")

	# ranges_bl = np.zeros(num_vals, dtype=np.float32)
	# ranges_rm = np.zeros(num_vals, dtype=np.float32)
	# ranges_cddt = np.zeros(num_vals, dtype=np.float32)
	# ranges_glt = np.zeros(num_vals, dtype=np.float32)

	# bl.calc_range_np(vals, ranges_bl)
	# rm.calc_range_np(vals, ranges_rm)
	# cddt.calc_range_np(vals, ranges_cddt)
	# glt.calc_range_np(vals, ranges_glt)

	# diff = ranges_rm - ranges_cddt
	# norm = np.linalg.norm(diff)
	# avg = np.mean(diff)
	# min_v = np.min(diff)
	# max_v = np.max(diff)
	# median = np.median(diff)
	# print( avg, min_v, max_v, median )

	# plt.hist(diff, bins=1000, normed=1, facecolor='green', alpha=0.75)
	# plt.show()

	# this is for testing the amount of raw functional call overhead, does not compute ranges
	# bench(null, "null")
print( "DONE" )

# -----------------
import numpy as np
from map2d import Map2D, SimAgent
from timeit import default_timer as timer
from pepper_2d_simulator import compiled_raytrace
map_folder = "."
map_name = "empty"
map2d = Map2D(map_folder, map_name)
esdf = map2d.as_sdf()
print("Map '{}' loaded.".format(map_name))

lidar_pos = np.array([0, 0, 0])
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
testMap = range_libc.PyOMap(map2d.occupancy.T >= map2d.thresh_free)
rm = range_libc.PyRayMarching(testMap, map2d.occupancy.shape[0])
cddt = range_libc.PyCDDTCast(testMap, 1000, 108)
bl = range_libc.PyBresenhamsLine(testMap, 1000)
vals = np.zeros((3, len(angles)))
angles = np.arange(kLidarMinAngle, kLidarMaxAngle, kLidarAngleIncrement) + lidar_pos[2]
rm_ranges = np.zeros_like(ranges, dtype=np.float32)
vals[1, :] = lidar_pos_ij[0]
vals[0, :] = lidar_pos_ij[1]
vals[2, :] = angles
vals = vals.astype(np.float32)
test_states = [None]*len(angles)
for i in xrange(len(angles)):
    test_states[i] = (vals[0,i], vals[1,i], vals[2,i])
obj = bl
tic =timer()
# obj.calc_range_many(vals, rm_ranges)
rm_ranges = np.array(map(lambda x: obj.calc_range(*x), test_states))
toc = timer()
print("rm raytrace: {}s".format(toc-tic))
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
plt.show()
