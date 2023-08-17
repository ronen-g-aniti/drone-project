import pdb
import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D

def voxel_builder(filename, res):
	"""
	Build a voxel map representation of the configuration space using the obstacle file.
	"""

	obstacles = np.genfromtxt(filename, delimiter=',', skip_header=2)
	#{(x, y, z, hx, hy, hz)}

	north_offset = np.min(obstacles[:, 0] - obstacles[:, 3], axis=0)
	east_offset = np.min(obstacles[:, 1] - obstacles[:, 4], axis=0)
	north_max = np.max(obstacles[:, 0] + obstacles[:, 3], axis=0)
	east_max = np.max(obstacles[:, 1] + obstacles[:, 4], axis=0)
	down_offset = 0
	down_max = np.max(obstacles[:, 2] + obstacles[:, 5], axis=0)
	north_size = int(north_max - north_offset) // res
	east_size = int(east_max - east_offset) // res
	down_size = int(down_max - down_offset) // res

	voxel_map_shape = (north_size, east_size, down_size)

	voxel_map = np.zeros(voxel_map_shape, dtype=bool)

	for obstacle in obstacles:
		center = obstacle[:3]
		halfsize = obstacle[3:]
		min_coords = np.maximum(((center - halfsize) - np.array([north_offset, east_offset, down_offset])) // res, 0).astype(int)
		max_coords = np.minimum(((center + halfsize) - np.array([north_offset, east_offset, down_offset])) // res, voxel_map_shape).astype(int)
		voxel_map[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]] = True

	return voxel_map

res = 5
voxel_map = voxel_builder('colliders.csv', res)

a = 0 // res
b = 150 // res
c = 0 // res
d = 150 // res
e = 0 // res
f = 150 // res

voxel_map = voxel_map[a:b, c:d, e:f]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Rotate the plot by 90 degrees around the Z-axis
ax.view_init(elev=70, azim=0)

# Due to matplotlib's different coordinate system, we need to transpose the voxel array
ax.voxels(voxel_map, facecolors='blue', edgecolor='k', alpha=0.5)

ax.set_xlabel("Meters North of North Offset")
ax.set_ylabel("Meters East of East Offset")
ax.set_zlabel("Altitude (m)")
ax.set_xlim(a, b)
ax.set_ylim(c, d)
ax.set_zlim(e, f)

plt.title("Voxel Map Representation of the Configuration Space")
plt.show()
pdb.set_trace()


