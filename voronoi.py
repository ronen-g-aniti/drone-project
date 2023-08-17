import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree
import pdb

def inbounds(p1, p2):
	x1 = p1[0]
	x2 = p2[0]
	y1 = p1[1]
	y2 = p2[1]
	if not (xmin<x1<xmax and xmin<x2<xmax and ymin<y1<ymax and ymin<y2<ymax):
		return False
	return True

def collides(start_point, end_point):
	"""
	Checks if the bounding box of the line segment connecting the start point and the end point intersects the bounding 
	box of any obstacle from the obstacle space. This collision checking routine is known as the "Axis-aligned Bounding Box
	Algorithm" and serves the purpose in this code as a simple way of checking whether or not a sampled point should be added 
	to the tree. 
	"""
	# Defines the maxima and minima x, y, z positions of the bounding box around the line segment
	minima = np.minimum(start_point, end_point)
	maxima = np.maximum(start_point, end_point)

	# Checks whether or not the bounding box around each obstacle collides with the bounding box around the line segment. 
	collision_mask = np.all(obstacles_min <= maxima+0.01, axis=1) & np.all(obstacles_max >= minima-0.01, axis=1)

	# Returns True if a collision with any obstacle is detected and false otherwise
	return np.any(collision_mask)

obstacles = np.genfromtxt('colliders.csv', delimiter=',', skip_header=2)
obstacles = obstacles[obstacles[:,2] + obstacles[:, 5] >= 5]

voronoi = Voronoi(obstacles[:, :2])

obstacles_min = obstacles[:, :2] - obstacles[:, 3:5]
obstacles_max = obstacles[:, :2] + obstacles[:, 3:5]

xmin = np.min(obstacles[:, 0] - obstacles[:, 3], axis=0)
ymin = np.min(obstacles[:, 1] - obstacles[:, 4], axis=0)
xmax = np.max(obstacles[:, 0] + obstacles[:, 3], axis=0)
ymax = np.max(obstacles[:, 1] + obstacles[:, 4], axis=0)

edges = voronoi.ridge_vertices
points = voronoi.vertices
valid_edges = []
for edge in edges:
	index1 = edge[0]
	index2 = edge[1]
	if not collides(points[index1], points[index2]):
		if inbounds(points[index1], points[index2]):

			valid_edges.append([index1, index2])


fig, ax = plt.subplots()
plt.title("Using a Voronoi Diagram to Represent the Free Space")
plt.xlabel("North (m)")
plt.ylabel("East (m)")
for valid_edge in valid_edges:
	index1 = valid_edge[0]
	index2 = valid_edge[1]
	point1 = points[index1]
	point2 = points[index2]
	x1 = point1[0]
	x2 = point2[0]
	y1 = point1[1]
	y2 = point2[1]
	ax.plot([x1, x2],[y1, y2], linewidth=0.5, color=(0,0,1))

for i in range(len(obstacles)):
	center = obstacles[i, :3]
	halfsize = obstacles[i, 3:]
	if center[2] + halfsize[2] >= 5:
		rect = plt.Rectangle((center[0] - halfsize[0], center[1] - halfsize[1]), 
							 2*halfsize[0], 2*halfsize[1], 
							 color='Gray', alpha=0.5)
		ax.add_patch(rect)

plt.show()
pdb.set_trace()