import numpy as np
import csv
import pdb
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


# Perameters that works reasonably well for the AABB collision check + PRM are this:
# connectivity=15, samples=3000, z=1
def sample(n=1000, z=1):
	return np.random.uniform(low=(x_min, y_min, z),high=(x_max, y_max, z), size=(n, 3))

def is_inside_obstacle(sample_point):
	return np.any(np.all((sample_point <= obstacles_max) & (sample_point >= obstacles_min), axis=1)) 

def collides_2(start_point, end_point):
	vector = end_point - start_point
	direction = vector / np.linalg.norm(vector)
	step = 0.25
	current = start_point
	while np.linalg.norm(current - end_point) > step:
		current = current + step * direction
		if is_inside_obstacle(current):
			return True
	return False


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
	collision_mask = np.all(obstacles_min <= maxima, axis=1) & np.all(obstacles_max >= minima, axis=1)

	# Returns True if a collision with any obstacle is detected and false otherwise
	return np.any(collision_mask)

if __name__ == "__main__":
	obstacles = np.genfromtxt('colliders.csv', delimiter=',', skip_header=2)
	obstacles_min = obstacles[:, :3] - obstacles[:, 3:]
	obstacles_max = obstacles[:, :3] + obstacles[:, 3:]

	x_min = np.min(obstacles[:, 0] - obstacles[:, 3], axis=0)
	x_max = np.max(obstacles[:, 0] + obstacles[:, 3], axis=0)
	y_min = np.min(obstacles[:, 1] - obstacles[:, 4], axis=0)
	y_max = np.max(obstacles[:, 1] + obstacles[:, 4], axis=0)
	z_min = 0
	z_max = np.max(obstacles[:, 2] + obstacles[:, 5], axis=0)

	samples = sample(n=2500, z=1)
	is_inside = np.apply_along_axis(is_inside_obstacle, 1, samples)
	filtered_samples = samples[~is_inside]

	filtered_samples_kd = KDTree(filtered_samples)

	connectivity = 8
	# Connect each filtered sample to up to 10 of its nearest neighbors that don't cause straight line collisions
	connections = {i: set() for i in range(len(filtered_samples))}
	for i, sample in enumerate(filtered_samples):
		indices = filtered_samples_kd.query(sample, connectivity + 1)[1]
		indices = indices[1:] # Ignore the first index since it's the sample itself
		for index in indices:
			if len(connections[i]) >= connectivity:
				break
			neighbor = filtered_samples[index]
			if not collides_2(sample, neighbor):
				if index not in connections[i]:
					connections[i].add(index)
				if i not in connections[index]:
					connections[index].add(i)

	print("Map generated")
	fig, ax = plt.subplots()
	plt.title("Representing Free Space as a Probabilistic Roadmap")
	plt.xlabel("North (m)")
	plt.ylabel("East (m)")
	for i in range(len(obstacles)):
		center = obstacles[i, :3]
		halfsize = obstacles[i, 3:]
		if center[2] + halfsize[2] >= 1:
			rect = plt.Rectangle((center[0] - halfsize[0], center[1] - halfsize[1]), 
								 2*halfsize[0], 2*halfsize[1], 
								 color='Gray', alpha=0.5)
			ax.add_patch(rect)
	plt.scatter([filtered_samples[:, 0]], [filtered_samples[:, 1]], s=1, color=(0,0,1))
	# plot lines for each connected points
	for key, values in connections.items():
	    for value in values:
	        ax.plot(*zip(filtered_samples[key], filtered_samples[value]), color=(0,0,1), linewidth=0.1)

	plt.show()
	pdb.set_trace()