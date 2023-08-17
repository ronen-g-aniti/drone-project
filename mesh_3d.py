from scipy.spatial import KDTree
import heapq
import csv
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D

def construct_obstacles(filename):
	"""Retruns a 2d array with obstacle x, y, z, dx, dy, dz values along with a KDTree referencing ground positions of obstacles x, y"""
	return np.genfromtxt(filename, delimiter=',', skip_header=2)

def discretize_environment(obstacles, resolution):
	"""Returns a 2d array of coarse states x, y, z that don't collide with obstacles"""
	xmin, ymin, zmin = np.min(obstacles[:,:3] - obstacles[:,3:], axis=0)
	xmax, ymax, zmax = np.max(obstacles[:,:3] + obstacles[:,3:], axis=0)
	x = np.arange(xmin, xmax, resolution)
	y = np.arange(ymin, ymax, resolution)
	z = np.arange(10.0, zmax, resolution)

	grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
	
	mask = np.zeros(grid.shape[0], dtype=bool)
	for obstacle in obstacles:
			mask |= is_inside_obstacle(grid, obstacle)

	return grid[~mask]

def is_inside_obstacle(states, obstacle):
	center, halfsizes = obstacle[:3], obstacle[3:]
	return np.all(np.abs(states - center) <= halfsizes, axis=1)

def coarse_path(start, goal, states, states_kd):
	"""Returns the shortest path between start and goal through coarse states"""
	start_idx = states_kd.query([start], k=1)[1][0]
	goal_idx = states_kd.query([goal], k=1)[1][0]

	frontier = [(0, start_idx)]
	came_from = {start_idx: None}
	cost_so_far = {start_idx: 0}
	goal_reached = False

	while frontier:
		current_priority, current_idx = heapq.heappop(frontier)
		if current_idx == goal_idx:
			goal_reached = True
			print("Rerouting successful")
			break

		current_point = states[current_idx]
		_, neighbors = states_kd.query([current_point], k=7)  # adjust k based on how many neighbors you want
		neighbors = neighbors[0]

		for neighbor_idx in neighbors:
			if neighbor_idx != current_idx:  # Avoid self as neighbor due to floating point inaccuracies
				if np.linalg.norm(states[current_idx] - states[neighbor_idx]) < 25.0 + 1.0: # Avoid navigating to further away neighbors that have potential for collision.
					new_cost = cost_so_far[current_idx] + np.linalg.norm(states[current_idx] - states[neighbor_idx])
					if neighbor_idx not in cost_so_far or new_cost < cost_so_far[neighbor_idx]:
						cost_so_far[neighbor_idx] = new_cost
						priority = new_cost + np.linalg.norm(states[goal_idx] - states[neighbor_idx])
						heapq.heappush(frontier, (priority, neighbor_idx))
						came_from[neighbor_idx] = current_idx

	if not goal_reached:
		return []

	# Reconstruct path
	current_idx = goal_idx
	path = []
	while current_idx is not None:
		path.append(states[current_idx])
		current_idx = came_from[current_idx]
	path.reverse()
	path.append(goal)

	return path

def collides(obstacles, start_point, end_point):
	"""
	Checks if the bounding box of the line segment connecting the start point and the end point intersects the bounding 
	box of any obstacle from the obstacle space. This collision checking routine is known as the "Axis-aligned Bounding Box
	Algorithm" and serves the purpose in this code as a simple way of checking whether or not a sampled point should be added 
	to the tree. 
	"""
	obstacle_centers = obstacles[:, :3]
	obstacle_halfsizes = obstacles[:, 3:]
	# Defines the maxima and minima x, y, z positions of the bounding box around the line segment
	minima = np.minimum(start_point, end_point)
	maxima = np.maximum(start_point, end_point)

	# Checks whether or not the bounding box around each obstacle collides with the bounding box around the line segment. 
	collision_mask = np.all((obstacle_centers - obstacle_halfsizes) <= maxima, axis=1) & np.all((obstacle_centers + obstacle_halfsizes) >= minima, axis=1)

	# Returns True if a collision with any obstacle is detected and false otherwise
	return np.any(collision_mask)


obstacles = construct_obstacles("colliders.csv")
obstacles_kd = KDTree(obstacles[:,:2])
resolution = 25.
states = discretize_environment(obstacles, resolution)
states_kd = KDTree(states)
start = np.array([100, 100, 5])
goal = np.array([800, 150, 5])
path = coarse_path(start, goal, states, states_kd)


# Fit a polynomial between waypoints
n = 3 # Degree of the polynomial
x = np.array([point[0] for point in path])
y = np.array([point[1] for point in path])
z = np.array([point[2] for point in path])
coeffs_xy = np.polyfit(x, y, n)
coeffs_xz = np.polyfit(x, z, n)
x_values = np.linspace(x[0], x[-1], 100)
y_values = np.polyval(coeffs_xy, x_values)
z_values = np.polyval(coeffs_xz, x_values)

# Evaluate the polynomial at multiple points along the path
path_points = np.linspace(0, 1, num=100)
path_x = x_values
path_y = np.polyval(coeffs_xy, x_values)
path_z = np.polyval(coeffs_xz, x_values)
path_points_xyz = np.column_stack((path_x, path_y, path_z))

# Check if any straight line segments collide with obstacles
collision_detected = False
for i in range(len(path_points_xyz) - 1):
	point1 = path_points_xyz[i]
	point2 = path_points_xyz[i + 1]
	if collides(obstacles, point1, point2):
		collision_detected = True
		break

if collision_detected:
	print("Straight line segments between waypoints collide with obstacles!")
	# Handle collision or take appropriate action
else:
	print("Straight line segments between waypoints are collision-free.")

# Visualize the polynomial path and obstacles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', label='Waypoints')
ax.plot(x_values, y_values, z_values, c='red', label="Polynomial curve")
#ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=(0, 0, 0), alpha=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

pdb.set_trace()


from scipy.interpolate import interp1d

# transpose waypoints so that each row is a waypoint
waypoints = np.array(path).T

# Define the parameter along the path
t = np.arange(waypoints.shape[1])
pdb.set_trace()
# Create interpolation functions for each dimension
x_interpolation = interp1d(t, waypoints[0, :], kind='cubic')
y_interpolation = interp1d(t, waypoints[1, :], kind='cubic')
z_interpolation = interp1d(t, waypoints[2, :], kind='cubic')

# Create new range for t and find interpolated values
t_new = np.linspace(t.min(), t.max(), 50)  # or any other range within the original t
x_new = x_interpolation(t_new)
y_new = y_interpolation(t_new)
z_new = z_interpolation(t_new)

# Visualizing the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(waypoints[0, :], waypoints[1, :], waypoints[2, :], color='red')  # original points
ax.plot(x_new, y_new, z_new, color='blue')  # interpolated points
plt.show()


# The problem with this is that it's easy to generate curves that collide with obstacles.
# A viable solution to this problem is to draw an integral curve through a potential field instead of using polynomials.
#


plt.set_trace()