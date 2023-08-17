from enum import Enum
import pdb
import matplotlib.pyplot as plt
from queue import PriorityQueue
import numpy as np
import csv
from typing import Tuple, List
from skimage.morphology import medial_axis
from skimage.util import invert
from skimage import morphology
from skimage import feature
from scipy.ndimage import distance_transform_edt


class Action(Enum):
	"""
	Defines the action class that is used inside of the A* search routine.
	"""
	NORTH = (1, 0, 1)
	NORTHEAST = (1, 1, np.sqrt(2))
	NORTHWEST = (1, -1, np.sqrt(2))
	SOUTH = (-1, 0, 1)
	SOUTHEAST = (-1, 1, np.sqrt(2))
	SOUTHWEST = (-1, -1, np.sqrt(2))
	EAST = (0, 1, 1)
	WEST = (0, -1, 1) 

	@property
	def delta(self):
		return self.value[:2]

	@property
	def cost(self):
		return self.value[2]

def astar(start: Tuple[float], goal: Tuple[float], grid: np.ndarray, north_offset: float, east_offset: float) -> Tuple[List[Tuple[int]], float]:
	"""
	Search an occupancy grid from a start location in the NE frame to a goal location in the NE frame. Return a
	path of waypoints in the NE frame that has been culled. 
	"""
	# Convert the start and goal points from the NE frame to the grid frame
	start_node = (int(start[0] - north_offset), int(start[1] - east_offset))
	goal_node = (int(goal[0] - north_offset), int(goal[1] - east_offset))

	# Initialize data structures that this A* implementation will use
	open_set = PriorityQueue()
	f_score = None
	g_score = None
	g_score_tentative = None
	h_score = None
	branch = {}
	visited = set()
	goal_found = False

	# For visualization purposes, record these points of information:
	open_set_history = []
	visited_set_history = []
	current_node_history = []



	# Place the start node inside of the priority queue
	open_set.put((0, start_node))

	while open_set:

		item = open_set.get()
		current_node = item[1]
		if grid[current_node[0], current_node[1]] == False: #This condition is necessary for visualization purposes
			visited.add(current_node)

		if current_node == goal_node:
			print("Goal is found")
			goal_found = True
			break

		if current_node != start_node:
			g_score = branch[current_node][1]
		else:
			g_score = 0

		moves = neighbors(grid, current_node)

		for move in moves:
			new_node = move[:2]
			if new_node not in visited and grid[new_node[0], new_node[1]] == False: # The second condition is necessary for visualization but not for the algorithm to run
				action_cost = move[2]
				g_score_tentative = g_score + action_cost
				h_score = ((new_node[:2][0] - goal_node[0])**2 + (new_node[:2][1] - goal_node[1])**2)**(1/2)
				f_score = g_score_tentative + h_score
				if branch.get(new_node) is not None:
					if g_score_tentative < branch[new_node][1]:
						branch[new_node] = (f_score, g_score_tentative, h_score, current_node)
						open_set.put((f_score, new_node))
				else:
					branch[new_node] = (f_score, g_score_tentative, h_score, current_node)
					open_set.put((f_score, new_node))
		
		#open_set_history.append([element[1] for element in list(open_set.queue)])
		#visited_set_history.append(list(visited))
		#current_node_history.append(current_node)

	if goal_found:
		path = []
		current_node = goal_node
		cost = branch[current_node][1]
		while current_node != start_node:
			path.append(current_node)
			current_node = branch[current_node][3]
		path.append(start_node)
		path = path[::-1]

		# For visualization purposes
		#np.save("open_set_history.npy", np.array(open_set_history))
		#np.save("visited_set_history.npy", np.array(visited_set_history))
		#np.save("current_node_history.npy", np.array(current_node_history))


	else:
		path = []
		cost = 0



	return path, cost, open_set_history, visited_set_history, current_node_history


def neighbors(grid: np.ndarray, index: Tuple[int]) -> List[Tuple[int, int, float]]:
	valid_neighbors = []
	actions = list(Action)
	for action in actions: 
		delta_north = action.delta[0] 
		delta_east = action.delta[1]
		new_north = index[0] + delta_north
		new_east = index[1] + delta_east
		if new_north >= grid.shape[0] or new_north < 0:
			continue
		if new_east >= grid.shape[1] or new_east < 0:
			continue
		if grid[new_north, new_east] == False:
			valid_neighbors.append((new_north, new_east, action.cost))
	return valid_neighbors


def gridmaker(filename: str, target_altitude: float) -> Tuple[np.ndarray, float, float]:
	"""
	This function reads the obstacle file that has been provided by Udacity and returns the file
	contents as a NumPy array. This function also returns the north and east grid offset.
	"""

	obstacles = np.genfromtxt(filename, delimiter=',', skip_header=2)
	filtered_obstacles = obstacles[obstacles[:, 2] + obstacles[:, 5] > target_altitude]
	north_offset = np.min(filtered_obstacles[:, 0] - filtered_obstacles[:, 3], axis=0)
	east_offset = np.min(filtered_obstacles[:, 1] - filtered_obstacles[:, 4], axis=0)
	north_max = np.max(filtered_obstacles[:, 0] + filtered_obstacles[:, 3], axis=0)
	east_max = np.max(filtered_obstacles[:, 1] + filtered_obstacles[:, 4], axis=0)
	north_size = int(north_max - north_offset)
	east_size = int(east_max - east_offset)
	grid = np.empty((north_size, east_size), dtype=bool)

	for i in range(filtered_obstacles.shape[0]):
		# Calculate the indices of the four bounding box coordinates for each obstacle
		north_indices = (int(filtered_obstacles[i, 0] - filtered_obstacles[i, 3] - north_offset),
						 int(filtered_obstacles[i, 0] + filtered_obstacles[i, 3] - north_offset))
		east_indices = (int(filtered_obstacles[i, 1] - filtered_obstacles[i, 4] - east_offset),
						 int(filtered_obstacles[i, 1] + filtered_obstacles[i, 4] - east_offset))

		# Set the region bounded inside of each obstacle equal to True in the occupancy grid
		grid[north_indices[0]:north_indices[1], east_indices[0]:east_indices[1]] = True

	return grid, north_offset, east_offset



def shorten(path, tolerance=0.01):
	"""
	Returns a path that is shortened according the a scheme that takes into consideration how collinear successive nodes are.
	"""
	shortened_path = [path[0]]

	for i in range(len(path)-2):
		node0 = np.array(path[i])
		node1 = np.array(path[i+1])
		node2 = np.array(path[i+2])

		vector1 = node1 - node0
		vector2 = node2 - node1

		dot_product = np.dot(vector1, vector1)
		mag_vector1 = np.linalg.norm(vector1)
		mag_vector2 = np.linalg.norm(vector2)
		mag_product = mag_vector1 * mag_vector2

		angle = np.arccos(dot_product / mag_product)

		if abs(angle) > tolerance:
			shortened_path.append(node1)


	shortened_path.append(path[-1])

	return shortened_path



def visualize(grid: np.ndarray, path=None):
	plt.figure()
	#plt.title("The Path Constructed Using A* Search\nthrough the Occupancy Grid")
	#plt.title("The Occupancy Grid Discretization of the Environment")
	plt.title("The Shortened Path through the Occupancy Grid")
	plt.xlabel("Meters North of Offset North Position")
	plt.ylabel("Meters East of Offset East Position")
	plt.xlim(0, 920) # Grid frame
	plt.ylim(0, 920) # Grid frame
	plt.imshow(grid.transpose(), cmap="binary")
	if path:
		north_indices = [node[0] for node in path]
		east_indices = [node[1] for node in path]
		plt.scatter(north_indices, east_indices, s=5, color=(0,0,1))
	plt.scatter(path[0][0], path[0][1], s=5, color=(0,1,0), label="Start")
	plt.scatter(path[-1][0], path[-1][1], s=5, color=(1,0,0), label="Goal")
	plt.legend()

	plt.show()


plt.rcParams['font.family'] = 'Verdana'

grid, north_offset, east_offset = gridmaker('colliders.csv', 5)
start = (0, 0) # NE frame
goal = (400,-300) # NE frame
path, cost,open_set_history, visited_set_history, current_node_history = astar(start, goal, grid, north_offset, east_offset)
#visualize(grid, path)
shortened_path = shorten(path)
#visualize(grid, shortened_path)
print(len(path))
print(len(shortened_path))

# Medial axis
skeleton = medial_axis(invert(grid))
fig, ax = plt.subplots()
plt.title("The Medial Axis Transform")
plt.xlabel("Meters North of Offset North Position")
plt.ylabel("Meters East of Offset East Position")
colors = [(0,0,0,0),(0,1,1,1)]
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors)
ax.imshow(np.flip(grid.transpose(),0), cmap="gray", alpha=1)
ax.imshow(np.flip(skeleton.transpose(),0), cmap=cmap)
plt.show()

# Using the distance transform function to add padding around obstacles.
fig, ax = plt.subplots()
distance = distance_transform_edt(np.flip(~grid.transpose(),0))
filtered_distance = distance > 5
#ax.imshow(np.flip(grid.transpose(),0), cmap="gray", alpha=0.2)
ax.imshow(filtered_distance, interpolation='nearest', cmap='gray')


plt.title("The Distance Transform")
plt.xlabel("Meters North of Offset North Position")
plt.ylabel("Meters East of Offset East Position")
plt.show()


