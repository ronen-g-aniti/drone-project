import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from queue import PriorityQueue, Queue
from typing import Tuple, List
import pdb
from matplotlib.animation import FuncAnimation

def bfs(start, goal, grid):
	open_set = Queue()
	open_set.put(start)
	branch = {}
	branch[start] = None
	goal_found = False
	closed_set = set()

	# For visualization purposes, record these points of information:
	open_set_history = []
	visited_set_history = []
	current_node_history = []

	while not open_set.empty():
		processing = open_set.get()
		closed_set.add(processing)
		if processing == goal:
			print("The goal is found")
			goal_found = True
			break
		moves = neighbors(grid, processing)
		for move in moves:
			new = move[:2]
			if new in closed_set or new in open_set.queue:
				continue
			open_set.put(new)
			branch[new] = processing

		# For visualization
		open_set_history.append([element for element in list(open_set.queue)])
		visited_set_history.append(list(closed_set))
		current_node_history.append(processing)

	if goal_found:
		path = [goal]
		cost = None
		previous = branch[goal]
		while previous is not None:
			path.append(previous)
			previous = branch[previous]
		path = path[::-1]
		return path, cost, open_set_history, visited_set_history, current_node_history

def euclidean_distance(start, end):
	return ((start[0] - end[0])**2 + (start[1] - end[1])**2) **0.5

def astar_revised_2(start, goal, grid, h):
	open_set = PriorityQueue()
	open_set.put((0, start))
	branch = {} # [new] : tentative_g_score, previous
	branch[start] = (0, None)
	goal_found = False
	closed_set = set()

	# For visualization purposes, record these points of information:
	open_set_history = []
	visited_set_history = []
	current_node_history = []

	while not open_set.empty():
		item = open_set.get()
		processing = item[1]
		closed_set.add(processing)
		if processing == goal:
			print("The goal is found")
			goal_found = True
			break
		g_score = branch[processing][0]
		moves = neighbors(grid, processing)
		for move in moves:
			new = move[:2]
			action_cost = move[2]
			if new in closed_set: # If the new node has already been visited, skip adding it to the open set and branching dictionary
				continue
			if branch.get(new) is None or g_score + action_cost < branch[new][0]:
				branch[new] = (g_score + action_cost, processing)
				open_set.put((g_score + action_cost + h(new, goal), new))

		# For visualization
		open_set_history.append([element[1] for element in list(open_set.queue)])
		visited_set_history.append(list(closed_set))
		current_node_history.append(processing)

	if goal_found:
		path = [goal]
		cost = branch[goal][0]
		previous = branch[goal][1]
		while previous is not None:
			path.append(previous)
			previous = branch[previous][1]
		path = path[::-1]
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


class Action(Enum):
	"""
	Defines the action class that is used inside of the A* search routine.
	"""
	NORTH = (1, 0, 1)
	EAST = (0, 1, 1)
	WEST = (0, -1, 1) 
	SOUTH = (-1, 0, 1)
	NORTHEAST = (1, 1, np.sqrt(2))
	NORTHWEST = (1, -1, np.sqrt(2))
	SOUTHEAST = (-1, 1, np.sqrt(2))
	SOUTHWEST = (-1, -1, np.sqrt(2))


	@property
	def delta(self):
		return self.value[:2]

	@property
	def cost(self):
		return self.value[2]


def visualize(occupancy_grid, path):
	fig, ax = plt.subplots()
	plt.imshow(~occupancy_grid.transpose(), origin='lower', cmap='gray')
	plt.plot([p[0] for p in path],[p[1] for p in path], 'ro-')
	ax.scatter([path[0][0], path[-1][0]], [path[0][1], path[-1][1]], s=10)
	ax.set_xlim(0,200)
	ax.set_ylim(0,200)
	plt.show()

def visualize_animated(occupancy_grid, open_set_history, visited_set_history, current_node_history, path):
	fig, ax = plt.subplots()
	def update(frame):
		ax.clear()
		ax.imshow(~occupancy_grid.transpose(), origin='lower', cmap='gray')
		open_set = open_set_history[frame]
		visited_set = visited_set_history[frame]
		current_node = current_node_history[frame]
		if open_set:
			ax.scatter([node[0] for node in open_set], [node[1] for node in open_set], color='blue', label='Open set')
		if visited_set:
			ax.scatter([node[0] for node in visited_set], [node[1] for node in visited_set], color='red', label="Visited set")
		#if current_node:
			#ax.scatter([current_node[0]], [current_node[1]], color='lime', label='Current node')
		ax.text(path[0][0]-15, path[0][1]-10, "Start")
		ax.text(path[-1][0]-15, path[-1][1]+6, "Goal")
		ax.scatter([path[0][0], path[-1][0]],[path[0][1], path[-1][1]], color=(0.5,0,0.5))
		ax.scatter([path[0][0], path[-1][0]],[path[0][1], path[-1][1]], color=(0.5,0,0.5))
		if frame==len(open_set_history)-1:
			ax.plot([node[0] for node in path],[node[1] for node in path], color=(1,1,0))
		ax.set_xlim(75,200)
		ax.set_ylim(75,200)
		ax.legend()
	frames_to_display = list(range(0, len(open_set_history), 100))
	ani = FuncAnimation(fig, update, frames=frames_to_display, repeat=False)
	
	plt.show()
	return ani

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

def shorten_path_integer_method(path):
	shortened_path = [path[0]]
	for i in range(len(path) - 2):
		p0 = path[i]
		p1 = path[i+1]
		p2 = path[i+2]
		
		x0, y0, z0 = p0[0], p0[1], 1
		x1, y1, z1 = p1[0], p1[1], 1
		x2, y2, z2 = p2[0], p2[1], 1

		# Find the determinant of the matrix constructed by the horizontal stacking of these coordinate triplets.
		# Use integer arithmetic only
		area = x0 * (y1 - y2) - x1 * (y0 - y2) + x2 * (y0 - y1)

		if area != 0:
			shortened_path.append(p1)
	return shortened_path

def shorten_path_angle_method(path):
	shortened_path = [path[0]]
	tolerance = 0.001 # radian
	for i in range(len(path) - 2):
		v0 = path[i + 1] - path[i]
		v1 = path[i + 2] - path[i + 1]
		angle = np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
		if angle > tolerance:
			shortened_path.append(path[i+1])
	return shortened_path





# Add the ability to change the resolution of the grid
# Add the ability to set a safety margin around obstacles
# Implement a waypoint shortening algorithm based on integer collinearity formula.

# Define the dimensions of the occupancy grid
m = 60 # number of rows
n = 60 # number of columns

# Define the number of obstacles
num_obstacles = 1500

# Create an empty occupancy grid, then fill it with the correct number of obstacles
occupancy_grid = np.full((m, n), False)
occupied_indices = np.random.choice(m*n, num_obstacles, replace=False)
np.put(occupancy_grid, occupied_indices, True)

# Select a random start and goal position from the unoccupied cells
false_indices = np.argwhere(occupancy_grid == False)
random_row_index_1 = np.random.choice(len(false_indices))
random_row_index_2 = np.random.choice(len(false_indices))
start_position = tuple(false_indices[random_row_index_1])
goal_position = tuple(false_indices[random_row_index_2])

# Use a random occupancy grid
# path, cost, open_set_history, visited_set_history, current_node_history = astar(start_position, goal_position, occupancy_grid)

# Use data from colliders for grid
occupancy_grid, _, _ = gridmaker('colliders.csv', 5)
path, cost, open_set_history, visited_set_history, current_node_history = astar_revised_2((100,100),(150,150), occupancy_grid, euclidean_distance)

# Use A* to generate a path from the start state to the goal state
#visualize(occupancy_grid, path)
#visualize_animated(occupancy_grid, open_set_history, visited_set_history, current_node_history, path)

# Use BFS to make a comparison
#path, cost, open_set_history, visited_set_history, current_node_history = bfs((100,100),(150,150), occupancy_grid)
#visualize(occupancy_grid, path)
#visualize_animated(occupancy_grid, open_set_history, visited_set_history, current_node_history, path)

# Compare Numpy cross product shortening algorithm to the integer determinant method.
visualize(occupancy_grid, shorten_path_integer_method(path))
visualize(occupancy_grid, shorten_path_angle_method(np.array(path)))