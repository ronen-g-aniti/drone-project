import numpy as np
from scipy.spatial import KDTree

class RRT:
	"""This is an implementation of the RRT algorithm. The algorithm is biased such that every 10th sample is the goal state."""
	def __init__(self, start, goal, obstacles):
		"""Initializes the tree"""
		self.start = start
		self.goal = goal
		self.explored = np.array([start])
		self.edges = {}		
		self.obstacle_centers = obstacles[:, :3]
		self.obstacle_halfsizes = obstacles[:, 3:]
		self.xbounds, self.ybounds, self.zbounds = self.bounds()

	def bounds(self):
		"""Determines the bounds of the obstacle map in the NEA frame"""
		obstacle_centers = self.obstacle_centers
		obstacle_halfsizes = self.obstacle_halfsizes 
		xmin = np.min(obstacle_centers[:,0] - obstacle_halfsizes[:,0])
		xmax = np.max(obstacle_centers[:,0] + obstacle_halfsizes[:,0])
		ymin = np.min(obstacle_centers[:,1] - obstacle_halfsizes[:,1])
		ymax = np.max(obstacle_centers[:,1] + obstacle_halfsizes[:,1])
		zmin = np.min(obstacle_centers[:,2] - obstacle_halfsizes[:,2])
		zmax = np.max(obstacle_centers[:,2] + obstacle_halfsizes[:,2])
		xbounds = (xmin, xmax)
		ybounds = (ymin, ymax)
		zbounds = (zmin, zmax)
		return xbounds, ybounds, zbounds

	def run(self, step=15.0, max_iters=100000):
		"""Runs the main algorithm. This is a form of the rapidly-exploring random tree algorithm"""

		goal_found = False

		# Set the current state to be the start state
		current = self.start

		# Loop until the goal is reached
		iteration = 0
		while not self.goal_reached(current) and iteration < max_iters:
			iteration += 1
			s_sample = self.sample()
			if iteration % 2 == 0: 
				s_sample = self.goal

			# Returns the nearest tree node `s_nearest` to `s_sample`
			s_nearest_index = np.argmin(np.linalg.norm(s_sample[:3] - self.explored[:, :3], axis=1))
			s_nearest = self.explored[s_nearest_index]

			# Compute s_new, a state that is `step` distance away from s_nearest in the direction `u_hat` of s_sample
			u_hat = (s_sample - s_nearest) / np.linalg.norm(s_sample - s_nearest)
			s_new = s_nearest + u_hat * step
			
			# Add a branch of maximum length possible
			#for substep in np.linspace(step, 1.0, num=10):
			#	s_new = s_nearest + u_hat * substep
			#	if not self.collides(s_nearest, s_new):
			#		break

			# Add s_new to the tree only if the segment connecting it to s_nearest doesn't collide with any obstacles
			if not self.collides(s_nearest, s_new):
				self.explored = np.vstack((self.explored, s_new))
				# Add a key-value pair to `edges` to represent the connection between s_new and s_nearest
				s_new_index = len(self.explored) - 1
				self.edges[s_new_index] = s_nearest_index


			# Set the current node to be s_new
			current = s_new

			# Break the loop if the current node is close enough to the goal state
			if self.goal_reached(current):
				print("Goal is reached")
				goal_found = True
				break
				
		# Reconstruct and shorten the path if the goal is reached, then return it.
		if goal_found:
			path = self.reconstruct()
			# Appends the actual goal state to path if doing so doesn't register a collision 
			if not self.collides(current, self.goal):
				path = np.vstack((path, self.goal))
			path = self.shorten(path)
		
		# Call the animate method at the end of the run method
		#self.animate(path)
		
		return path

	def sample(self):
		"""Samples a random state from within the bounds of the obstacle space"""
		x = np.random.uniform(low=self.xbounds[0], high=self.xbounds[1], size=1)[0]
		y = np.random.uniform(low=self.ybounds[0], high=self.ybounds[1], size=1)[0]
		# Restrict z to goal altitude
		z = self.goal[2]
		#z = np.random.uniform(low=self.zbounds[0], high=self.zbounds[1], size=1)[0]
		return np.array([x, y, z])

	def collides(self, start_point, end_point):
		"""
		Checks if the bounding box of the line segment connecting the start point and the end point intersects the bounding 
		box of any obstacle from the obstacle space. This collision checking routine is known as the "Axis-aligned Bounding Box
		Algorithm" and serves the purpose in this code as a simple way of checking whether or not a sampled point should be added 
		to the tree. 
		"""
		obstacle_centers = self.obstacle_centers
		obstacle_halfsizes = self.obstacle_halfsizes
		# Defines the maxima and minima x, y, z positions of the bounding box around the line segment
		minima = np.minimum(start_point, end_point)
		maxima = np.maximum(start_point, end_point)

		# Checks whether or not the bounding box around each obstacle collides with the bounding box around the line segment. 
		safety = 5
		collision_mask = np.all((obstacle_centers - obstacle_halfsizes - safety) <= maxima, axis=1) & np.all((obstacle_centers + obstacle_halfsizes + safety) >= minima, axis=1)

		# Returns True if a collision with any obstacle is detected and false otherwise
		return np.any(collision_mask)

	def goal_reached(self, current, tolerance=15.0):
		"""Checks whether or not the goal state has been reached within the tolerance specified"""
		current = np.array([current[0], current[1], current[2]])
		if np.linalg.norm(current - self.goal) < tolerance:
			return True
		return False


	def reconstruct(self):
		"""Reconstructs and returns the path from start state to goal state"""
		goal_index = len(self.explored) - 1
		start_index = 0
		path = [goal_index]
		current_index = goal_index
		while current_index != start_index:
			came_from_index = self.edges[current_index]
			path.append(came_from_index)
			current_index = came_from_index
		path = path[::-1]
		path = self.explored[path]
		return path

	def shorten(self, path):
		# Initialize an empty list for the shortened path
		shortened_path = [self.start]

		# Set the current state equal to the start state
		current_state = path[0]

		# Check for collision between the start state and all other states in the path
		collision_results = np.array([self.collides(path[0][:3], state[:3]) for state in path])

		# Get the maximum index that is False (indicating no collision)
		last_false_index = np.where(collision_results == False)[0][-1]

		# Append the path state corresponding to the max_false_index to the shortened_path list
		shortened_path.append(path[last_false_index])

		# Repeat steps 3-5 until reaching the end state
		while not np.array_equal(current_state, path[-1]):
			# Update the current state to be the last state added to the shortened path
			current_state = shortened_path[-1]

			# Check for collision between the current state and all other states in the path
			collision_results = np.array([self.collides(current_state[:3], state[:3]) for state in path])

			# Get the maximum index that is False (indicating no collision)
			last_false_index = np.where(collision_results == False)[0][-1]

			# Append the path state corresponding to the max_false_index to the shortened_path list
			if not np.array_equal(path[last_false_index], current_state):
				shortened_path.append(path[last_false_index])
		
		shortened_path = np.array(shortened_path)

		return shortened_path

import pdb
import matplotlib.pyplot as plt

class PotentialField:
	def __init__(self, obstacles, waypoints):
		self.obstacles = obstacles
		self.start = waypoints[0]
		self.goal = waypoints[-1]
		self.obstacle_ground_positions = obstacles[:, :2]
		self.obstacle_ground_positions_kd = KDTree(self.obstacle_ground_positions)
		self.waypoints = waypoints


	def attractive_vector(self):
		pass
	def repulsive_vector(self, current):
		current_x = current[0]
		current_y = current[1]
		halfwidth = 50
		xmax = current_x + halfwidth
		xmin = current_x - halfwidth
		ymax = current_y + halfwidth
		ymin = current_y - halfwidth
		obstacles = self.obstacles
		filtered_obstacles = obstacles[(obstacles[:, 0] >= xmin) & (obstacles[:, 0] <= xmax) & (obstacles[:, 1] >= ymin) & (obstacles[:, 1] <= ymax)]
		vectors_from_obstacles = current[:2]-filtered_obstacles[:,:2]
		distances = np.linalg.norm(vectors_from_obstacles, axis=1)
		scale_factors = 1/(distances-5*np.sqrt(5))**2
		direction_vectors_from_obstacles = vectors_from_obstacles/((distances).T[:,np.newaxis])
		scale_factors_reshaped = scale_factors[:, np.newaxis]
		scaled_vectors_from_obstacles = direction_vectors_from_obstacles * scale_factors_reshaped
		vector_sums = np.sum(scaled_vectors_from_obstacles, axis=0)
		repulsive_direction = vector_sums/np.linalg.norm(vector_sums)
		repulsive_direction = np.append(repulsive_direction, 0)
		return repulsive_direction

	def original_repulsive_vector_computation(self, current):
		pass

	def command_direction(self):
		pass

	def integrate(self, step=0.1):
		waypoints = self.waypoints
		start = self.start
		goal = self.goal
		obstacles = self.obstacles
		obstacles_kd = self.obstacle_ground_positions_kd
		obstacle_ground_positions = self.obstacle_ground_positions
		collision_distance = 5*np.sqrt(2)
		krmax = 2 #2
		kamax = 4#4
		current = start
		current_index = 0
		max_index = len(waypoints) - 1
		path = []

		current = waypoints[current_index]
		iteration_num = 0
		while current_index < max_index and iteration_num < 10000:

			# Get the current ground position and current height
			current_ground_pos = current[:2]
			current_height = current[2]

			# Append current point to path
			path.append(current)
			"""
			# Return the (ox, oy, ch) array
			# compute the vector from that point to the current position
			# compute the direction of that vector
			distances, indices = obstacles_kd.query([current_ground_pos], k=20)
			for index in indices[0]:
				if obstacles[index][2] >= current_height:
					distance_to_obstacle = np.linalg.norm(obstacle_ground_positions[index] - current_ground_pos)
					vector_from_obstacle = np.append(np.array([obstacle_ground_positions[index][:2]]), current_height)
					direction_vector_from_obstacle = vector_from_obstacle / np.linalg.norm(vector_from_obstacle)
					break
			"""
			# get the next waypoint in the sequence
			next_waypoint = waypoints[current_index + 1]

			# compute the vector from current to the next waypoint
			vector_to_next_waypoint = next_waypoint - current

			# compute the direction of that vector
			distance_to_next_waypoint = np.linalg.norm(vector_to_next_waypoint)
			direction_to_next_waypoint = vector_to_next_waypoint / distance_to_next_waypoint

			# take a linear combination of those two vectors to be the result vector
			waypoint_distance = np.linalg.norm(waypoints[current_index] - next_waypoint)
			command_vector = kamax * direction_to_next_waypoint + krmax * self.repulsive_vector(current)

			# take the direction of the result vector
			command_direction = command_vector / np.linalg.norm(command_vector)

			# set current to be current + a step in that direction
			current = current + command_direction * step

			# if the distance between current and the next waypoint is <= 2, increment the current_index
			if current_index < max_index-1:
				if np.linalg.norm(current - next_waypoint) < 5:
					current_index += 1
			else:
				if np.linalg.norm(current - next_waypoint) < 0.1:
					current_index += 1



			# if after incrementing the current_index is greater than the maximum index, then break out of this loop
			iteration_num += 1
		safe_path = np.array(path)

		return safe_path

		

start = np.array([100, 100, 0.1])
#goal = np.array([540, 280, 0.1])
goal = np.array([442, -260, 0.1])
obstacles = np.genfromtxt('colliders.csv', delimiter=',', skip_header=2)
rrt = RRT(start, goal, obstacles)
path = rrt.run()
shortened_path = rrt.shorten(path)
potential_field = PotentialField(obstacles, path)
safe_path = potential_field.integrate()
skip_count = 100

fig, ax = plt.subplots()
plt.title("Hybrid Motion Planning")
plt.xlabel("North (m)")
plt.ylabel("East (m)")
for i in range(len(obstacles)):
	center = obstacles[i, :3]
	halfsize = obstacles[i, 3:]
	if center[2] + halfsize[2] >= 5:
		rect = plt.Rectangle((center[0] - halfsize[0], center[1] - halfsize[1]), 
							 2*halfsize[0], 2*halfsize[1], 
							 color='Gray', alpha=0.5)
		ax.add_patch(rect)
plt.scatter([shortened_path[:, 0]], [shortened_path[:, 1]], s=5, color=(0,0,1), label='RRT path')
#plt.plot(safe_path[:,0], safe_path[:,1], c=(1,0,0))
plt.scatter(safe_path[:,0][::skip_count], safe_path[:,1][::skip_count],color=(1,0,0), s=0.25, label='Potential field waypoints')
plt.legend()
#skip_count=1
#plt.plot(safe_path[:,0][::skip_count], safe_path[:,1][::skip_count],color=(0,0,0), linewidth=0.5)
plt.show()
pdb.set_trace()



# Add code here to generate a polynomial interpolation through these `safe_path` waypoints

# Extract waypoints along the integral curve generated by the potential field algorithm
key_points = safe_path[::skip_count*2]
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

wp = key_points
# Separate x, y and z coordinates
x = wp[:, 0]
y = wp[:, 1]
z = wp[:, 2]

# Parameter t along the path
t = np.arange(wp.shape[0])

# Create cubic splines for x, y and z
cs_x = CubicSpline(t, x, bc_type='natural')
cs_y = CubicSpline(t, y, bc_type='natural')
cs_z = CubicSpline(t, z, bc_type='natural')

# Evaluate splines at finer resolution
t_fine = np.linspace(t.min(), t.max(), 500)
x_fine = cs_x(t_fine)
y_fine = cs_y(t_fine)
z_fine = cs_z(t_fine)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_fine, y_fine, z_fine, label='Cubic Spline Interpolation')
ax.scatter(x, y, z, color='red', label='Waypoints')
ax.scatter(shortened_path[:,0], shortened_path[:,1], shortened_path[:,2], color='lime', label='RRT Waypoints')
ax.legend()
plt.show()