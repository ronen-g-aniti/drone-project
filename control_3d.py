import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.spatial import KDTree
import heapq

class Drone:
	def __init__(self):

		# Constants and properties
		self.i_x = 0.01
		self.i_y = 0.01
		self.i_z = 0.01
		self.m = 1.0
		self.g = 9.81
		self.k_f = 1e-5
		self.k_m = 1e-6 # reactive momement constant
		self.l = 0.5 / 2 * np.sqrt(2)  
		self.dt = 0.01
		
		# State variables and derivatives
		self.x = 0
		self.y = 0
		self.z = 0 
		self.x_dot = 0
		self.y_dot = 0
		self.z_dot = 0
		self.phi = 0
		self.theta = 0
		self.psi = 0
		self.phi_dot = 0
		self.theta_dot = 0
		self.psi_dot = 0
		self.p = 0
		self.q = 0
		self.r = 0
		self.p_dot = 0
		self.q_dot = 0
		self.r_dot = 0
		self.x_ddot = 0
		self.y_ddot = 0
		self.z_ddot = 0 

		# Motor speeds
		self.omega_1 = 0.
		self.omega_2 = 0.
		self.omega_3 = 0.
		self.omega_4 = 0.
		
		# Motor configuration
		#
		#   #1      #2
		#	  -	  -
		#		-
		#     -   -
		#	#4      #3
		#

		# The x axis points ^
		# The y axis points >
		# The z axis points down into the page x

		# self.l is the distance from the drone center to each rotor

		# 1 = clockwise
		# 2 = counter
		# 3 = clockwise
		# 4 = counter

	# Motor speeds
	@property
	def omega(self):
		return [self.omega_1, self.omega_2, self.omega_3, self.omega_4]

	@property
	def rotation_matrix(self):
		R_x = np.array([
			[1, 0, 0],
			[0, np.cos(self.phi), -np.sin(self.phi)],
			[0, np.sin(self.phi), np.cos(self.phi)]
			])
		R_y = np.array([
			[np.cos(self.theta), 0, np.sin(self.theta)],
			[0, 1, 0],
			[-np.sin(self.theta), 0, np.cos(self.theta)]
			])

		R_z = np.array([
			[np.cos(self.psi), -np.sin(self.psi), 0],
			[np.sin(self.psi), np.cos(self.psi), 0],
			[0, 0, 1]
			])

		return R_z @ R_y @ R_x

	@property
	def state(self):
		return np.array([self.x, self.y, self.z, self.x_dot, self.y_dot, self.z_dot, self.phi, self.theta, self.psi, self.p, self.q, self.r])
	
	@property
	def collective_force(self):
		return self.k_f * (self.omega[0]**2 + self.omega[1]**2 + self.omega[2]**2 + self.omega[3]**2)
	
	def set_motor_speeds(self, F, T_x, T_y, T_z):
		pdb.set_trace()		
		# Given commanded body-frame force and torques, set the new motor speeds
		self.p_dot = float((T_x - (self.i_z - self.i_y) * self.q * self.r) / self.i_x)
		self.q_dot = float((T_y - (self.i_x - self.i_z) * self.p * self.r) / self.i_y)
		self.r_dot = float((T_z - (self.i_y - self.i_x) * self.p * self.q) / self.i_z)

		# The matrix that defines how the motor speeds are related to the commanded force and torques 
		transformation_matrix = np.array([
			[-self.k_f, -self.k_f, -self.k_f, -self.k_f],
			[self.k_f*self.l, -self.k_f*self.l, -self.k_f*self.l, self.k_f*self.l],
			[self.k_f*self.l, self.k_f*self.l, -self.k_f*self.l, -self.k_f*self.l],
			[-self.k_m, self.k_m, -self.k_m, self.k_m]
			])
	
		commands = np.array([F, T_x, T_y, T_z])

		# Uses matrix algebra to convert those 4 inputs into 4 motor speeds (4 equations and 4 unknowns)
		omega = np.sqrt(np.linalg.solve(transformation_matrix, commands))
		self.omega_1 = omega[0]
		self.omega_2 = -omega[1]
		self.omega_3 = omega[2]
		self.omega_4 = -omega[3]
		print(self.omega_1, self.omega_2, self.omega_3, self.omega_4)

		# Convert the body-frame collective linear force into a x,y,z component force in the inertial frame
		linear_forces_inertial_frame = self.rotation_matrix @ np.array([0,0,F]) # F should already be negative 
		linear_accelerations_inertial_frame = 1 / self.m * linear_forces_inertial_frame
		self.x_ddot, self.y_ddot, self.z_ddot = np.array([0, 0, self.g]) + linear_accelerations_inertial_frame

		# Update linear velocity
		self.x_dot += self.x_ddot * self.dt
		self.y_dot += self.y_ddot * self.dt
		self.z_dot += self.z_ddot * self.dt

		# Update linear position
		self.x += self.x_dot * self.dt
		self.y += self.y_dot * self.dt
		self.z += self.z_dot * self.dt

		# Update angular velocities in the body frame
		self.p += self.p_dot * self.dt
		self.q += self.q_dot * self.dt
		self.r += self.r_dot * self.dt

		# Convert angular velocities in the body frame to euler angle rates in the inertial frame
		# Transformation matrix from Udacity course
		conversion_matrix = np.array([
			[1, np.sin(self.phi)*np.tan(self.theta), np.cos(self.phi)*np.tan(self.theta)],
			[0, np.cos(self.phi), -np.sin(self.phi)],
			[0, np.sin(self.phi)/np.cos(self.theta), np.cos(self.phi)/np.cos(self.theta)]
			])
		
		self.phi_dot, self.theta_dot, self.psi_dot = conversion_matrix @ np.array([self.p, self.q, self.r])
		
		# Update euler angles
		self.phi += self.phi_dot * self.dt
		self.theta += self.theta_dot * self.dt
		self.psi += self.psi_dot * self.dt


class Controller:
	def __init__(self, drone):
		self.g = 9.81
		self.drone = drone

	def subcontroller_1(self, zt, zt_dot):
		kp = 1.0
		kd = 1.0
		z_ddot_ff = 9.81
		F = kp * (zt - self.drone.z) + kd * (zt_dot - self.drone.z_dot) + z_ddot_ff
		return -1.0 * F


	def subcontroller_2(self, xt, xt_dot, yt, yt_dot):
		# Produces R_13c and R_23c based on the PD equation given the input trajectory and measured state variables
		kpx = 2.0
		kdx = 2.0
		kpy = 2.0
		kdy = 2.0
		R_13c = kpx * (xt - self.drone.x) + kdx * (xt_dot - self.drone.x_dot)
		R_23c = kpy * (yt - self.drone.y) + kdy * (yt_dot - self.drone.y_dot)
		return R_13c, R_23c

	def subcontroller_3(self, R_13c, R_23c):
		# produces a p_c and a q_c from a P equation
		kpp = 0.01
		kpq = 0.01
		R_33a = self.drone.rotation_matrix[2,2]
		R_21a = self.drone.rotation_matrix[1,0]
		R_11a = self.drone.rotation_matrix[0,0]
		R_22a = self.drone.rotation_matrix[1,1]
		R_12a = self.drone.rotation_matrix[0,1]
		R_13a = self.drone.rotation_matrix[0,2]
		R_23a = self.drone.rotation_matrix[1,2]

		# This equation is from the Udacity course material
		pq_vector = 1 / R_33a * np.array([[R_21a, -R_11a],[R_22a, -R_12a]]) @ np.array([[kpp*(R_13c-R_13a)],[kpq*(R_23c-R_23a)]])
		p_c, q_c = pq_vector[0][0], pq_vector[1][0]
		return p_c, q_c

	def subcontroller_4(self, psi_c):
		kpr = 0.1
		r_c = kpr * (psi_c - self.drone.psi)
		return r_c


	def subcontroller_5(self, p_c, q_c, r_c):
		kpp = 20.0
		kpq = 20.0
		kpr = 20.0
		T_x = kpp * (p_c - self.drone.p)
		T_y = kpq * (q_c - self.drone.q)
		T_z = kpr * (r_c - self.drone.r)
		return T_x, T_y, T_z

############################################
############################################
# Generate waypoints #######################
############################################
############################################
pdb.set_trace()
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
		halfwidth = 100
		xmax = current_x + halfwidth
		xmin = current_x - halfwidth
		ymax = current_y + halfwidth
		ymin = current_y - halfwidth
		obstacles = self.obstacles
		filtered_obstacles = obstacles[(obstacles[:, 0] >= xmin) & (obstacles[:, 0] <= xmax) & (obstacles[:, 1] >= ymin) & (obstacles[:, 1] <= ymax)]
		vectors_from_obstacles = current[:2]-filtered_obstacles[:,:2]
		distances = np.linalg.norm(vectors_from_obstacles, axis=1)
		scale_factors = 1/(distances-5*np.sqrt(2))**2
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
		krmax = 1 #2
		kamax = 4 #4
		current = start
		current_index = 0
		max_index = len(waypoints) - 1
		path = []

		current = waypoints[current_index]
		iteration_num = 0
		while current_index < max_index and iteration_num < 20000:
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
				if np.linalg.norm(current - next_waypoint) < 1:
					current_index += 1

			# if after incrementing the current_index is greater than the maximum index, then break out of this loop
			iteration_num += 1
		safe_path = np.array(path)

		return safe_path

#####################################################
#####################################################
#####################################################
# Generate a trajectory:
# 1. Generate an obstacle environment in 3D
#####################################################
#####################################################
#####################################################

hx=5.0
hy=5.0
hz=50.0
xmin = 0
xmax = 100
ymin = 0
ymax = 100
zmin = 0
zmax = 100
num_obs = 5
x_values = np.random.uniform(xmin, xmax, size=num_obs)
y_values = np.random.uniform(ymin, ymax, size=num_obs)
obstacles = np.column_stack((x_values, y_values, np.full(num_obs, hz), np.full(num_obs, hx), np.full(num_obs, hy), np.full(num_obs, hz)))

#####################################################
#####################################################
#####################################################
# 2. Run RRT algorithm with potential field to find waypoints
#####################################################
#####################################################
#####################################################

start = np.array([xmin,ymin,zmin])
goal = np.array([xmax+10,ymax+10,zmax-10])
rrt = RRT(start, goal, obstacles)
path = rrt.run()
shortened_path = rrt.shorten(path)
potential_field = PotentialField(obstacles, path)
safe_path = potential_field.integrate()
skip_count = 100
safe_path = safe_path[::skip_count]
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax.plot(safe_path[:,0],safe_path[:,1],safe_path[:,2], color='blue')
ax.scatter(safe_path[:,0],safe_path[:,1],safe_path[:,2], color='green', s=10)
ax.scatter(x_values, y_values,hz)

#####################################################
#####################################################
#####################################################
# Plot the obstacles as boxes
#####################################################
#####################################################
#####################################################
for obstacle in obstacles:
	x_center, y_center, z_center, x_hw, y_hw, z_hw = obstacle
	x_corners = [x_center + x_hw, x_center - x_hw]
	y_corners = [y_center + y_hw, y_center - y_hw]
	z_corners = [z_center + z_hw, z_center - z_hw]

	# Define the vertices that compose each of the 6 faces of the box
	vertices = [
		[(x, y, zc) for x in x_corners for y in y_corners] for zc in z_corners
	] + [
		[(xc, y, z) for xc in x_corners for z in z_corners] for y in y_corners
	] + [
		[(x, yc, z) for x in x_corners for z in z_corners] for yc in y_corners
	]

	# Plot the faces
	for verts in vertices:
		ax.add_collection3d(Poly3DCollection([verts], color='r', alpha=0.5))

plt.show()

#####################################################
#####################################################
#####################################################
# Code to find a trajectory with a  b low snap
#####################################################
#####################################################
#####################################################
x = safe_path[:,0]
y = safe_path[:,1]
z = safe_path[:,2]
points = np.column_stack((x,y,z))
differences = np.diff(points, axis=0)
squared_differences = differences**2
distances = np.sqrt(np.sum(squared_differences, axis=1))
cumulative_distances = np.cumsum(distances)
cost = cumulative_distances[-1]

valid_average_speeds = [1., 2., 4., 6., 8., 10., 12.]
valid_average_speeds = valid_average_speeds[::-1]

def minimum_snap_trajectory(x, y, z, speed):
	desired_positions = (x,y,z)
	position_list = []
	snap_list = []
	coeffs_list = []
	cost = 0 
	for axis_points in desired_positions:
		x = axis_points
		waypoint_count = len(x)
		segment_count = waypoint_count - 1
		number_of_unknowns = 8 * segment_count
		average_speed = speed
		waypoint_start_times = [0]
		#seconds_between_waypoints = 5
		for i in range(len(x)-1):
			waypoint_start_times.append(waypoint_start_times[-1] + np.linalg.norm(points[i+1]-points[i]) / average_speed)
			#waypoint_start_times.append(abs(x[i+1] - x[i]) / average_speed + waypoint_start_times[-1])
			#waypoint_start_times.append(waypoint_start_times[-1] + seconds_between_waypoints) 

		t = waypoint_start_times
		# The vector equation: M * c = b, where c is a 8*(waypoint_count-1) element vector of unknown coefficients
		# Assume the drone is at rest position-snap = 0 at the start and at the end.

		M = np.zeros((8*(waypoint_count-1), 8*(waypoint_count-1)))
		b = np.zeros((8*(waypoint_count-1),))

		# Initial and final conditions add 8 equations

		# Four initial conditions: 

		# Initial position = x_0
		initial_position_condition = np.array([1,1*t[0],1*t[0]**2,1*t[0]**3,1*t[0]**4,1*t[0]**5,1*t[0]**6,1*t[0]**7])
		M[0, :8] = initial_position_condition
		b[0] = x[0]

		# Initial velocity = 0
		initial_velocity_condition = np.array([0,1,2*t[0],3*t[0]**2,4*t[0]**3,5*t[0]**4,6*t[0]**5,7*t[0]**6])
		M[1, :8] = initial_velocity_condition
		b[1] = 0 

		# Initial acceleration = 0
		initial_acceleration_condition = np.array([0,0,2,6*t[0],12*t[0]**2,20*t[0]**3,30*t[0]**4,42*t[0]**5])
		M[2, :8] = initial_acceleration_condition
		b[2] = 0 

		# Initial jerk = 0 
		initial_jerk_condition = np.array([0,0,0,6,24*t[0],60*t[0]**2,120*t[0]**3,210*t[0]**4])
		M[3, :8] = initial_jerk_condition
		b[3] = 0

		# Four final conditions:

		# Final position = x_n
		final_position_condition = np.array([1,1*t[-1],1*t[-1]**2,1*t[-1]**3,1*t[-1]**4,1*t[-1]**5,1*t[-1]**6,1*t[-1]**7])
		M[4, 8*(segment_count-1):] = final_position_condition
		b[4] = x[-1]

		# Final velocity = 0 
		final_velocity_condition = np.array([0,1,2*t[-1],3*t[-1]**2,4*t[-1]**3,5*t[-1]**4,6*t[-1]**5,7*t[-1]**6])
		M[5, 8*(segment_count-1):] = final_velocity_condition
		b[5] = 0

		# Final acceleration = 0
		final_acceleration_condition = np.array([0,0,2,6*t[-1],12*t[-1]**2,20*t[-1]**3,30*t[-1]**4,42*t[-1]**5])
		M[6, 8*(segment_count-1):] = final_acceleration_condition
		b[6] = 0

		# Final jerk = 0
		final_jerk_condition = np.array([0,0,0,6,24*t[-1],60*t[-1]**2,120*t[-1]**3,210*t[-1]**4])
		M[7, 8*(segment_count-1):] = final_jerk_condition
		b[7] = 0

		# Intermediate waypoint conditions

		# Position continuity conditions for the intermediate waypoints add waypoints - 2 equations
		for i in range(1,waypoint_count-1):
			position_continuity_conditions_1 = np.array([1,1*t[i],1*t[i]**2,1*t[i]**3,1*t[i]**4,1*t[i]**5,1*t[i]**6,1*t[i]**7])
			position_continuity_conditions_2 = -1.0 * position_continuity_conditions_1
			position_continuity_conditions = np.append(position_continuity_conditions_1, position_continuity_conditions_2)
			M[8+(i-1), 8*(i-1):8*(i+1)] = position_continuity_conditions
			b[8+(i-1)] = 0

		# Position conditions at the intermediate waypoints add waypoints - 2 equations
		for i in range(1,waypoint_count-1):
			position_conditions = np.array([1,1*t[i],1*t[i]**2,1*t[i]**3,1*t[i]**4,1*t[i]**5,1*t[i]**6,1*t[i]**7])
			M[8+(i-1)+1*(waypoint_count-2), 8*(i-1):8*i] = position_conditions
			b[8+(i-1)+1*(waypoint_count-2)] = x[i]

		# Velocity continuity at the intermediate waypoints add waypoints - 2 equations
		for i in range(1, waypoint_count-1):
			velocity_continuity_conditions_1 = np.array([0,1,2*t[i],3*t[i]**2,4*t[i]**3,5*t[i]**4,6*t[i]**5,7*t[i]**6])
			velocity_continuity_conditions_2 = -1.0 * velocity_continuity_conditions_1
			velocity_continuity_conditions = np.append(velocity_continuity_conditions_1, velocity_continuity_conditions_2)
			M[8+(i-1)+2*(waypoint_count-2), 8*(i-1):8*(i+1)] = velocity_continuity_conditions
			b[8+(i-1)+2*(waypoint_count-2)] = 0

		# Acceleration continuity at the intermediate waypoints add waypoints - 2 equations
		for i in range(1, waypoint_count-1):
			acceleration_continuity_conditions_1 = np.array([0,0,2,6*t[i],12*t[i]**2,20*t[i]**3,30*t[i]**4,42*t[i]**5])
			acceleration_continuity_conditions_2 = -1.0 * acceleration_continuity_conditions_1
			acceleration_continuity_conditions = np.append(acceleration_continuity_conditions_1, acceleration_continuity_conditions_2)
			M[8+(i-1)+3*(waypoint_count-2), 8*(i-1):8*(i+1)] = acceleration_continuity_conditions
			b[8+(i-1)+3*(waypoint_count-2)] = 0

		# Jerk continuity at the intermediate waypoints adds waypoints - 2 equations
		for i in range(1, waypoint_count-1):
			jerk_continuity_conditions_1 = np.array([0,0,0,6,24*t[i],60*t[i]**2,120*t[i]**3,210*t[i]**4])
			jerk_continuity_conditions_2 = -1.0 * jerk_continuity_conditions_1
			jerk_continuity_conditions = np.append(jerk_continuity_conditions_1, jerk_continuity_conditions_2)
			M[8+(i-1)+4*(waypoint_count-2), 8*(i-1):8*(i+1)] = jerk_continuity_conditions
			b[8+(i-1)+4*(waypoint_count-2)] = 0

		# Snap continuity at the intermediate waypoints adds waypoints - 2 equations
		for i in range(1, waypoint_count-1):
			snap_continuity_conditions_1 = np.array([0,0,0,0,24,120*t[i],360*t[i]**2,840*t[i]**3])
			snap_continuity_conditions_2 = -1.0 * snap_continuity_conditions_1
			snap_continuity_conditions = np.append(snap_continuity_conditions_1, snap_continuity_conditions_2)
			M[8+(i-1)+5*(waypoint_count-2), 8*(i-1):8*(i+1)] = snap_continuity_conditions
			b[8+(i-1)+5*(waypoint_count-2)] = 0

		# Crackle (5th derivative of position) continuity at the intermediate waypoints adds waypoints-2 equations
		for i in range(1, waypoint_count-1):
			crackle_continuity_conditions_1 = np.array([0,0,0,0,0,120,720*t[i],2520*t[i]**2])
			crackle_continuity_conditions_2 = -1.0 * crackle_continuity_conditions_1
			crackle_continuity_conditions = np.append(crackle_continuity_conditions_1, crackle_continuity_conditions_2)
			M[8+(i-1)+6*(waypoint_count-2), 8*(i-1):8*(i+1)] = crackle_continuity_conditions
			b[8+(i-1)+6*(waypoint_count-2)] = 0

		# Pop (6th derivative of position) continuity at the intermediate waypoints adds waypoints-2 equations
		for i in range(1, waypoint_count-1):
			pop_continuity_conditions_1 = np.array([0,0,0,0,0,0,720,5040*t[i]])
			pop_continuity_conditions_2 = -1.0 * pop_continuity_conditions_1
			pop_continuity_conditions = np.append(pop_continuity_conditions_1, pop_continuity_conditions_2)
			M[8+(i-1)+7*(waypoint_count-2), 8*(i-1):8*(i+1)] = pop_continuity_conditions
			b[8+(i-1)+7*(waypoint_count-2)] = 0

		# Solve the linear system of equations
		c = np.linalg.solve(M, b)

		# Plot the results
		#fig, ax = plt.subplots()
		all_position_values = []
		all_time_values = []
		for i in range(waypoint_count-1):
			time_values = np.linspace(t[i], t[i+1], num=1000)
			all_time_values.append(time_values)
			position_values = np.polyval(c[8*i:8*(i+1)][::-1], time_values)
			all_position_values.append(position_values)
			#ax.plot(time_values, position_values, label=str(i))
		#ax.scatter(t, x, label="Waypoints")
		#plt.legend()
		#plt.title("Position")
		#plt.show()

		#fig, ax = plt.subplots()
		for i in range(waypoint_count-1):
			time_values = np.linspace(t[i], t[i+1], num=1000)
			velocity_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=1)
			velocity_values = np.polyval(velocity_coeff, time_values)
			#ax.plot(time_values, velocity_values, label=str(i))
		#plt.legend()
		#plt.title("Velocity")
		#plt.show()

		#fig, ax = plt.subplots()
		for i in range(waypoint_count-1):
			time_values = np.linspace(t[i], t[i+1], num=1000)
			acceleration_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=2)
			acceleration_values = np.polyval(acceleration_coeff, time_values)
			#ax.plot(time_values, acceleration_values, label=str(i))
		#plt.legend()
		#plt.title("Acceleration")
		#plt.show()

		#fig, ax = plt.subplots()
		for i in range(waypoint_count-1):
			time_values = np.linspace(t[i], t[i+1], num=1000)
			jerk_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=3)
			jerk_values = np.polyval(jerk_coeff, time_values)
			#ax.plot(time_values, jerk_values, label=str(i))
		#plt.legend()
		#plt.title("Jerk")
		#plt.show()

		#fig, ax = plt.subplots()
		all_snap_values = []
		for i in range(waypoint_count-1):
			time_values = np.linspace(t[i], t[i+1], num=1000)
			snap_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=4)
			snap_values = np.polyval(snap_coeff, time_values)
			all_snap_values.append(snap_values)
			#ax.plot(time_values, snap_values, label=str(i))
		#plt.legend()
		#plt.title("Snap")
		#plt.show()


		# Compute the cost of the path using the cost function (the objective function)
		all_time_values = np.array(all_time_values).flatten()
		all_snap_values = np.array(all_snap_values).flatten()
		all_position_values = np.array(all_position_values).flatten()

		time_diffs = np.diff(all_time_values)
		snap_avg = 0.5 * (all_snap_values[:-1] + all_snap_values[1:])
		cost += np.sum(time_diffs * snap_avg**2)
		position_list.append(all_position_values)
		snap_list.append(all_snap_values)
		coeffs_list.append(c)

	return (cost, all_time_values, position_list, snap_list, speed, coeffs_list, t)


def optimize_snap(x,y,z, valid_average_speeds, max_accel=4, max_jerk=4, max_snap=1):
	# Tries to assign times using the valid average speeds list. Returns the trajectory that has the lowest cost integral |snap|^2 * dt
	candidate_trajectory = []
	for speed in valid_average_speeds:
		cost, time_data, position_data, snap_data, speed, coeffs_list, t = minimum_snap_trajectory(x,y,z,speed)
		heapq.heappush(candidate_trajectory, (cost, (time_data, position_data, snap_data, speed, coeffs_list, t)))
		print("max time",np.max(time_data))
	cost, item = heapq.heappop(candidate_trajectory) # Pop the lowest-cost trajectory item
	while np.max(item[0]) > 500:
		cost, item = heapq.heappop(candidate_trajectory)


	return cost, item

cost, item = optimize_snap(x,y,z,valid_average_speeds)


time_data, position_data, snap_data, speed, coeffs_list, t = item
fig, axs = plt.subplots(2,3, figsize=(15,15))
axs[0,0].plot(position_data[0],position_data[1], label=f"Minimum snap trajectory {speed} m/s average speed", color="red")
for i in range(len(obstacles)):
	center = obstacles[i, :3]
	halfsize = obstacles[i, 3:]
	if center[2] + halfsize[2] >= 5:
		rect = plt.Rectangle((center[0] - halfsize[0], center[1] - halfsize[1]), 
							 2*halfsize[0], 2*halfsize[1], 
							 color='Gray', alpha=0.5)
		axs[0,0].add_patch(rect)
axs[0,0].scatter([shortened_path[:, 0]], [shortened_path[:, 1]], s=5, color=(0,0,1), label='RRT path')
axs[0,0].scatter(safe_path[:,0][::skip_count], safe_path[:,1][::skip_count], alpha=1, color=(0,.5,.5), s=3, label='Potential field waypoints')
axs[0,0].legend()
axs[0,1].plot(time_data, position_data[1], label="Y vs T")
axs[0,1].legend()
axs[0,2].plot(time_data, snap_data[0], label="snap_x")
axs[0,2].plot(time_data, snap_data[1], label="snap_y")
axs[0,2].legend()
plt.show()


# A function that takes an input time value and returns the set of 8 polynomial coefficients that apply for position and the 8 that apply for desired velocity
# at that time.
def time_to_coeffs(given_time, t, coeffs_list):
	target_waypoint_number = np.searchsorted(t, given_time)
	segment_index = target_waypoint_number - 1
	x_coeffs = coeffs_list[0]
	y_coeffs = coeffs_list[1]
	z_coeffs = coeffs_list[2]

	coeff_start_index = segment_index * 8
	coeff_end_index = (segment_index + 1) * 8

	x_position_coeffs = x_coeffs[coeff_start_index:coeff_end_index]
	x_velocity_coeffs = np.polyder(x_position_coeffs[::-1])

	y_position_coeffs = y_coeffs[coeff_start_index:coeff_end_index]
	y_velocity_coeffs = np.polyder(y_position_coeffs[::-1])

	z_position_coeffs = z_coeffs[coeff_start_index:coeff_end_index]
	z_velocity_coeffs = np.polyder(y_position_coeffs[::-1])

	return x_position_coeffs, y_position_coeffs, z_position_coeffs, x_velocity_coeffs, y_velocity_coeffs, z_velocity_coeffs

x_position_coeffs, y_position_coeffs, z_position_coeffs,  x_velocity_coeffs, y_velocity_coeffs, z_velocity_coeffs = time_to_coeffs(2, t, coeffs_list)

# 3. Run low-snap trajectory algorithm to turn the waypoint sequence into a trajectory
# 4. Run the control loop, logging the drone's state at each timestep, along with the error in the trajectory variables vs. time.
# 5. Visualize the results using a 3d scatter or line plot, with the desired trajectory in 
#    one color and the actual trajectory in another color.



#####################################################
#####################################################
#####################################################
# Control loop 

# Initialize a drone
# Initialize a controller paired with the drone
# Evaluate subcontroller 1 to compute F
# Evaluate subcontroller 2 to compute R13c, R23c
# Evaluate subcontroller 3 to compute p_c, q_c
# Evaluate subcontroller 4 to compute r_c
# Evaluate subcontroller 5 to compute Tx, Ty, Tz
# Send F, Tx, Ty, Tz to the drone the update its state, then repeat this loop using the next timestamp.


# Based on the timestep, determine the desired waypoint (x, y, z, x_dot, y_dot, z_dot, psi)
drone = Drone()
controller = Controller(drone)

# Logging
trajectory_x, trajectory_y, trajectory_z, trajectory_psi = [], [], [], []
trajectory_t = []
x_desired_history, y_desired_history, z_desired_history, psi_desired_history = [], [], [], []
f1_history, f2_history, f3_history, f4_history = [], [], [], []

# Time Settings
dt = 0.01
total_time = np.max(time_data)
time_steps = int(total_time / dt)


# Simulation
for time_step in range(time_steps):
	pdb.set_trace()
	# Desired state
	time_value = time_step*dt
	x_desired = float(np.polyval(time_to_coeffs(time_value, t, coeffs_list)[0][::-1], time_value))
	y_desired = float(np.polyval(time_to_coeffs(time_value, t, coeffs_list)[1][::-1], time_value))
	z_desired = float(np.polyval(time_to_coeffs(time_value, t, coeffs_list)[2][::-1], time_value))
	psi_desired = 0
	x_dot_desired = float(np.polyval(time_to_coeffs(time_value, t, coeffs_list)[3], time_value))
	y_dot_desired = float(np.polyval(time_to_coeffs(time_value, t, coeffs_list)[4], time_value))
	z_dot_desired = float(np.polyval(time_to_coeffs(time_value, t, coeffs_list)[5], time_value))
	x_desired_history.append(x_desired)
	y_desired_history.append(y_desired)
	z_desired_history.append(z_desired)
	psi_desired_history.append(psi_desired)


	F = controller.subcontroller_1(z_desired, z_dot_desired)
	R_13c, R_23c = controller.subcontroller_2(x_desired, x_dot_desired, y_desired, y_dot_desired)
	p_c, q_c = controller.subcontroller_3(R_13c, R_23c)
	r_c = controller.subcontroller_4(psi_desired)
	T_x, T_y, T_z = controller.subcontroller_5(p_c, q_c, r_c)
	drone.set_motor_speeds(F, T_x, T_y, T_z)
	
	trajectory_t.append(time_step)
	trajectory_x.append(drone.state[0])
	trajectory_y.append(drone.state[1])
	trajectory_z.append(drone.state[2])
	trajectory_psi.append(drone.state[8])

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(trajectory_x, trajectory_y, trajectory_z)
ax.plot(x_desired_history, y_desired_history, z_desired_history)
plt.show()
plt.subplot(4,4,1)
plt.plot(trajectory_t, trajectory_x)
plt.plot(trajectory_t, x_desired_history)
plt.subplot(4,4,2)
plt.plot(trajectory_t, trajectory_y)
plt.plot(trajectory_t, y_desired_history)
plt.subplot(4,4,3)
plt.plot(trajectory_t, trajectory_z)
plt.plot(trajectory_t, z_desired_history)
plt.subplot(4,4,4)
plt.plot(trajectory_t, trajectory_psi)
plt.plot(trajectory_t, psi_desired_history)
plt.tight_layout()
plt.show()
pdb.set_trace()

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################











omega_history=[]
state_history=[]
t = []
for i in range(1000):
	drone.set_motor_speeds(-drone.m*drone.g-0.001, 0,0,-0.001)
	t.append(i*drone.dt)
	state_history.append(drone.state)
	omega_history.append(drone.omega)
state_history=np.array(state_history)
plt.subplot(4,4,1)
plt.plot(t, [w[0] for w in omega_history], label='omega 1')
plt.subplot(4,4,2)
plt.plot(t, [w[1] for w in omega_history], label='omega 2')
plt.subplot(4,4,3)
plt.plot(t, [w[2] for w in omega_history], label='omega 3')
plt.subplot(4,4,4)
plt.plot(t, [w[3] for w in omega_history], label='omega 4')
plt.subplot(4,4,5)
plt.plot(t, state_history[:,0], label='x')
plt.legend()
plt.subplot(4,4,6)
plt.plot(t, state_history[:,1], label='y')
plt.legend()
plt.subplot(4,4,7)
plt.plot(t, state_history[:,2], label='z')
plt.legend()
plt.subplot(4,4,8)
plt.plot(t, state_history[:,3], label='x_dot')
plt.legend()
plt.subplot(4,4,9)
plt.plot(t, state_history[:,4], label='y_dot')
plt.legend()
plt.subplot(4,4,10)
plt.plot(t, state_history[:,5], label='z_dot')
plt.legend()
plt.subplot(4,4,11)
plt.plot(t, state_history[:,6], label='phi')
plt.legend()
plt.subplot(4,4,12)
plt.plot(t, state_history[:,7], label='theta')
plt.legend()
plt.subplot(4,4,13)
plt.plot(t, state_history[:,8], label='psi')
plt.legend()
plt.subplot(4,4,14)
plt.plot(t, state_history[:,9], label='p')
plt.legend()
plt.subplot(4,4,15)
plt.plot(t, state_history[:,10], label='q')
plt.legend()
plt.subplot(4,4,16)
plt.plot(t, state_history[:,11], label='r')
plt.legend()

plt.tight_layout()
plt.show()
