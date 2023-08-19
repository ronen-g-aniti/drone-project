import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.spatial import KDTree

"""
My plan is for this file to contain the code I develop for a motion planning and control simulation of a quadrotor
moving in 3D space. Expect this file to contain sections for obstacle generation, waypoint generation, trajectory
generation, dynamics, and a control system.

Author: Ronen G. Aniti
Developed during the summer of 2023

"""
class Drone:
	"""
	This is a math model of a quadcopter modeled in 3D using 6 degrees of freedom.

	All quantities are measured in standard SI units. 

	The state of the quadcopter is defined by this array:
		X = [x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, p, q, r]

	and the state derivative is this:
		X_dot = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]
	
	Where the quantities are defined in this way:
	x = forward position, inertial frame, m
	y = right position, inertial frame, m
	z = up position, inertial frame, m
	phi = Euler angle of rotation about x axis, inertial frame, rad
	theta = Euler angle of rotation about y axis, inertial frame, rad
	psi = Euler angle of rotation about z axis, inertial frame, rad
	p = angular rate about the x axis, body frame, rad/s
	q = angular rate about the y axis, body frame, rad/s
	r = angular rate about the z axis, body frame, rad/s

	So, I am using a a forward, right, up right-handed coordinate system.
	The positive X axis is forward
	The positive Y axis is to the right
	The positive Z axis is up (in reference to the diagram below--out of the page)
	
	The motor configuration and geometry for the drone is defined as follows:


          X AXIS BODY FRAME

			^
	        .
	        .
	        .
	        .                               (.) Z AXIS (OUT OF THE PAGE) BODY FRAME 
	        .


	| ----- L -------|

	1 -------------- 2
	. .            . .
	.   .        .   .
	.     .    .     .
	.       cm       .       -----------> Y AXIS (BODY FRAME)
	.     .    .     .
	.   .        .   . 
	. .            . .
	4 -------------- 3



	
	Where L is the motor-to-motor distance in meters

	This is the way the motors are rotating:

	Motor 1 = CW
	Motor 2 = CCW
	Motor 3 = CW
	Motor 4 = CCW
	
	The force produced by each motor is directly proportional to the square of the rotors angular speed. The constant
	of proportionality is k_f, and it is a function of the rotor geometry and environmental conditions. The other
	motor constant k_m is the constant of proportionality that defines the relationship between the moment produced by
	each motor according to that motors angular velocity.
	
	The drone is considered to be symmetric. It's mass is assumed to be symmetrically distributed through its body about each axis.
	So, its inertia matrix is diagonal. 
	
	This model assumes that the motors are capable of changing rotational speeds instantaneously. 
	"""

	def __init__(self, L=0.5, mass=0.5, k_f=2e-5, k_m=3e-6, I_x=0.01, I_y=0.01, I_z=0.02):

		self.X = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		self.X_dot = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		self.omega = np.array([0., 0., 0., 0.])
		self.L = 0.5
		self.l = 0.5 / 2 * np.sqrt(2) # This is the distance from the drone's center to each motor
		self.I = np.array([
			[I_x, 0, 0], 
			[0, I_y, 0], 
			[0, 0, I_z]
			])
		self.mass = mass
		self.g = 9.81
		self.k_f = k_f
		self.k_m = k_m



	@property
	def p(self):
		return self.X[9]

	@property
	def q(self):
		return self.X[10]

	@property
	def r(self):
		return self.X[11]

	@property
	def phi(self):
		return self.X[6]

	@property
	def theta(self):
		return self.X[7]

	@property
	def psi(self):
		return self.X[8]

	@property
	def f_1(self):
		return self.k_f * self.omega[0]**2

	@property
	def f_2(self):
		return self.k_f * self.omega[1]**2

	@property
	def f_3(self):
		return self.k_f * self.omega[2]**2

	@property
	def f_4(self):
		return self.k_f * self.omega[3]**2

	@property
	def F(self):
		return self.f_1 + self.f_2 + self.f_3 + self.f_4

	@property
	def tau_x(self):
		return self.L/2 * (self.f_1 + self.f_4 - self.f_2 - self.f_3)

	@property
	def tau_y(self):
		return self.L/2 * (self.f_1 + self.f_2 - self.f_3 - self.f_4)

	@property
	def tau_z(self):
		return self.k_m * (self.omega[0]**2 + self.omega[2]**2 - self.omega[1]**2 - self.omega[3]**2)
	
	def R(self):
		"""
		Returns the 3x3 rotation matrix describing the transformation from the body frame to the inertial frame.
		A rotation about each of the component body-frame axes is defined. The final transformation matrix 
		is the result of multiplying the component transformation matrices in sequence. The reasoning behind
		this has to do with matrix multiplication representing a composite function. R = Rz(Ry(Rx(v)), where v is 
		a position vector in R3 and R represents the composite transformation.
		Since a rotation matrix must not dilate the resulting vector, the rotation matrix's determinant must be 1. 
		In addition, other properties must be satisfied, including 1. The columns and rows must be unit vectors,
		2. The columns and rows must be othogonal to each other R*R^T = R^T*R = I, 3. Its inverse is equal to its transpose.
		3x3 Rotation matrices belong to the mathematical group called the special orthogonal group SO(3) (special orthogonal group of 3x3 matrices with a determinant of 1).
		
		These are the three rotation matrices:
				     | 1          0               0         |
		R_x(phi)   = | 0          cos(phi)       -sin(phi)  |    
		             | 0          sin(phi)        cos(phi)  |

                     | cos(theta)          0          sin(theta) |
		R_y(theta) = | 0                   1          0          |
		             |-sin(theta)          0          cos(theta) |
                     
                     | cos(psi)           -sin(psi)             0 |
		R_z(psi)   = | sin(psi)            cos(psi)             0 |
                     | 0                   0                    1 |
                              
                              |  c(psi)*c(theta)     c(psi)*s(phi)*s(theta) - s(psi)*c(phi)        c(psi)*c(phi)*s(theta) + s(psi)*s(phi)  |
        R = R_z * R_y * R_x = |  s(psi)*c(theta)     s(psi)*s(phi)*s(theta) + c(psi)*c(phi)        s(psi)*c(phi)*s(theta) - c(psi)*s(phi)  |
                              | -s(phi)              s(phi)*c(theta)                               c(phi)*c(theta)                         |
                              
		"""
		R_x = np.array([[1,0,0],[0,np.cos(self.phi),-np.sin(self.phi)],[0,np.sin(self.phi),np.cos(self.phi)]]) # Rotation about the body-frame x-axis by phi radians
		R_y = np.array([[np.cos(self.theta),0,np.sin(self.theta)],[0,1,0],[-np.sin(self.theta),0,np.cos(self.theta)]]) # Rotation aboot the body-frame y-axis by theta radians
		R_z = np.array([[np.cos(self.psi),-np.sin(self.psi),0],[np.sin(self.psi), np.cos(self.psi),0],[0,0,1]]) # Rotation about the body-frame z-axis by psi radians
		R = R_z @ R_y @ R_x
		tolerance = 1e-6
		if abs(np.linalg.det(R) - 1.0) >= tolerance:
			raise ValueError("The determinant is not 1, so this is an invalid rotation matrix.")
		return R


	
	def set_motor_speeds(self, F, tau_x, tau_y, tau_z):
		""" 
		Given F, a collective, body-frame upwards thrust, and tau_x, tau_y, tau_z, a set of body-frame torques,
		compute the required propeller velocities and update them. Also, advance the state of the drone by dt seconds.
		
		This function is rooted in the quadrotor's 3D dynamics. 
		Because of Newton's third law, each motor will produce a torque in the direction opposite to it's rotation as well as an upwards force.

		So, in the body frame, these equations hold at all moments in time:

		k_f * omega_1**2 + k_f * omega_2**2 + k_f * omega_3**2 + k_f * omega_4**2 = F                (Eq. 1) [Force balance in the Z direction]
		L/2 * [k_f * omega_1**2 - k_f * omega_2**2 - k_f * omega_3**2 + k_f * omega_4**2] = tau_x    (Eq. 2) [Torque balance about the X axis]
		L/2 * [k_f * omega_1**2 + k_f * omega_2**2 - k_f * omega_3**2 - k_f * omega_4**2] = tau_y    (Eq. 3) [Torque balance about the Y axis]
		k_m * omega_1**2 - k_m * omega_2**2 + k_m * omega_3**2 - k_m * omega_4**2 = tau_z            (Eq. 4) [Torque balance about the Z axis]
		
		This system can be expressed as a matrix equation:

		| F     |     | k_f,        k_f,        k_f,        k_f         | | omega_1**2 |
		| tau_x |     | k_f * L/2, -k_f * L/2, -k_f * L/2,  k_f * L/2   | | omega_2**2 |
		| tau_y |  =  | k_f * L/2,  k_f * L/2, -k_f * L/2, -k_f * L/2   | | omega_3**2 |
		| tau_z |     | k_m,       -k_m,        k_m,       -k_m         | | omega_4**2 |
		
		So, for some combinations of F, tau_x, tau_y, tau_z values, there does exist a set omega_1 ... omega_4 that satisfies the system. Though, this will not always be the case. 
		For there to exist a set of valid propeller rotational velocities for a given set of input force and moments, these two conditions must be true:
		1. the 4x4 matrix of constants must be invertable
		2. each element of the solution vector ([omega_1 .. omega_4]) must be non-negative
		
		In general, if these two necessary conditions hold, then, mathematically, there will exist a solution for the omega vector. However, this model could be further adjusted in the 
		future to account for other restrictions that would limit the solution set, such as restrictions governing the maximum and minimum possible angular velocities that
		the rotors are able to generate. 
		"""

		
		# Set up the matrix equation

		# 1. Place the inputs into a vector
		input_vector = np.array([F, tau_x, tau_y, tau_z])
		
		# 2. Write out the constant matrix
		k_f = self.k_f
		k_m = self.k_m
		L = self.L
		constant_matrix = np.array([
			[k_f,        k_f,        k_f,        k_f],
			[k_f * L/2, -k_f * L/2, -k_f * L/2,  k_f * L/2],
			[k_f * L/2,  k_f * L/2, -k_f * L/2, -k_f * L/2],
			[k_m,       -k_m,        k_m,       -k_m]
			])
		
		# 3. Attempt to solve the system using NumPy
		omega_squared = np.linalg.solve(constant_matrix, input_vector)
		
		# 4. Exit the code if the system would imply nonreal solutions for omega
		if omega_squared[0] < 0 or omega_squared[1] < 0 or omega_squared[2] < 0 or omega_squared[3] < 0:
			print("The inputs are unable to be resolved into angular velocities")
			print("At least one of the omega_squared quantities is negative")
			return

		# An update to the original code is this: Impose the constraint that each rotor speed must lie on the interval [0, 700] rad/s
		allowable_speed = 700
		if omega_squared[0] > allowable_speed**2 or omega_squared[1] > allowable_speed**2 or omega_squared[2] > allowable_speed**2 or omega_squared[3] > allowable_speed**2:
			print("The desired motor speed exceeds what is an allowable motor speed")
			return

		# 5. Set the angular speed of each rotor based on the solution to the matrix equation.
		# 	 Ensure the direction of rotation is aligned with the free body diagram indicated at the beginning of this code (CW rotation is negative)
		self.omega[0] = -np.sqrt(omega_squared[0])
		self.omega[1] = np.sqrt(omega_squared[1])
		self.omega[2] = -np.sqrt(omega_squared[2])
		self.omega[3] = np.sqrt(omega_squared[3])
		





	def update_linear_derivatives(self, dt=0.01):
		"""
		Now that the speeds of the motors have been set, the state variables can begin to be updated. 
		This function will update the derivatives of the state variables that involve the drone's linear motion.
		In particular, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot will be updated.

		Recall that X_dot = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, phi_dot, theta_dot, psi_dot, r_dot, q_dot, r_dot].
		
		A force balance in each of the three basic directions of the inertial frame (X, Y, and Z) would yield three equations.
		In matrix form, that system is this:

		    | x_ddot |     |    0   |      | 0 |
		m * | y_ddot |  =  |    0   |  + R | 0 |
		    | z_ddot |     | -m * g |      | F |

		or, dividing both sides by mass, m,

		| x_ddot |      |  0  |              | 0 |
		| y_ddot |  =   |  0  |  + (1/m) * R | 0 |
		| z_ddot |      | -g  |              | F |

		The second set of terms on the right hand side represents the x, y, and z components of the thrust force, rotated from the
		body frame into the inertial frame using the drone's 3x3 rotation matrix, R. 
		"""

		term_1 = np.array([0, 0, -self.g])
		term_2 = 1 / self.mass * self.R() @ np.array([0,0,self.F])
		linear_accelerations = term_1 + term_2
		self.X_dot[3] = linear_accelerations[0] 
		self.X_dot[4] = linear_accelerations[1]
		self.X_dot[5] = linear_accelerations[2]
		
		# Set the linear velocities of the state derivative array to be the values of the linear velocities of the 
		# current state. Later, the time step will be integrated these velocities, so the velocities at the start
		# of the time step. 
		self.X_dot[0] = self.X[3]
		self.X_dot[1] = self.X[4]
		self.X_dot[2] = self.X[5]

		# Now, self.X_dot[:5] is updated. These elements are the second derivatives for the linear motion state variables
		# The code will not update the state array yet, but rather, it will update the state array all at once in the advance_state function.

	def update_rotational_derivatives(self):
		"""
		The purpose of this function is to compute p_dot, q_dot, r_dot (self.X_dot[9:]) given the torques about each of the body-frame axes that are being produced by
		the motors. The math and physics behind this function is the Newton-Euler equations of motion describing how the net torques about the principle axes
		of a rigid body affect the body's rotational accelerations.

		| tau_x |       | p_dot |    | p |     | p | 
		| tau_y |  =  I | q_dot | +  | q | X I | q |
		| tau_z |       | r_dot |    | r |     | r |

		  tau           rhs_term_1      rhs_term_2


		Since I is the inertia matrix, and the mass is being assumed the be evenly distributed through the aircraft body, then the first rhs term is this:

		| I_x  0   0  | | p_dot |    | I_x * p_dot | 
		| 0   I_y  0  | | q_dot | =  | I_y * q_dot |    <- ( Another form of rhs_term_1 )
		| 0    0  I_z | | r_dot |    | I_z * r_dot |

		
		For the sake of documentation, I denote this result matrix | I | | omega_dot | or `rhs_term_1`
		"""

		tau = np.array([self.tau_x, self.tau_y, self.tau_z])
		I = self.I
		omega = self.X[9:] # Get p, q, r from the current state array
		rhs_term_2 = np.cross(omega, I @ omega)
		
		# Isolating rhs_term_1
		rhs_term_1 = tau - rhs_term_2
		
		# Dividing out the factor of inertia from each expression
		p_dot = rhs_term_1[0] / self.I[0,0]
		q_dot = rhs_term_1[1] / self.I[1,1]
		r_dot = rhs_term_1[2] / self.I[2,2]

		# Update the state derivative array, self.X_dot, for the p_dot, q_dot, and r_dot elements (elements 10 to 12)
		self.X_dot[9] = p_dot
		self.X_dot[10] = q_dot
		self.X_dot[11] = r_dot

	def update_euler_derivatives(self):
		"""
		The purpose of this function is the compute the Euler rate derivatives and update the corresponding elements
		of the state derivative array, elements 7 to 9. This is an equation from rigid-body mechanics and is provided 
		in the course materials for the Udacity course in the following form:

		| phi_dot    |    | 1     sin(phi)*tan(theta)     cos(phi)*tan(theta) |   | p |
		| theta_dot  | =  | 0     cos(phi)               -sin(phi)            | X | q |
		| psi_dot    |    | 0     sin(phi)*sec(theta)     cos(phi)*sec(theta) |   | r |
		
		In other words, this matrix equation transforms a set of body-frame angular rates into a set of Euler rates, which
		are in reference to the inertial frame. These Euler rates will be integrated in the advance state function to update
		the Euler angle state variables.

		Note, certain values of phi or theta could cause some matrix elements to become undefined. To elaborate, tan(theta) is
		undefined at +- pi/2 radians and sec(theta) is also undefined at those angles. The concequence of some matrix elements
		being undefined would be that the Euler rates would be unable to be determined. This is a drawback of using Euler angles
		to represent attitude rather than using a quaternion representation. 
		"""

		# Exit the routine if a singularity is detected as a concequence of using Euler angles
		tolerance = 1e-6
		if np.abs(np.abs(self.theta) - np.pi/2) < tolerance:
			raise ValueError("Pitch angle is detected to be either positive or negative pi/2 radians, which causes an undefined matrix element (gimbal lock)")

		# Construct the RHS transformation matrix
		transformation_matrix = np.array([
			[1, np.sin(self.phi)*np.tan(self.theta), np.cos(self.phi)*np.tan(self.theta)],
			[0, np.cos(self.phi), -np.sin(self.phi)],
			[0, np.sin(self.phi)/np.cos(self.theta), np.cos(self.phi)/np.cos(self.theta)]
			])

		# Check if the transformation_matrix 
		# Get p, q, r from the state array
		omega = np.array([self.p, self.q, self.r])

		# Multiply the transformation matrix by the angular rate vector
		euler_rates = transformation_matrix @ omega

		phi_dot = euler_rates[0]
		theta_dot = euler_rates[1]
		psi_dot = euler_rates[2]

		self.X_dot[6] = phi_dot
		self.X_dot[7] = theta_dot
		self.X_dot[8] = psi_dot

	def advance_state(self, dt=0.01):
		"""
		This function integrates the state of the drone forward in time by dt seconds. In other words, This function will update the state
		array of the drone to reflect the state of the drone after the motors have been spinning at the newly set angular speeds
		for dt seconds. This function assumes that all 12 elements of the state derivative vector have been updated to reflect the new motor speeds. 
		The math behind this function is Euler integration. The state derivates are assumed to remain constant for the duration of the time interval. This model
		could later incorporate other numerical integration schemes, like Runge-Kutta.
		"""
		# This module uses the Euler angles and the body-frame angular rates to set elements 7 to 9 of the state derivative array: phi_dot, theta_dot, psi_dot
		self.update_euler_derivatives()

		# This module updates elements 10 to 12 of the state derivative array: p_dot, q_dot, r_dot
		self.update_rotational_derivatives() 

		# This module sets elements 1 to 6 of the state derivative array: x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot
		self.update_linear_derivatives() 

		# At this point, all 12 state derivative elements have been updated to reflect the new motor speeds. It's now
		# time to integrate the state of the drone forward in time by dt seconds. 
		self.X += self.X_dot * dt


class Controller:
	"""
	The purpose of this cascaded control system is to compute a valid combination of collective thrust and 
	moments to send to the drone in order to reach a desired state, specified as input. 

	The desired trajectory that is input into the controller must contain the following information:

	x(t), x_ddot(t), x_3dot(t), x_4dot(t)
	y(t), y_ddot(t), y_3dot(t), y_4dot(t)
	z(t), z_ddot(t), z_3dot(t), z_4dot(t)
	psi(t), psi_dot(t), psi_ddot(t), psi_3dot(t), psi_4dot(t)


	"""
	def __init__(self, drone, desired_trajectory):
		self.drone = drone
		self.desired_trajectory = desired_trajectory

	


	def altitude_controller(self):
		pass
	def lateral_position_controller(self):
		pass
	def roll_pitch_controller(self):
		pass
	def yaw_controller(self):
		pass
	def body_rate_controller(self):
		pass
	def compute_signal(self, desired_trajectory_point):
		commanded_signal = (0,0,0,0)
		return commanded_signal

class ObstacleGenerator:
	"""
	The purpose of this class is to provide a way of generating N randomly placed box obstacles of uniform size in a 3D space.
	Each obstacle that is generated is associated with these parameters: [x, y, z, hx, hy, hz]. 
	In this parameterization, x, y, and z are obstacle center positions, and hx, hy, and hz are obstacle halfsizes.
	"""
	def __init__(self, hx=5.0, hy=5.0, hz=50.0, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0, zmin=0.0, zmax=100.0):
		self.hx = hx
		self.hy = hy
		self.hz = hz
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.zmin = zmin
		self.zmax = zmax

	def generate_n_box_obstacles(self, number_of_obstacles):
		x_values = np.random.uniform(self.xmin, self.xmax, size=number_of_obstacles)
		y_values = np.random.uniform(self.ymin, self.ymax, size=number_of_obstacles)
		obstacles = np.column_stack((x_values, y_values, np.full(number_of_obstacles, self.hz), np.full(number_of_obstacles, self.hx), np.full(number_of_obstacles, self.hy), np.full(number_of_obstacles, self.hz)))
		return obstacles


class WaypointGenerator:
	"""
	The purpose of this class is to provide a way of generating a path of (x, y, z, phi) waypoints, given
	an array of obstacle data generated using an ObstacleGenerator object. This class will use the rapidly
	exploring random tree (RRT) algorithm to search for a path between the start state and the goal state.
	Once this path has been established, it will be improved with another set of functions from this class.
	That set of functions incorporates a potential field algorithm to modify the path such that its waypoints
	move further from obstacles than they might be already. This modification will also populate the path
	with additional, safe intermediate waypoints. Once this final path is developed, it will be ready
	to be processed by a TrajectoryGenerator object, which comprises a class of functions that produces a smooth
	trajectory from the waypoint information provided by a WaypointGenerator object. 
	"""

	def __init__(self, obstacles):
		self.start = None
		self.goal = None
		self.explored = None
		self.edges = {}
		self.obstacles = obstacles
		self.obstacle_centers = obstacles[:,:3]
		self.obstacle_halfsizes = obstacles[:,3:]
		self.xbounds, self.ybounds, self.zbounds = self.bounds()
		self.waypoints = None
		self.obstacle_ground_positions = obstacles[:, :2]
		self.obstacle_ground_positions_kd = KDTree(self.obstacle_ground_positions)

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

	def integrate(self, waypoints, step=0.1):
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

	def navigate(self, start, goal, step=15.0, max_iters=100000):
		"""Runs the main algorithm. This is a form of the rapidly-exploring random tree algorithm"""

		self.start = start
		self.goal = goal
		self.explored = np.array([start])

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

		return path

	def thin_out(self, waypoints, skipcount=150):
		return waypoints[::skipcount]

class TrajectoryGenerator:
	pass



####################
####################
####################
####################

# 1. Create the obstacles for this simulation
obstacle_generator = ObstacleGenerator()
obstacles = obstacle_generator.generate_n_box_obstacles(5)


# 2. Generate a sequence of waypoints from the obstacle data
start = np.array([-20,-20,-20])
goal = np.array([120,120,120])
waypoint_generator = WaypointGenerator(obstacles)
waypoints = waypoint_generator.navigate(start, goal)
waypoints = waypoint_generator.shorten(waypoints)
waypoints = waypoint_generator.integrate(waypoints)
waypoints = waypoint_generator.thin_out(waypoints)

fig, ax = plt.subplots()
plt.scatter(waypoints[:,0],waypoints[:,1], s=5, color="blue")
plt.show()

# 3. From the waypoint sequence, generate a trajectory
trajectory_generator = TrajectoryGenerator()
trajectory = trajectory_generator.minimum_snap_trajectory(waypoints)

# 4. Initialize a drone object
drone = Drone()

# 5. Initialize a control system object and link it to the drone object
controller = Controller(drone, desired_trajectory)

# 6. Command the drone to follow the trajectory
drone.set_motor_speeds(10,0,0,0)
drone.advance_state()

# 7. Visualize the state history and the trajectory error against time

# 8. Create a 3D animation of the simulation


####################
####################
####################
####################

# Other code. 
		
"""
I create the following simulation loop to experiment with sending the drone a variety of different input signals.
Some input signals raise value errors. Some don't. The plots provide me an indication of how different signals
affect the drone's motion.
"""
drone = Drone()
drone.set_motor_speeds(10,0,0,0)
state_history = [drone.X.copy()]
omega_history = [drone.omega.copy()]
t = [0]
for _ in range(1000):
	drone.set_motor_speeds(9.81*0.5,0,0,10)
	drone.advance_state()
	state_history.append(drone.X.copy())
	omega_history.append(drone.omega.copy())
	t.append(t[-1] + 0.01)

fig, ax = plt.subplots(6,3)
ax[0,0].plot([timestamp for timestamp in t], [state[0] for state in state_history], label="X-position")
ax[0,0].legend()
ax[0,1].plot([timestamp for timestamp in t], [state[1] for state in state_history], label="Y-position")
ax[0,1].legend()
ax[0,2].plot([timestamp for timestamp in t], [state[2] for state in state_history], label="Z-position")
ax[0,2].legend()
ax[1,0].plot([timestamp for timestamp in t], [state[3] for state in state_history], label="X_dot")
ax[1,0].legend()
ax[1,1].plot([timestamp for timestamp in t], [state[4] for state in state_history], label="Y_dot")
ax[1,1].legend()
ax[1,2].plot([timestamp for timestamp in t], [state[5] for state in state_history], label="Z_dot")
ax[1,2].legend()
ax[2,0].plot([timestamp for timestamp in t], [state[6] for state in state_history], label="phi")
ax[2,0].legend()
ax[2,1].plot([timestamp for timestamp in t], [state[7] for state in state_history], label="theta")
ax[2,1].legend()
ax[2,2].plot([timestamp for timestamp in t], [state[8] for state in state_history], label="psi")
ax[2,2].legend()
ax[3,0].plot([timestamp for timestamp in t], [state[9] for state in state_history], label="p")
ax[3,0].legend()
ax[3,1].plot([timestamp for timestamp in t], [state[10] for state in state_history], label="q")
ax[3,1].legend()
ax[3,2].plot([timestamp for timestamp in t], [state[11] for state in state_history], label="r")
ax[3,2].legend()
ax[4,0].plot([timestamp for timestamp in t], [omega[0] for omega in omega_history], label="omega_1")
ax[4,0].legend()
ax[4,1].plot([timestamp for timestamp in t], [omega[1] for omega in omega_history], label="omega_2")
ax[4,1].legend()
ax[4,2].plot([timestamp for timestamp in t], [omega[0] for omega in omega_history], label="omega_3")
ax[4,2].legend()
ax[5,0].plot([timestamp for timestamp in t], [omega[3] for omega in omega_history], label="omega_4")
ax[5,0].legend()
plt.tight_layout()
plt.show()