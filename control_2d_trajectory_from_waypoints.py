import numpy as np
import matplotlib.pyplot as plt
import pdb
# Define waypoints
wp = [[0, 0],[60, 80],[172,202],[172, 118],[363, 80],[417, 80],[371, -150]] #172, 202

segment_start_time = 0
average_speed = 0
total_distance_so_far = 0
average_speed = 10
segment_coefficients = []

for i in range(len(wp)-1):
	wp1 = np.array(wp[i])
	wp2 = np.array(wp[i+1])
	segment_distance = np.linalg.norm(wp2-wp1)
	segment_vector = wp2 - wp1
	segment_elapsed_time = segment_distance / average_speed
	segment_velocity = segment_vector / segment_elapsed_time
	segment_coefficients.append({'x0': wp1[0], 'y0': wp1[1], 'vx': segment_velocity[0], 'vy': segment_velocity[1], 'start_time': segment_start_time})
	segment_start_time += segment_elapsed_time
	total_distance_so_far += segment_distance

def time_to_waypoint(given_time):
	for i in range(len(segment_coefficients)):
		if i < len(segment_coefficients)-1:
			if segment_coefficients[i]['start_time'] <= given_time and segment_coefficients[i+1]['start_time'] > given_time:
				segment = segment_coefficients[i]
				x = segment['x0'] + segment['vx'] * (given_time - segment['start_time'])
				y = segment['y0'] + segment['vy'] * (given_time - segment['start_time'])
				vx = segment['vx']
				vy = segment['vy']
				return [x, y, vx, vy]
		else:
			segment = segment_coefficients[-1]
			if segment['start_time'] + segment_elapsed_time >= given_time:
				x = segment['x0'] + segment['vx'] * (given_time - segment['start_time'])
				y = segment['y0'] + segment['vy'] * (given_time - segment['start_time'])
				vx = segment['vx']
				vy = segment['vy']
				return [x, y, vx, vy]
			else:
				given_time = segment['start_time'] + segment_elapsed_time
				x = segment['x0'] + segment['vx'] * (given_time - segment['start_time'])
				y = segment['y0'] + segment['vy'] * (given_time - segment['start_time'])
				vx = segment['vx']
				vy = segment['vy']
				return [x, y, vx, vy]	


# Constants
m = 1.5
I = 1.
l = 0.5
g = 9.81

# PD Gains
Kp_x, Kd_x, ki_x = 0.08, 0.085, 0 #.05, .55
Kp_y, Kd_y, ki_y = 0.02, 20, 0#.05, .55
Kp_phi, Kd_phi, ki_phi = 400,100,0#400, 30

# Desired State
x_desired, y_desired, phi_desired = -1, -1, 0.0
x_dot_desired, y_dot_desired, phi_dot_desired = 0, 0, 0
phi_desired = 0.0

# Initial State
x, y, phi = 0.0, 0.0, 0.0
x_dot, y_dot, phi_dot = 0.0, 0.0, 0.0
integral_term_ex = 0
integral_term_ey = 0
integral_term_e_phi = 0

# Time Settings
dt = 0.01
total_time = segment_coefficients[-1]['start_time'] + segment_elapsed_time
time_steps = int(total_time / dt)

# Logging
trajectory_x, trajectory_y, trajectory_phi = [], [], []
time_log = []
x_desired_log, y_desired_log, x_dot_desired_log, y_dot_desired_log = [], [], [], []
x_dot_error_log, y_dot_error_log, phi_dot_error_log = [], [], []
f1_log, f2_log = [], [] 

# Simulation
for t in range(time_steps):
	x_desired, y_desired, x_dot_desired, y_dot_desired = time_to_waypoint(t * dt)
	phi_desired = 0
	phi_dot_desired = 0


	# Outer Loop (Position Control)
	e_x = x_desired - x
	integral_term_ex += e_x * dt
	u_phi_desired = Kp_x * e_x + Kd_x * (x_dot_desired - x_dot) + ki_x * integral_term_ex

	# Inner Loop (Orientation Control)
	e_phi = u_phi_desired - phi
	integral_term_e_phi += e_phi * dt
	u_phi = Kp_phi * e_phi + Kd_phi * (phi_dot_desired - phi_dot) + ki_phi * integral_term_e_phi

	# Vertical Control (Thrust Control)
	e_y = y_desired - y
	integral_term_ey += e_y * dt
	u_y = Kp_y * e_y + Kd_y * (y_dot_desired - y_dot) + ki_y * integral_term_ey
	total_thrust = m * u_y + m * g # The total thrust is linearly related to the vertical position errors

	# Rotor Forces (The rotors are commanded to output these forces based on the control inputs)
	f1 = 0.5 * total_thrust + (I * u_phi) / (2 * l)
	f2 = 0.5 * total_thrust - (I * u_phi) / (2 * l)

	# Update Dynamics
	y_ddot = (1/m) * ((f1 + f2) * np.cos(phi) - m * g)
	x_ddot = (1/m) * (f1 + f2) * np.sin(phi)  
	phi_ddot = l * (f1 - f2) / I

	x_dot += x_ddot * dt
	y_dot += y_ddot * dt
	phi_dot += phi_ddot * dt

	x += x_dot * dt
	y += y_dot * dt
	phi += phi_dot * dt

	if abs(x - x_desired) < 0.01 and abs(y - y_desired) < 0.01 and abs(phi - phi_desired) < 0.01:
		break

	# Logging
	trajectory_x.append(x)
	trajectory_y.append(y)
	trajectory_phi.append(phi)
	time_log.append(t*dt)

	x_desired_log.append(x_desired)
	y_desired_log.append(y_desired)
	x_dot_desired_log.append(x_dot_desired)
	y_dot_desired_log.append(y_dot_desired)
	x_dot_error_log.append(e_x)
	y_dot_error_log.append(e_y)
	phi_dot_error_log.append(e_phi)
	f1_log.append(f1)
	f2_log.append(f2)


# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(3, 4, 1)
plt.plot(trajectory_x, trajectory_y, label="Trajectory")
plt.plot([p[0] for p in wp], [p[1] for p in wp], color="red", label="Goal")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()

plt.subplot(3, 4, 2)
plt.plot(time_log, trajectory_phi, label="Orientation (phi)")
plt.legend()

plt.subplot(3,4,3)
plt.plot(time_log, trajectory_x, label="X")
plt.plot(time_log, x_desired_log)
plt.legend()

plt.subplot(3,4,4)
plt.plot(time_log, trajectory_y, label="Y")
plt.plot(time_log, y_desired_log)
plt.legend()

plt.subplot(3,4,5)
plt.plot(time_log, list(np.array(trajectory_y) - np.array(y_desired_log)), label="y error")
plt.legend()

plt.subplot(3,4,6)
plt.plot(time_log, list(np.array(trajectory_x) - np.array(x_desired_log)), label="x error")
plt.legend()

plt.subplot(3,4,7)
plt.plot(time_log, x_dot_error_log, label="x_dot error")
plt.legend()

plt.subplot(3,4,8)
plt.plot(time_log, y_dot_error_log, label="y dot error")
plt.legend()

plt.subplot(3,4,9)
plt.plot(time_log, phi_dot_error_log, label="phi dot error")
plt.legend()

plt.subplot(3,4,10)
plt.plot(time_log, f1_log, label="f1")
plt.legend()

plt.subplot(3,4,11)
plt.plot(time_log, f2_log, label="f2")
plt.legend()


plt.tight_layout()
plt.show()