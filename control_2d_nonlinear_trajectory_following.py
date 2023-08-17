import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 1.0
I = 1.0
l = 0.5
g = 9.81

# PD Gains

Kp_x, Kd_x, ki_x = 1, 1, 2
Kp_y, Kd_y, ki_y = 2.1, 25, 2
Kp_phi, Kd_phi, ki_phi = 400, 30, 2#400, 30

# Desired State
x_desired, y_desired, phi_desired = -1, 1, 0.0
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
total_time = 10.0
time_steps = int(total_time / dt)

# Logging
trajectory_x, trajectory_y, trajectory_phi = [], [], []
trajectory_t = []
x_desired_history, y_desired_history = [], []
# Simulation
for t in range(time_steps):

	# Desired state
	x_desired = t * dt
	A = 5
	omega = 2 * np.pi * 1/10
	y_desired = A * np.sin(omega * t * dt)
	x_dot_desired = 1.0
	y_dot_desired = A * omega * np.cos(omega * t * dt)
	x_desired_history.append(x_desired)
	y_desired_history.append(y_desired)


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

	#if abs(x - x_desired) < 0.01 and abs(y - y_desired) < 0.01 and abs(phi - phi_desired) < 0.01:
	#	break

	# Logging
	trajectory_x.append(x)
	trajectory_y.append(y)
	trajectory_phi.append(phi)
	trajectory_t.append(t)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(2,2,1)
plt.plot(trajectory_x, trajectory_y, label="Trajectory")
plt.scatter([x_desired], [y_desired], color="red", label="Goal")
plt.plot(x_desired_history, y_desired_history, label="Desired")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.subplot(2,2,2)
plt.plot(trajectory_t, trajectory_phi, label="Phi")
plt.title("Phi")
plt.legend()
plt.subplot(2,2,3)
plt.plot(trajectory_t, trajectory_x, label="X")
plt.plot(trajectory_t, x_desired_history)
plt.title("X")
plt.legend()
plt.subplot(2,2,4)
plt.plot(trajectory_t, trajectory_y, label="Y")
plt.plot(trajectory_t, y_desired_history)
plt.title("Y")
plt.legend()
plt.tight_layout()
plt.show()