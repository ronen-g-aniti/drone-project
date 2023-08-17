import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 1.0
I = 1.0
l = 0.5
g = 9.81

# PD Gains
Kp_x, Kd_x = .8, 1 #.05, .55
Kp_y, Kd_y = .8, 1.5 #.05, .55
Kp_phi, Kd_phi = 400, 30 #400, 30

# Desired State
x_desired, y_desired, phi_desired = -1, -1, 0.0
x_dot_desired, y_dot_desired, phi_dot_desired = 0, 0, 0
phi_desired = 0.0
# Initial State
x, y, phi = 0.0, 0.0, 0.0
x_dot, y_dot, phi_dot = 0.0, 0.0, 0.0

# Time Settings
dt = 0.01
total_time = 100.0
time_steps = int(total_time / dt)

# Logging
trajectory_x, trajectory_y, trajectory_phi = [], [], []

# Simulation
for t in range(time_steps):


	# Outer Loop (Position Control)
	e_x = x_desired - x
	u_phi_desired = Kp_x * e_x + Kd_x * (x_dot_desired - x_dot)

	# Inner Loop (Orientation Control)
	e_phi = u_phi_desired - phi
	u_phi = Kp_phi * e_phi + Kd_phi * (phi_dot_desired - phi_dot) 

	# Vertical Control (Thrust Control)
	e_y = y_desired - y
	u_y = Kp_y * e_y + Kd_y * (y_dot_desired - y_dot)
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

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(trajectory_x, trajectory_y, label="Trajectory")
plt.scatter([x_desired], [y_desired], color="red", label="Goal")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(trajectory_phi, label="Orientation (phi)")
plt.xlabel("Time Step")
plt.ylabel("Phi (radians)")
plt.legend()
plt.tight_layout()
plt.show()