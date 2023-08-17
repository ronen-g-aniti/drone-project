import numpy as np
import matplotlib.pyplot as plt
import pdb
import control as ctrl

m = 0.5#0.5 # kg
g = 9.81
kp = 25
kd = 10
ki = 15
max_allowable_thrust = m * g * 3
t = np.linspace(0, 10, 1000)
time_step=t[1]-t[0]
target_height = 1
z_target_values = np.linspace(target_height, target_height, 1000)
z_dot_target = 0


# m * z_dot_dot_required = T - mg
# Assume down is negative


# P Control
z_actual = 0 
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	command_thrust = kp * (z_target - z_actual)
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	result_z_dot_dot = command_thrust / m - g
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)



z_actual_history = np.array(z_actual_history)
plt.plot(t, z_actual_history, label="P")
plt.plot(t, z_target_values)
#plt.show()

# PD Control
z_actual = 0
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	command_thrust = kp * (z_target - z_actual) + kd * (z_dot_target - z_dot_actual)
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	result_z_dot_dot = command_thrust / m - g
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)



z_actual_history = np.array(z_actual_history)
plt.plot(t, z_actual_history, label="PD")
plt.plot(t, z_target_values)
#plt.show()


# PID Control
z_actual = 0
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
command_thrust_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	integration_term = np.sum(np.array(z_target_values[:time_index] - z_actual_history[:time_index]) * time_step)
	command_thrust = kp * (z_target - z_actual) + kd * (z_dot_target - z_dot_actual) + ki * integration_term
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	command_thrust_history.append(command_thrust)
	result_z_dot_dot = command_thrust / m - g
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)

z_actual_history = np.array(z_actual_history)
plt.plot(t, z_actual_history, label="PID")
plt.plot(t, z_target_values)

# This is a check to see if any commanded thrust exceeds the maximum allowable thrust of the motor
print(np.max(command_thrust_history), "Maximum thrust in Newtons")
if np.max(command_thrust_history) > max_allowable_thrust: 
	print("Can't command that high of a thrust")
print(max_allowable_thrust, "Max allowable thrust in Newtons")

plt.plot(t, z_target_values, label="Step input")
plt.legend()
plt.show()


# PD Control with Feedforward term
kp = 10
kd = 7
z_actual = 0
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
command_thrust_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	command_thrust = kp * (z_target - z_actual) + kd * (z_dot_target - z_dot_actual)  + m*g
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	command_thrust_history.append(command_thrust)
	result_z_dot_dot = command_thrust / m - g
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)

z_actual_history = np.array(z_actual_history)

# For PD control with FF, the control rule could be recast as a second order ODE with constant coefficients. 
# This recasting allows for an understanding of the relationship between kp, kd, natural frequency, and damping ratio. 
# 0 = error_dot_dot + kd * error_dot + kp * error
# The characteristic polynomial is 0 = r^2 + kd * r + kp 
coeffs = [1, kd, kp]
roots = np.roots(coeffs)
print(roots)
# According to lecture, kd is thus 2 * damping_ratio * natural_frequency and
# kd is natural_frequency^2
# Therefore, 
natural_frequency = np.sqrt(kp)
damping_ratio = kd / (2 * natural_frequency)
print(f"For the PD w/FF controlled 1D drone, the natural frequency is {natural_frequency} rad/s and the damping ratio is {damping_ratio}")

# This is a check to see if any commanded thrust exceeds the maximum allowable thrust of the motor
print(np.max(command_thrust_history), "Maximum thrust in Newtons")
if np.max(command_thrust_history) > max_allowable_thrust: 
	print("Can't command that high of a thrust")
print(max_allowable_thrust, "Max allowable thrust in Newtons")

plt.plot(t, z_actual_history, label="PD with FF")
plt.plot(t, z_target_values, label="Step input")
plt.legend()
plt.show()

def tune(damping_ratio, natural_frequency):
	kp = natural_frequency**2
	kd = 2 * damping_ratio * natural_frequency
	return kp, kd

# PD Control with FF, with gains tuned by specifying natural frequency and damping ratio rather than kp and kd directly.
damping_ratio = 0.61
natural_frequency = 3.1 # rad/s
kp, kd = tune(damping_ratio, natural_frequency)
z_actual = 0
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
command_thrust_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	command_thrust = kp * (z_target - z_actual) + kd * (z_dot_target - z_dot_actual)  + m*g
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	command_thrust_history.append(command_thrust)
	result_z_dot_dot = command_thrust / m - g
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)

z_actual_history = np.array(z_actual_history)

# For PD control with FF, the control rule could be recast as a second order ODE with constant coefficients. 
# This recasting allows for an understanding of the relationship between kp, kd, natural frequency, and damping ratio. 
# 0 = error_dot_dot + kd * error_dot + kp * error
# The characteristic polynomial is 0 = r^2 + kd * r + kp 
coeffs = [1, kd, kp]
roots = np.roots(coeffs)
print(roots)
# According to lecture, kd is thus 2 * damping_ratio * natural_frequency and
# kd is natural_frequency^2
# Therefore, 
natural_frequency = np.sqrt(kp)
damping_ratio = kd / (2 * natural_frequency)
print(f"For the PD w/FF controlled 1D drone, the natural frequency is {natural_frequency} rad/s and the damping ratio is {damping_ratio}")

# This is a check to see if any commanded thrust exceeds the maximum allowable thrust of the motor
print(np.max(command_thrust_history), "Maximum thrust in Newtons")
if np.max(command_thrust_history) > max_allowable_thrust: 
	print("Can't command that high of a thrust")
print(max_allowable_thrust, "Max allowable thrust in Newtons")
plt.plot(t, z_actual_history, label="PD with FF")
plt.plot(t, z_target_values, label="Step input")
plt.legend()
plt.show()


## With the PDFF control system, I am now running some simulations to get an idea for how robust the system is.
# PD Control with FF, with gains tuned by specifying natural frequency and damping ratio rather than kp and kd directly.
damping_ratio = 0.61
natural_frequency = 3.1 # rad/s
m = 0.5
kp, kd = tune(damping_ratio, natural_frequency)
z_actual = 0
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
command_thrust_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	command_thrust = kp * (z_target - z_actual) + kd * (z_dot_target - z_dot_actual)  + m*g*1.
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	command_thrust_history.append(command_thrust)
	result_z_dot_dot = command_thrust / (m) - g # Multiplying the mass by a constant simulates a percieved mass error
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)

z_actual_history = np.array(z_actual_history)

# For PD control with FF, the control rule could be recast as a second order ODE with constant coefficients. 
# This recasting allows for an understanding of the relationship between kp, kd, natural frequency, and damping ratio. 
# 0 = error_dot_dot + kd * error_dot + kp * error
# The characteristic polynomial is 0 = r^2 + kd * r + kp 
coeffs = [1, kd, kp]
roots = np.roots(coeffs)
print(roots)
# According to lecture, kd is thus 2 * damping_ratio * natural_frequency and
# kd is natural_frequency^2
# Therefore, 
natural_frequency = np.sqrt(kp)
damping_ratio = kd / (2 * natural_frequency)
print(f"For the PD w/FF controlled 1D drone, the natural frequency is {natural_frequency} rad/s and the damping ratio is {damping_ratio}")

# This is a check to see if any commanded thrust exceeds the maximum allowable thrust of the motor
print(np.max(command_thrust_history), "Maximum thrust in Newtons")
if np.max(command_thrust_history) > max_allowable_thrust: 
	print("Can't command that high of a thrust")
print(max_allowable_thrust, "Max allowable thrust in Newtons")
plt.plot(t, z_actual_history, label="PD with FF")
plt.plot(t, z_target_values, label="Step input")
plt.legend()
plt.show()


## With the PID w/ FF control system, I am now running some simulations to get an idea for how robust the system is.
m = 0.5
mass_error = 1.5
kp, kd = 5, 3
ki=2
z_actual = 0
z_dot_actual = 0
z_dot_dot_actual = 0
z_actual_history = []
command_thrust_history = []
for time_index, time_value in enumerate(t):
	z_target = z_target_values[time_index]
	integration_term = np.sum(np.array(z_target_values[:time_index] - z_actual_history[:time_index]) * time_step)
	command_thrust = kp * (z_target - z_actual) + kd * (z_dot_target - z_dot_actual) + ki * integration_term + m*mass_error*g
	if command_thrust > max_allowable_thrust:
		command_thrust = max_allowable_thrust
	command_thrust_history.append(command_thrust)
	result_z_dot_dot = command_thrust / m - g 
	z_dot_dot_actual = result_z_dot_dot
	z_dot_actual = z_dot_actual + time_step * z_dot_dot_actual
	z_actual = z_actual + time_step * z_dot_actual
	z_actual_history.append(z_actual)

z_actual_history = np.array(z_actual_history)

# This is a check to see if any commanded thrust exceeds the maximum allowable thrust of the motor
print(np.max(command_thrust_history), "Maximum thrust in Newtons")
if np.max(command_thrust_history) > max_allowable_thrust: 
	print("Can't command that high of a thrust")
print(max_allowable_thrust, "Max allowable thrust in Newtons")
plt.plot(t, z_actual_history, label="PID with FF")
plt.plot(t, z_target_values, label="Step input")
plt.legend()
plt.show()

# I want to create a PD Control with FF for the 1D drone that tunes the gains to reach a desired natural frequency and damping ratio.



"""
# I derived the transfer function by hand, and I'm performing some simple analysis of the system's stability
# and performance

# These are the coefficients of the denominator of the transfer function that I derived
coeffs = [m, 2*kd, 2*kp, 2*ki] 
print(np.roots(coeffs))

# This is a check to see if any commanded thrust exceeds the maximum allowable thrust of the motor
print(np.max(command_thrust_history), "Maximum thrust in Newtons")
if np.max(command_thrust_history) > max_allowable_thrust: 
	print("Can't command that high of a thrust")
print(max_allowable_thrust, "Max allowable thrust in Newtons")


# I'm plotting the time response of the system to the input defined at the start of this file
plt.plot(t, z_target_values)
plt.legend()
plt.show()


# I derived the closed loop transfer function by hand, and I'm modeling it using Python's `control` library
numerator = [kd, kp, ki]
denominator = [m, 2*kd, 2*kp, 2*ki]
transfer_function = ctrl.TransferFunction(numerator, denominator)
print(transfer_function)

"""