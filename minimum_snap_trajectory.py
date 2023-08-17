import numpy as np
import matplotlib.pyplot as plt
import pdb

#x = np.array([0, 10, 30, 50, 30, 50, 60, 70]) #np.array([1, 15, 25, 5, 73, 10, 24, 53, 63, 20, 40, 45, 50, 92])
#y = np.array([0, 5, 10, 15, 20, 25, 30, 35]) # For 2D
#z = np.array([0, 10, 0, 10, 20, 20, 30, 30]) # For 3D

x = np.array([100.        ,  90.08437988,  82.93738482,  79.03167519,
        73.31198808,  66.04442875,  62.73017469,  58.16449276,
        54.46223565,  51.72731901,  50.16273737,  50.03440086,
        52.99040308,  61.08587347,  70.151785  ,  79.03467681,
        88.46502216,  96.83370597, 104.42860821, 111.68648688,
       117.30890181, 122.79220738, 128.11222566, 129.4193877 ,
       127.50589017, 125.01289547, 122.50691334, 121.38517298,
       122.51121621, 122.84274833, 123.0770158 , 123.96785003,
       131.54708223, 141.1106103 , 150.83540684, 160.81884893,
       170.54763674, 180.06136748, 189.54586471, 199.52944761,
       209.3467077 , 219.3290034 , 229.32437245, 239.20702119,
       248.60035158, 258.22690943, 267.8158862 , 277.18046325,
       287.04275357, 296.45229948, 306.06936491, 315.99836709,
       325.82511027, 335.74846459, 345.53438495, 355.38112397,
       364.8688256 , 372.91053519, 377.64758576, 379.39166724,
       382.60853168, 388.34277269, 394.58588649, 403.24621357,
       413.1570321 , 422.74933179, 431.10837509])

y = np.array([ 100.        ,   98.73256776,   92.11265536,   83.03872798,
         74.93330831,   68.14317713,   59.66201095,   50.7714785 ,
         41.48818162,   31.87531604,   22.00648959,   12.02059418,
          2.56550108,   -2.21254043,   -6.20754364,   -9.92593665,
        -13.21456183,  -17.76541562,  -24.2533258 ,  -31.12516025,
        -39.37806459,  -47.73959938,  -56.20547395,  -65.82777552,
        -75.63813301,  -85.32047197,  -95.00048816, -104.91236677,
       -114.83821313, -124.7961481 , -134.7659641 , -144.70397465,
       -149.36762429, -152.17747147, -154.17953412, -154.54699955,
       -156.76012072, -159.81591616, -162.310769  , -162.29060896,
       -164.08424656, -164.26135503, -164.51039809, -165.91259694,
       -169.27432871, -171.92536002, -174.72925451, -177.06077943,
       -178.53205823, -179.63512452, -182.34736497, -183.06416118,
       -181.23237743, -181.33309655, -183.38591465, -185.12313328,
       -188.2112386 , -193.91954572, -202.62882838, -212.46009759,
       -221.63127166, -229.81774162, -237.62547962, -241.94957847,
       -243.18806431, -245.9689601 , -251.09560457])

target_altitude = 5.0
z = target_altitude * np.ones((len(y),))

# Calculate the straight line distance of the path
points = np.column_stack((x, y, z))
differences = np.diff(points, axis=0)
squared_differences = differences**2
distances = np.sqrt(np.sum(squared_differences, axis=1))
cumulative_distances = np.cumsum(distances)
cost = cumulative_distances[-1]

valid_average_speeds = [2., 4., 6., 8., 10., 12.]
pdb.set_trace()

# TODO: Figure out a good way to assign time values to the trajectory.

def minimum_snap_trajectory(x):
	waypoint_count = len(x)
	segment_count = waypoint_count - 1
	number_of_unknowns = 8 * segment_count

	average_speed = valid_average_speeds[0]
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
		print("row index",8+(i-1))

	# Position conditions at the intermediate waypoints add waypoints - 2 equations
	for i in range(1,waypoint_count-1):
		position_conditions = np.array([1,1*t[i],1*t[i]**2,1*t[i]**3,1*t[i]**4,1*t[i]**5,1*t[i]**6,1*t[i]**7])
		M[8+(i-1)+1*(waypoint_count-2), 8*(i-1):8*i] = position_conditions
		b[8+(i-1)+1*(waypoint_count-2)] = x[i]
		print("row index",8+(i-1)+1*(waypoint_count-2))

	# Velocity continuity at the intermediate waypoints add waypoints - 2 equations
	for i in range(1, waypoint_count-1):
		velocity_continuity_conditions_1 = np.array([0,1,2*t[i],3*t[i]**2,4*t[i]**3,5*t[i]**4,6*t[i]**5,7*t[i]**6])
		velocity_continuity_conditions_2 = -1.0 * velocity_continuity_conditions_1
		velocity_continuity_conditions = np.append(velocity_continuity_conditions_1, velocity_continuity_conditions_2)
		M[8+(i-1)+2*(waypoint_count-2), 8*(i-1):8*(i+1)] = velocity_continuity_conditions
		b[8+(i-1)+2*(waypoint_count-2)] = 0
		print("row index",8+(i-1)+2*(waypoint_count-2))

	# Acceleration continuity at the intermediate waypoints add waypoints - 2 equations
	for i in range(1, waypoint_count-1):
		acceleration_continuity_conditions_1 = np.array([0,0,2,6*t[i],12*t[i]**2,20*t[i]**3,30*t[i]**4,42*t[i]**5])
		acceleration_continuity_conditions_2 = -1.0 * acceleration_continuity_conditions_1
		acceleration_continuity_conditions = np.append(acceleration_continuity_conditions_1, acceleration_continuity_conditions_2)
		M[8+(i-1)+3*(waypoint_count-2), 8*(i-1):8*(i+1)] = acceleration_continuity_conditions
		b[8+(i-1)+3*(waypoint_count-2)] = 0
		print("row index",8+(i-1)+3*(waypoint_count-2))

	# Jerk continuity at the intermediate waypoints adds waypoints - 2 equations
	for i in range(1, waypoint_count-1):
		jerk_continuity_conditions_1 = np.array([0,0,0,6,24*t[i],60*t[i]**2,120*t[i]**3,210*t[i]**4])
		jerk_continuity_conditions_2 = -1.0 * jerk_continuity_conditions_1
		jerk_continuity_conditions = np.append(jerk_continuity_conditions_1, jerk_continuity_conditions_2)
		M[8+(i-1)+4*(waypoint_count-2), 8*(i-1):8*(i+1)] = jerk_continuity_conditions
		b[8+(i-1)+4*(waypoint_count-2)] = 0
		print("row index",8+(i-1)+4*(waypoint_count-2))

	# Snap continuity at the intermediate waypoints adds waypoints - 2 equations
	for i in range(1, waypoint_count-1):
		snap_continuity_conditions_1 = np.array([0,0,0,0,24,120*t[i],360*t[i]**2,840*t[i]**3])
		snap_continuity_conditions_2 = -1.0 * snap_continuity_conditions_1
		snap_continuity_conditions = np.append(snap_continuity_conditions_1, snap_continuity_conditions_2)
		M[8+(i-1)+5*(waypoint_count-2), 8*(i-1):8*(i+1)] = snap_continuity_conditions
		b[8+(i-1)+5*(waypoint_count-2)] = 0
		print("row index",8+(i-1)+5*(waypoint_count-2))

	# Crackle (5th derivative of position) continuity at the intermediate waypoints adds waypoints-2 equations
	for i in range(1, waypoint_count-1):
		crackle_continuity_conditions_1 = np.array([0,0,0,0,0,120,720*t[i],2520*t[i]**2])
		crackle_continuity_conditions_2 = -1.0 * crackle_continuity_conditions_1
		crackle_continuity_conditions = np.append(crackle_continuity_conditions_1, crackle_continuity_conditions_2)
		M[8+(i-1)+6*(waypoint_count-2), 8*(i-1):8*(i+1)] = crackle_continuity_conditions
		b[8+(i-1)+6*(waypoint_count-2)] = 0
		print("row index",8+(i-1)+6*(waypoint_count-2))

	# Pop (6th derivative of position) continuity at the intermediate waypoints adds waypoints-2 equations
	for i in range(1, waypoint_count-1):
		pop_continuity_conditions_1 = np.array([0,0,0,0,0,0,720,5040*t[i]])
		pop_continuity_conditions_2 = -1.0 * pop_continuity_conditions_1
		pop_continuity_conditions = np.append(pop_continuity_conditions_1, pop_continuity_conditions_2)
		M[8+(i-1)+7*(waypoint_count-2), 8*(i-1):8*(i+1)] = pop_continuity_conditions
		b[8+(i-1)+7*(waypoint_count-2)] = 0
		print("row index",8+(i-1)+7*(waypoint_count-2))

	# Solve the linear system of equations
	c = np.linalg.solve(M, b)

	# Plot the results
	fig, ax = plt.subplots()
	all_position_values = []
	all_time_values = []
	for i in range(waypoint_count-1):
		time_values = np.linspace(t[i], t[i+1], num=1000)
		all_time_values.append(time_values)
		position_values = np.polyval(c[8*i:8*(i+1)][::-1], time_values)
		all_position_values.append(position_values)
		ax.plot(time_values, position_values, label=str(i))
	ax.scatter(t, x, label="Waypoints")
	plt.legend()
	plt.title("Position")
	#plt.show()

	fig, ax = plt.subplots()
	for i in range(waypoint_count-1):
		time_values = np.linspace(t[i], t[i+1], num=1000)
		velocity_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=1)
		velocity_values = np.polyval(velocity_coeff, time_values)
		ax.plot(time_values, velocity_values, label=str(i))
	plt.legend()
	plt.title("Velocity")
	#plt.show()

	fig, ax = plt.subplots()
	for i in range(waypoint_count-1):
		time_values = np.linspace(t[i], t[i+1], num=1000)
		acceleration_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=2)
		acceleration_values = np.polyval(acceleration_coeff, time_values)
		ax.plot(time_values, acceleration_values, label=str(i))
	plt.legend()
	plt.title("Acceleration")
	#plt.show()

	fig, ax = plt.subplots()
	for i in range(waypoint_count-1):
		time_values = np.linspace(t[i], t[i+1], num=1000)
		jerk_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=3)
		jerk_values = np.polyval(jerk_coeff, time_values)
		ax.plot(time_values, jerk_values, label=str(i))
	plt.legend()
	plt.title("Jerk")
	#plt.show()

	fig, ax = plt.subplots()
	all_snap_values = []
	for i in range(waypoint_count-1):
		time_values = np.linspace(t[i], t[i+1], num=1000)
		snap_coeff = np.polyder(c[8*i:8*(i+1)][::-1], m=4)
		snap_values = np.polyval(snap_coeff, time_values)
		all_snap_values.append(snap_values)
		ax.plot(time_values, snap_values, label=str(i))
	plt.legend()
	plt.title("Snap")
	#plt.show()

	return (np.array(all_position_values).flatten(), np.array(all_snap_values).flatten(), np.array(all_time_values).flatten())

from mpl_toolkits.mplot3d import Axes3D
position_values = []
for pos in (x, y, z):
	position_values.append(minimum_snap_trajectory(pos)[0])
pdb.set_trace()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(position_values[0], position_values[1], position_values[2], label="Minimum snap trajectory", color="red")
ax.scatter(x, y, z, label="Waypoints")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.show()



