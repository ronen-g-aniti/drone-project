import numpy as np

# Import the obstacle data into a NumPy array
obstacles = np.genfromtxt('colliders.csv', delimiter=',', skip_header=2)

# Measure the boundaries of the environment
north_min = np.min(obstacles[:, 0] - obstacles[:, 3], axis=0)
east_min = np.min(obstacles[:, 1] - obstacles[:, 4], axis=0)
north_max = np.max(obstacles[:, 0] + obstacles[:, 3], axis=0)
east_max = np.max(obstacles[:, 1] + obstacles[:, 4], axis=0)
alt_min = np.max(obstacles[:, 2] - obstacles[:, 5], axis=0)
alt_max = np.max(obstacles[:, 2] + obstacles[:, 5], axis=0)

# Compute the total size of the environment
dimensions = (north_max - north_min, east_max - east_min, alt_max - alt_min)
print(f"North is {dimensions[0]} meters across; East is {dimensions[1]} meters across; Altitude is {dimensions[2]} meters across")

# Compute the total volume of the environment
total_volume = dimensions[0] * dimensions[1] * dimensions[2]
print(f"The total volume is {total_volume} cubic meters")

# Compute the total volume of all obstacles
obstacle_volume = np.sum(2 * obstacles[:, 3] * 2 * obstacles[:, 4] * 2 * obstacles[:, 5])
print(f"The obstacle volume is {obstacle_volume} cubic meters")

# Compute the fraction of the total volume occupied by obstacles
fraction = obstacle_volume / total_volume
print(f"The fraction of the total volume occupied by obstacles is {fraction}")

# Compute the fraction of the total base area occupied by obstacles
total_obstacle_base_area = np.sum(2 * obstacles[:, 3] * 2 * obstacles[:, 4])
total_base_area = dimensions[0] * dimensions[1]
fraction = total_obstacle_base_area / total_base_area
print(f"The fraction of the base area occupied by obstacles is {fraction}")

# Compute the total volume that of the space within the altitude bounds 0 to 100 meters.
subvolume = dimensions[0] * dimensions[1] * 100.0
obstacle_volumes = 0
for x, y, z, hx, hy, hz  in obstacles:
	if z + hz > 100.0:
		obstacle_volumes += 2 * hx * 2 * hy * 100
	else:
		obstacle_volumes += 2 * hx * 2 * hy * 2 * hz
fraction = obstacle_volumes / subvolume
print(f"The fraction of the environment bounded between 0 and 100 meters that is occupied by obstacles is {fraction} ({obstacle_volumes} out of {subvolume})")

# Compute the total volume that of the space within the altitude bounds 0 to 50.0 meters.
subvolume = dimensions[0] * dimensions[1] * 50.0
obstacle_volumes = 0
for x, y, z, hx, hy, hz  in obstacles:
	if z + hz > 50.0:
		obstacle_volumes += 2 * hx * 2 * hy * 50
	else:
		obstacle_volumes += 2 * hx * 2 * hy * 2 * hz
fraction = obstacle_volumes / subvolume
print(f"The fraction of the environment bounded between 0 and 50 meters that is occupied by obstacles is {fraction} ({obstacle_volumes} out of {subvolume})")

# Compute the total volume that of the space within the altitude bounds 0 to 100 meters.
subvolume = dimensions[0] * dimensions[1] * 120
obstacle_volumes = 0
for x, y, z, hx, hy, hz  in obstacles:
	if z + hz > 120.0:
		obstacle_volumes += 2 * hx * 2 * hy * 120
	else:
		obstacle_volumes += 2 * hx * 2 * hy * 2 * hz
fraction = obstacle_volumes / subvolume
free_volume = subvolume - obstacle_volume
print(f"The fraction of the environment bounded between 0 and 120 meters that is occupied by obstacles is {fraction} ({obstacle_volumes} out of {subvolume})")
print(f"The ratio of obstacle volume to free volume is {obstacle_volume} to {free_volume} ({obstacle_volume/free_volume})")