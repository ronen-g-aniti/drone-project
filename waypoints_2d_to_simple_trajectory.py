import numpy as np
import matplotlib.pyplot as plt
import pdb

wp = [[0, 0],[60, 80],[172, 202],[172, 118],[363, 80],[417, 80],[371, -150]]

segment_start_time = 0
average_speed = 0
total_distance_so_far = 0
average_speed = 15
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
		for index, segment in enumerate(segment_coefficients):
			if segment['start_time'] >= given_time:
				x = segment_coefficients[index-1]['x0'] + segment_coefficients[index-1]['vx'] * (given_time - segment_coefficients[index-1]['start_time'])
				y = segment_coefficients[index-1]['y0'] + segment_coefficients[index-1]['vy'] * (given_time - segment_coefficients[index-1]['start_time'])
				vx = segment_coefficients[index-1]['vx']
				vy = segment_coefficients[index-1]['vy']
				return [x, y, vx, vy]

pdb.set_trace()












