import numpy as np
import matplotlib.pyplot as plt
import pdb

# Update method
# Define four x,y waypoints
x0 = -30
x1 = 15
x2 = 40
x3 = 90
x4 = 120
y0 = 50
y1 = 100
y2 = 155
y3 = 140
y4 = 100

wp = np.array([[x0, y0],[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
n = wp.shape[0]

A = np.array([
	[1*x0**3,1*x0**2,1*x0,1,0,0,0,0,0,0,0,0,0,0,0,0],
	[3*x0**2,2*x0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1*x1**3,1*x1**2,1*x1,1,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,1*x1**3,1*x1**2,1*x1,1,0,0,0,0,0,0,0,0],
	[-3*x1**2,-2*x1,-1,0,3*x1**2,2*x1,1,0,0,0,0,0,0,0,0,0],
	[-6*x1,-2,0,0,6*x1,2,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,1*x2**3,1*x2**2,1*x2,1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,1*x2**3,1*x2**2,1*x2,1,0,0,0,0],
	[0,0,0,0,-3*x2**2,-2*x2,-1,0,3*x2**2,2*x2,1,0,0,0,0,0],
	[0,0,0,0,-6*x2,-2,0,0,6*x2,2,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,1*x3**3,1*x3**2,1*x3,1,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,1*x3**3,1*x3**2,1*x3,1],
	[0,0,0,0,0,0,0,0,-3*x3**2,-2*x3,-1,0,3*x3**2,2*x3,1,0],
	[0,0,0,0,0,0,0,0,-6*x3,-2,0,0,6*x3,2,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,1*x4**3,1*x4**2,1*x4,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,3*x4**2,2*x4,1,0]	
		])


b = np.array([y0,0,y1,y1,0,0,y2,y2,0,0,y3,y3,0,0,y4,0])

coeffs = np.linalg.solve(A, b)
poly1 = np.poly1d(coeffs[:4])
poly2 = np.poly1d(coeffs[4:8])
poly3 = np.poly1d(coeffs[8:12])
poly4 = np.poly1d(coeffs[12:])

X1 = np.linspace(wp[0,0], wp[1,0], num=100)
X2 = np.linspace(wp[1,0], wp[2,0], num=100)
X3 = np.linspace(wp[2,0], wp[3,0], num=100)
X4 = np.linspace(wp[3,0], wp[4,0], num=100)


fig, ax = plt.subplots()
plt.plot(X1, poly1(X1), color='blue')
plt.plot(X2, poly2(X2), color='blue')
plt.plot(X3, poly3(X3), color='blue')
plt.plot(X4, poly4(X4), color='blue')
plt.scatter([p[0] for p in wp],[p[1] for p in wp], c='blue')
plt.xlim(-40,125)
plt.ylim(40,250)
rect = plt.Rectangle((55, 75), 
							 20, 40, 
							 color='black')
ax.add_patch(rect)
rect = plt.Rectangle((100, 180), 
							 20, 40, 
							 color='black')
ax.add_patch(rect)

rect = plt.Rectangle((10, 180), 
							 20, 40, 
							 color='black')
ax.add_patch(rect)


plt.title('Cubic Spline Interpolation Example')
plt.show()

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Waypoints
wp = np.array([[-30, 50],[15, 100],[40, 155],[90, 140],[120, 100]])

# Separate x and y coordinates
x = wp[:, 0]
y = wp[:, 1]

# Parameter t along the path
t = np.arange(wp.shape[0])

# Create cubic splines for x and y
cs_x = CubicSpline(t, x, bc_type='natural')
cs_y = CubicSpline(t, y, bc_type='natural')

# Evaluate splines at finer resolution
t_fine = np.linspace(t.min(), t.max(), 500)
x_fine = cs_x(t_fine)
y_fine = cs_y(t_fine)

# Plot
fig, ax = plt.subplots()
plt.plot(x_fine, y_fine, label='Cubic Spline Interpolation')
plt.scatter(x, y, color='red', label='Waypoints')
plt.legend()
plt.scatter([p[0] for p in wp],[p[1] for p in wp], c='blue')
plt.xlim(-40,125)
plt.ylim(40,250)
rect = plt.Rectangle((55, 75), 
							 20, 40, 
							 color='black')
ax.add_patch(rect)
rect = plt.Rectangle((100, 180), 
							 20, 40, 
							 color='black')
ax.add_patch(rect)

rect = plt.Rectangle((10, 180), 
							 20, 40, 
							 color='black')
ax.add_patch(rect)
plt.show()

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Waypoints
wp = np.array([[0, 50, 0],[100, 100, 50],[150,50,100],[175, 150, 100],[100, 200, 150]])

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
ax.legend()
plt.show()


# Add code to import the high level plan from the RRT with the potential field modification.
# Apply cubic splines to further smooth the path between waypoints.
# Polynomial splines provides the advantage of producing a smooth curve whereas the potential field modification may not be as smooth.
