# We will use the 5-point Laplacian aprroximation stencil explicit method 
# below to simulate the 2D Heat Equation
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as pyPlot

# Global Constants
xLength = 10
yLength = 10
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 200
numPointsTime = 2000

# Length Vector plotted on x-axis and y-axis
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
# Time Vector plotted on z-axis
timeDomain = np.linspace(0, maxTime, numPointsTime)

# Step-sizes- dt, dx, dy
timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# lambda functions for u0- initial condition and alpha and beta- boundary condtions
u0 = lambda x, y: 5 * np.sin(x) + 3 * np.cos(y)
left = lambda t, y: 3 * np.cos(y) + t
right = lambda t, y: 5 * np.sin(xLength) + 3 * np.cos(y) + t
down = lambda t, x: 5 * np.sin(x) + t
up = lambda t, x: 5 * np.sin(x) + 3 * np.cos(yLength) + t
# error assertion for intial nd boundary conditions
eps = 1e-12
err1 = np.abs(u0(0,0) - left(0, 0))
err2 = np.abs(u0(0,0) - down(0, 0))
assert(err1 < eps)

# Empty Matrix/NestedList with zeroes
tempMatrix = np.zeros((numPointsSpace, numPointsSpace, numPointsTime))
# X-axis boundaries (first and last rows)
tempMatrix[0, :, :] = left(timeDomain, yDomain)   # Left x boundary
tempMatrix[-1, :, :] = right(timeDomain, yDomain) # Right x boundary

# Y-axis boundaries (first and last columns)
tempMatrix[:, 0, :] = down(timeDomain, xDomain) # Bottom y boundary 
tempMatrix[:, -1, :] = up(timeDomain, xDomain) # Top y boundary

# Initial conditions for entire 2D space at t=0
tempMatrix[:, :, 0] = np.outer(u0(xDomain, yDomain), u0(xDomain, yDomain))

# Calculate lambda constants for x and y (they're the same if dx = dy)
lambdaConstant = (diffusivityConstant * timeStepSize) / spaceStepSize**2

# Stability condition for 2D heat equation (stricter than 1D)
# assert(lambdaConstant < 0.25)  # Note: 0.25 instead of 0.5 for 2D
print(lambdaConstant)

# Time-stepping loop
for tau in range(1, numPointsTime):
    # Loop over interior points (excluding boundaries)
    for i in range(1, numPointsSpace-1):
        for j in range(1, numPointsSpace-1):
            # 5-point stencil implementation
            tempMatrix[i,j,tau] = tempMatrix[i,j,tau-1] + lambdaConstant * (
                # x-direction terms
                (tempMatrix[i-1,j,tau-1] - 2*tempMatrix[i,j,tau-1] + tempMatrix[i+1,j,tau-1]) +
                # y-direction terms
                (tempMatrix[i,j-1,tau-1] - 2*tempMatrix[i,j,tau-1] + tempMatrix[i,j+1,tau-1])
            )

print(tempMatrix)

# Visualization
# Create meshgrid for x and y coordinates
X, Y = np.meshgrid(xDomain, yDomain)

# Create the figure and 3D axes
fig = pyPlot.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface for the specific time step
surface = ax.plot_surface(X, Y, tempMatrix[:,:,-1], cmap='inferno')

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Temperature')
ax.set_title('2D Heat Diffusion')

pyPlot.show()