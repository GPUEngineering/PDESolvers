# Crank-Nicolson method is an implicit numerical method
# for derving the solution to PDEs

# we will use the crank-nicolson method below to simulate the 1D Heat Equation
import numpy as np
import matplotlib.pyplot as pyPlot
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Global Constants
xLength = 10
yLength = 10
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 1000
numPointsTime = 2000

# Length Vector plotted on x-axis and y-axis
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
# Time Vector plotted on z-axis
timeDomain = np.linspace(0, maxTime, numPointsTime)

timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# lambda functions for u0- initial condition and alpha and beta- boundary condtions
u0 = lambda n: np.sin(n)
alpha = lambda t, n: 2 * n + 5 * t
beta = lambda t, n: np.sin(n) + 2 * t

# error assertion for intial nd boundary conditions
eps = 1e-12
err = np.abs(u0(0) - alpha(0,0))
assert(err < eps)

# Empty Matrix/NestedList with zeroes
tempMatrix = np.zeros((numPointsSpace, numPointsSpace, numPointsTime))
# X-axis boundaries (first and last rows)
tempMatrix[0, :, :] = alpha(timeDomain, xLength)  # Left x boundary
tempMatrix[-1, :, :] = beta(timeDomain, xLength)  # Right x boundary

# Y-axis boundaries (first and last columns)
tempMatrix[:, 0, :] = alpha(timeDomain, yLength)  # Bottom y boundary 
tempMatrix[:, -1, :] = beta(timeDomain, yLength)  # Top y boundary

# Initial conditions for entire 2D space at t=0
tempMatrix[:, :, 0] = np.outer(u0(xDomain), u0(yDomain))

# lambdaConstant = (diffusivityConstant * timeStepSize) / (2*spaceStepSize**2)
# print(lambdaConstant)

# # Set up tridiagonal matrix coefficients
# mainDiagonal = (1 + 2 * lambdaConstant) * np.ones(numPointsSpace - 2)
# lowerDiagonal = -lambdaConstant * np.ones(numPointsSpace - 3)
# upperDiagonal = -lambdaConstant * np.ones(numPointsSpace - 3)

# # Create the sparse tridiagonal matrix A
# A = diags([lowerDiagonal, mainDiagonal, upperDiagonal], offsets=[-1, 0, 1], format='csr')

# # Time-stepping loop
# for tau in range(1, numPointsTime):
#     # Right-hand side (RHS) based on previous time step
#     rhs = lambdaConstant * tempMatrix[0:-2, tau-1] + (1 - 2 * lambdaConstant) * tempMatrix[1:-1, tau-1] + lambdaConstant * tempMatrix[2:, tau-1]
#     rhs[0] += lambdaConstant * tempMatrix[0, tau]    # Incorporate left boundary
#     rhs[-1] += lambdaConstant * tempMatrix[-1, tau]  # Incorporate right boundary
    
#     # Solve for the current time step's interior points
#     solution = spsolve(A, rhs)
    
#     # Update the temperature matrix with the new time step values for interior points
#     tempMatrix[1:-1, tau] = solution

# # Visualization
# # Create meshgrid for x and y coordinates
# X, Y = np.meshgrid(xDomain, yDomain)

# # Create the figure and 3D axes
# fig = pyPlot.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface for the last time step
# surface = ax.plot_surface(X, Y, tempMatrix[:,:,-1], cmap='inferno')

# # Set labels and title
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Temperature')
# ax.set_title('2D Heat Diffusion')

# pyPlot.show()