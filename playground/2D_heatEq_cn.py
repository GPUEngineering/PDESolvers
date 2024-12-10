# We will use the 5-point Laplacian aprroximation stencil explicit method 
# below to simulate the 2D Heat Equation
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as pyPlot
from scipy.sparse import diags, block_diag ,kron, eye
from scipy.sparse.linalg import spsolve

# Global Constants
xLength = 10
yLength = 10
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 200
numPointsTime = 200

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
# eps = 1e-12
# err1 = np.abs(u0(0,0) - left(0, 0))
# err2 = np.abs(u0(0,0) - down(0, 0))
# assert(err1 < eps)

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

# Set-up the sparse matrix G

# set up matrix coefficients
cx = (diffusivityConstant * timeStepSize) / (2 * spaceStepSize**2)  # x-direction constant
cy = (diffusivityConstant * timeStepSize) / (2 * spaceStepSize**2)  # y-direction constant
alpha = 1 + 2 * cx + 2 * cy  # Main diagonal
beta = 1 - 2 * cx - 2 * cy

# Number of interior points
nx = numPointsSpace - 2
ny = numPointsSpace - 2
N = nx * ny

# # set up main diagonal and off-diagonals
# triDiag = diags([-cx, alpha, -cx], [-1, 0, 1], format='csr')
# offTriDiag = diags([0, -cy, 0], [-1, 0, 1], format='csr')
# G = block_diag([offTriDiag, triDiag, offTriDiag], [-1, 0, 1], format='csr')

# Construct 1D operator for x-direction (Nx x Nx)
Ax = diags([-cx, alpha, -cx], [-1, 0, 1], shape=(nx, nx), format='csr')

# Construct 2D operator using Kronecker product
Ix = eye(nx, format='csr')  # Identity for x
Iy = eye(ny, format='csr')  # Identity for y

# Combine x and y contributions
Lx = kron(Iy, Ax)  # Tridiagonal blocks (x-direction)
Ly = kron(diags([-1, 1], [-1, 1], shape=(ny, ny)), Ix) * -cy  # Off-tridiagonal blocks (y-direction)

# Final sparse matrix
G = Lx + Ly
print (G)

# Time-stepping loop
for tau in range(1, numPointsTime):
    # Right-hand side (RHS) based on previous time step
    rhs = (cx * tempMatrix[0:-2, 1:-1, tau-1] + 
           beta * tempMatrix[1:-1, 1:-1, tau-1] + 
           cx * tempMatrix[2:, 1:-1, tau-1] +
           cy * tempMatrix[1:-1, 0:-2, tau-1] +
           cy * tempMatrix[1:-1, 2:, tau-1])
    
    # Flatten RHS to 1D array
    rhs = rhs.flatten()
    # Solve for the current time step's interior points
    solution = spsolve(G, rhs)
    
    # Update the temperature matrix with the new time step values for interior points
    # Reshape solution back to 2D and update tempMatrix
    tempMatrix[1:-1, 1:-1, tau] = solution.reshape((nx, ny))

print(tempMatrix)