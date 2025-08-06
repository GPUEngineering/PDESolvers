# We will use the 5-point Laplacian aprroximation stencil explicit method 
# below to simulate the 2D Heat Equation
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Global Constants
xLength = 50
yLength = 50
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 200
numPointsTime = 60

# Length Vector plotted on x-axis and y-axis
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
# Time Vector plotted on z-axis
timeDomain = np.linspace(0, maxTime, numPointsTime)

# Step-sizes- dt, dx, dy
timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# Stability condition: dt <= dx^2/(2*k)
err_dt= np.abs((spaceStepSize**2)/(2*diffusivityConstant))
print(err_dt)
assert(timeStepSize <= err_dt)

# Calculate lambda constant for x and y (they're the same if dx = dy)
lambdaConstant = (diffusivityConstant * timeStepSize) / (spaceStepSize**2)
# Stability condition: kdt/dx^2 <= 0.5
assert(lambdaConstant <= 0.5) 
print(lambdaConstant)

# lambda functions for initial condition and boundary condtions
# 2 sin x sin 2y + 3 sin 4x sin 5y + t
u0 = lambda x, y: 2 * np.sin(x) * np.sin(2*y) + 3 * np.sin(4*x) * np.sin(5*y)
left = lambda t, y: 2 * np.sin(0) * np.sin(2*y) + 3 * np.sin(4*0) * np.sin(5*y) + t
right = lambda t, y: 2 * np.sin(xLength) * np.sin(2*y) + 3 * np.sin(4*xLength) * np.sin(5*y) + t
bottom = lambda t, x: 2 * np.sin(x) * np.sin(2*0) + 3 * np.sin(4*x) * np.sin(5*0) + t 
top = lambda t, x: 2 * np.sin(x) * np.sin(2*yLength) + 3 * np.sin(4*x) * np.sin(5*yLength) + t


# error assertion for intial & boundary conditions
err1_u0 = np.abs(u0(0, 0) - left(0, 0))
assert(err1_u0 < 1e-12)
err2_u0 = np.abs(u0(0, yLength) - right(0, 0))
assert(err2_u0 < 1e-12)
err3_u0 = np.abs(u0(0, 0) - bottom(0, 0))
assert(err3_u0 < 1e-12)
err4_u0 = np.abs(u0(0, xLength) - top(0, 0))
assert(err4_u0 < 1e-12)

def initMatrix():
    # initialize an empty matrix
    matrix = np.empty((numPointsTime, numPointsSpace, numPointsSpace))
    # set the boundary conditions for the entire 2D space
    for tau in range(numPointsTime):
        for i in range(numPointsSpace):
            matrix[tau, i, 0] = left(timeDomain[tau], xDomain[i])
            matrix[tau, i, -1] = right(timeDomain[tau], xDomain[i])
    for tau in range(numPointsTime):
        for j in range(numPointsSpace):
            matrix[tau, 0, j] = bottom(timeDomain[tau], yDomain[j])
            matrix[tau, -1, j] = top(timeDomain[tau], yDomain[j])
    # set the initial condition for the entire 2D space at t=0
    for i in range(len(xDomain)):
        for j in range(len(yDomain)):
            matrix[0,i,j] = u0(xDomain[i], yDomain[j])
    
    return matrix


# Time-stepping loop
def calculateTemperature(U):
    for tau in range(0, numPointsTime-1, 1):
        for i in range(1, numPointsSpace-1, 1):
            for j in range(1, numPointsSpace-1, 1):
                # 5-point stencil implementation
                U[tau+1,i,j] = U[tau,i,j] + lambdaConstant * (
                    # x-direction terms
                    (U[tau,i-1,j] - 2*U[tau,i,j] + U[tau,i+1,j]) +
                    # y-direction terms
                    (U[tau,i,j-1] - 2*U[tau,i,j] + U[tau,i,j+1])
                )
    return U


def plot_surface(u_k, k, ax):
    ax.clear()
    X, Y = np.meshgrid(xDomain, yDomain)
    
    surf = ax.plot_surface(X, Y, u_k, 
                          cmap='jet',
                          vmin=0, vmax=100)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Temperature')
    ax.set_title(f'Temperature at t = {k*timeStepSize:.3f}')
    ax.view_init(elev=30, azim=45)
    return surf

# Create figure and 3D axes outside animation loop
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def animate(k):
    plot_surface(tempMatrix[k], k, ax)

emptyMatrix = initMatrix()
tempMatrix = calculateTemperature(emptyMatrix)


anim = FuncAnimation(fig, animate, interval=10, frames=numPointsTime, repeat=False)
anim.save("heat_equation_solution_surf_explicit.gif", writer='pillow')