# Corrected 2D Heat Equation simulation using explicit finite difference method
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global Constants
xLength = 10  # Lx
yLength = 10  # Ly  
maxTime = 0.5  # tmax
diffusivityConstant = 4  # kappa
numPointsSpace = 50  # x_points = y_points
numPointsTime = 2000  # t_points

# Domain setup
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
timeDomain = np.linspace(0, maxTime, numPointsTime)

# Step sizes
timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# Stability condition: dt <= dx^2/(4*k) for 2D
# stability_limit = (spaceStepSize**2)/(4*diffusivityConstant)
# print(f"Stability limit: {stability_limit:.6f}")
# print(f"Time step: {timeStepSize:.6f}")
# assert(timeStepSize <= stability_limit), "Time step too large for stability!"

# Lambda constant
lambdaConstant = (diffusivityConstant * timeStepSize) / (spaceStepSize**2)
# print(f"Lambda: {lambdaConstant:.6f}")
# assert(lambdaConstant <= 0.25), "Lambda too large for 2D stability!"  # 0.25 for 2D

# Initial condition - a hot spot in the center
u0 = lambda x, y: 10 * np.exp(-((x - xLength/2)**2 + (y - yLength/2)**2) / 2)

# Boundary conditions based on your MATLAB code
# u0 = @(x, y) 20;
u0 = lambda x, y: 20 * np.ones_like(x) if hasattr(x, '__iter__') else 20

# left = @(t,y) 20 + 10*y*(Ly-y)*t^2;  
left = lambda t, y: 20 + 10 * y * (yLength - y) * t**2

# right = @(t, y) 20 + 100*y*(Ly-y)^3*(t-1)^2*(t>1);
right = lambda t, y: 20 + 100 * y * (yLength - y)**3 * (t - 1)**2 * (t > 1)

# up = @(t, x) 20 + 5*x*(Lx-x)*t^4;
top = lambda t, x: 20 + 5 * x * (xLength - x) * t**4

# down = @(t, x) 20;
bottom = lambda t, x: 20                             # Warm top boundary

def initMatrix():
    matrix = np.zeros((numPointsTime, numPointsSpace, numPointsSpace))
    
    # Set boundary conditions for all time steps
    for tau in range(numPointsTime):
        t = timeDomain[tau]
        # Left and right boundaries (x = 0 and x = xLength)
        for i in range(numPointsSpace):
            y = yDomain[i]
            matrix[tau, 0, i] = left(t, y)      # Left boundary (x=0)
            matrix[tau, -1, i] = right(t, y)    # Right boundary (x=xLength)
        
        # Bottom and top boundaries (y = 0 and y = yLength)  
        for j in range(numPointsSpace):
            x = xDomain[j]
            matrix[tau, j, 0] = bottom(t, x)    # Bottom boundary (y=0)
            matrix[tau, j, -1] = top(t, x)      # Top boundary (y=yLength)
    
    # Set initial condition at t=0
    for i in range(numPointsSpace):
        for j in range(numPointsSpace):
            matrix[0, i, j] = u0(xDomain[i], yDomain[j])
    
    # Ensure corners are consistent (use boundary values)
    for tau in range(numPointsTime):
        t = timeDomain[tau]
        matrix[tau, 0, 0] = left(t, yDomain[0])      # Bottom-left
        matrix[tau, 0, -1] = left(t, yDomain[-1])    # Top-left  
        matrix[tau, -1, 0] = right(t, yDomain[0])    # Bottom-right
        matrix[tau, -1, -1] = right(t, yDomain[-1])  # Top-right
    
    return matrix

def calculateTemperature(U):
    for tau in range(numPointsTime-1):
        for i in range(1, numPointsSpace-1):
            for j in range(1, numPointsSpace-1):
                # 5-point stencil for 2D heat equation
                U[tau+1, i, j] = U[tau, i, j] + lambdaConstant * (
                    # Second derivative in x-direction
                    (U[tau, i-1, j] - 2*U[tau, i, j] + U[tau, i+1, j]) +
                    # Second derivative in y-direction  
                    (U[tau, i, j-1] - 2*U[tau, i, j] + U[tau, i, j+1])
                )
    return U

def plot_surface(u_k, k, ax):
    ax.clear()
    X, Y = np.meshgrid(xDomain, yDomain)
    
    # Transpose u_k to match meshgrid orientation
    surf = ax.plot_surface(X, Y, u_k.T, 
                          cmap='hot',
                          alpha=0.9)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_zlabel('Temperature')
    ax.set_title(f'2D Heat Equation: t = {k*timeStepSize:.4f}')
    ax.view_init(elev=30, azim=45)
    
    # Set consistent z-axis limits for better visualization
    ax.set_zlim(0, 100)
    
    return surf

# Main execution
if __name__ == "__main__":
    # Initialize and solve
    print("Initializing matrix...")
    emptyMatrix = initMatrix()
    print("Calculating temperature evolution...")
    tempMatrix = calculateTemperature(emptyMatrix)
    
    # Save temperature matrix to CSV file
    print("Saving temperature matrix to file...")
    np.savetxt("temperature_data.csv", tempMatrix.reshape(numPointsTime, -1), delimiter=",")
    
    # Create animation
    print("Creating animation...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(k):
        return plot_surface(tempMatrix[k], k, ax)
    
    # anim = FuncAnimation(fig, animate, interval=100, frames=numPointsTime, repeat=True)
    
    # Show the plot
    # plt.tight_layout()
    # plt.show()
    
    # Optionally save animation
    # anim.save("heat_equation_corrected.gif", writer='pillow', fps=10)