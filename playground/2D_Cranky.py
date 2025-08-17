import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

# Global Constants
xLength = 10  
yLength = 10  
maxTime = 0.5  
diffusivityConstant = 4  
numPointsSpace = 50  
numPointsTime = 2000

# Domain setup
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
timeDomain = np.linspace(0, maxTime, numPointsTime)

# Step sizes
timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

print(f"Time step: {timeStepSize:.6f}")
print(f"Space step: {spaceStepSize:.6f}")

# Crank-Nicolson parameters (following your notes)
c = diffusivityConstant * timeStepSize / 2  # Note: factor of 2 for Crank-Nicolson
cx = c / (spaceStepSize**2)
cy = c / (spaceStepSize**2)
alpha = 1 + 2*cx + 2*cy
beta = 1 - 2*cx - 2*cy

print(f"c = {c:.6f}")
print(f"cx = cy = {cx:.6f}")
print(f"alpha = {alpha:.6f}")
print(f"beta = {beta:.6f}")

# Initial condition - a hot spot in the center
u0 = lambda x, y: 10 * np.exp(-((x - xLength/2)**2 + (y - yLength/2)**2) / 2)
left = lambda t, y: 20 + 10 * y * (yLength - y) * t**2
right = lambda t, y: 20 + 100 * y * (yLength - y)**3 * (t - 1)**2 * (t > 1)
top = lambda t, x: 20 + 5 * x * (xLength - x) * t**4
bottom = lambda t, x: 20

def initMatrix(t_nodes, x_nodes, y_nodes, left, right, bottom, top, u0, xDomain, yDomain, tDomain):
    """Initialize matrix with boundary and initial conditions"""
    matrix = np.zeros((t_nodes, x_nodes, y_nodes))
    
    # Set boundary conditions for all time steps
    for tau in range(t_nodes):
        t = tDomain[tau]
        # Left and right boundaries
        for i in range(y_nodes):
            y = yDomain[i]
            matrix[tau, 0, i] = left(t, y)
            matrix[tau, -1, i] = right(t, y)
        # Bottom and top boundaries
        for j in range(x_nodes):
            x = xDomain[j]
            matrix[tau, j, 0] = bottom(t, x)
            matrix[tau, j, -1] = top(t, x)
    
    # Set initial condition at t=0
    for i in range(x_nodes):
        for j in range(y_nodes):
            matrix[0, i, j] = u0(xDomain[i], yDomain[j])
    
    return matrix

def build_2d_crank_nicolson_matrix(nx, ny, cx, cy):
    """
    Build sparse matrix G for: -cx*U[i-1,j] + α*U[i,j] - cx*U[i+1,j] - cy*U[i,j-1] - cy*U[i,j+1] = RHS
    """
    n_interior_x = nx - 2
    n_interior_y = ny - 2
    n_total = n_interior_x * n_interior_y
    
    # Main diagonal: α coefficients
    main_diagonal = np.full(n_total, alpha)
    
    # x-direction coupling: -cx coefficients (±1 in flattened index)
    x_off_diagonal = np.full(n_total - 1, -cx)
    # Zero out connections across y-boundaries
    for i in range(n_interior_x - 1, n_total - 1, n_interior_x):
        if i < len(x_off_diagonal):
            x_off_diagonal[i] = 0
    
    # y-direction coupling: -cy coefficients (±n_interior_x in flattened index)
    y_off_diagonal = np.full(n_total - n_interior_x, -cy)
    
    # Assemble sparse matrix
    diagonals = [y_off_diagonal, x_off_diagonal, main_diagonal, 
                 x_off_diagonal[:n_total-1], y_off_diagonal]
    offsets = [-n_interior_x, -1, 0, 1, n_interior_x]
    
    G = diags(diagonals, offsets=offsets, shape=(n_total, n_total), format='csr')
    
    return G, n_interior_x, n_interior_y

def calculateTemperatureCN(U):
    """Crank-Nicolson time stepping"""
    nx, ny = numPointsSpace, numPointsSpace
    
    G, n_interior_x, n_interior_y = build_2d_crank_nicolson_matrix(nx, ny, cx, cy)
    
    for tau in range(numPointsTime - 1):
        rhs = np.zeros(n_interior_x * n_interior_y)
        
        idx = 0
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # RHS = β*U_τ + cx*(neighbors_x) + cy*(neighbors_y) + boundary_terms
                rhs[idx] = beta * U[tau, i, j]
                rhs[idx] += cx * (U[tau, i-1, j] + U[tau, i+1, j])
                rhs[idx] += cy * (U[tau, i, j-1] + U[tau, i, j+1])
                
                # Boundary contributions
                if i == 1:
                    rhs[idx] += cx * U[tau+1, 0, j]
                if i == nx-2:
                    rhs[idx] += cx * U[tau+1, -1, j]
                if j == 1:
                    rhs[idx] += cy * U[tau+1, i, 0]
                if j == ny-2:
                    rhs[idx] += cy * U[tau+1, i, -1]
                
                idx += 1
        
        # Solve G*u_{τ+1} = rhs
        u_next_interior = spsolve(G, rhs)
        U[tau+1, 1:-1, 1:-1] = u_next_interior.reshape((n_interior_x, n_interior_y))
    
    return U

def plot_surface(u_k, k, ax):
    ax.clear()
    X, Y = np.meshgrid(xDomain, yDomain)
    
    surf = ax.plot_surface(X, Y, u_k.T, 
                          cmap='hot',
                          alpha=0.9)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_zlabel('Temperature')
    ax.set_title(f'2D Heat Equation (Crank-Nicolson): t = {k*timeStepSize:.4f}')
    ax.view_init(elev=30, azim=45)
    
    # Dynamic z-limits
    z_max = max(100, np.max(u_k))
    ax.set_zlim(0, z_max)
    
    return surf

# Main execution
if __name__ == "__main__":
    print("Initializing matrix...")
    emptyMatrix = initMatrix(
        t_nodes=numPointsTime,
        x_nodes=numPointsSpace, 
        y_nodes=numPointsSpace,
        left=left,
        right=right, 
        bottom=bottom,
        top=top,
        u0=u0,
        xDomain=xDomain,
        yDomain=yDomain,
        tDomain=timeDomain
    )
    
    print("Calculating temperature evolution using Crank-Nicolson...")
    tempMatrix = calculateTemperatureCN(emptyMatrix)
    
    # Create animation
    print("Creating animation...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(k):
        return plot_surface(tempMatrix[k], k, ax)
    
    anim = FuncAnimation(fig, animate, interval=100, frames=min(numPointsTime, 200), repeat=True)
    anim.save("heat_equation_2d_crank_nicolson.gif", writer='pillow', fps=10)
    plt.show()