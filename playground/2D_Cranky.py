import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

# Global Constants - STABLE VERSION
xLength = 10  
yLength = 10  
maxTime = 0.5  
diffusivityConstant = 4  
numPointsSpace = 50  
numPointsTime = 200

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
cy = c / (spaceStepSize**2)  # Same as cx since dx = dy
alpha = 1 + 2*cx + 2*cy
beta = 1 - 2*cx - 2*cy

print(f"c = {c:.6f}")
print(f"cx = cy = {cx:.6f}")
print(f"alpha = {alpha:.6f}")
print(f"beta = {beta:.6f}")

# Boundary conditions (from your MATLAB code)
u0 = lambda x, y: 20
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

def build_2d_crank_nicolson_matrix(nx, ny, cx, cy, alpha):
    """
    Build the sparse matrix G for the 2D Crank-Nicolson system Gu_{tau+1} = g
    
    Following your notes:
    -cx*U[i-1,j] + alpha*U[i,j] - cx*U[i+1,j] - cy*U[i,j-1] - cy*U[i,j+1] = RHS
    """
    # Interior points only (excluding boundaries)
    n_interior_x = nx - 2
    n_interior_y = ny - 2
    n_total = n_interior_x * n_interior_y
    
    print(f"Building matrix for {n_interior_x} x {n_interior_y} = {n_total} interior points")
    
    # Create 1D tridiagonal matrices
    # For x-direction: -cx, alpha, -cx (but we need to account for y-coupling)
    main_diag_1d = np.ones(n_interior_x)
    off_diag_1d = -cx * np.ones(n_interior_x - 1)
    
    # Create tridiagonal matrix for x-direction
    Tx = diags([off_diag_1d, main_diag_1d, off_diag_1d], 
               offsets=[-1, 0, 1], 
               shape=(n_interior_x, n_interior_x), 
               format='csr')
    
    # Identity matrix for y-direction
    Iy = eye(n_interior_y, format='csr')
    
    # Identity matrix for x-direction  
    Ix = eye(n_interior_x, format='csr')
    
    # Create tridiagonal matrix for y-direction
    main_diag_1d_y = np.ones(n_interior_y)
    off_diag_1d_y = -cy * np.ones(n_interior_y - 1)
    
    Ty = diags([off_diag_1d_y, main_diag_1d_y, off_diag_1d_y], 
               offsets=[-1, 0, 1], 
               shape=(n_interior_y, n_interior_y), 
               format='csr')
    
    # Build 2D matrix using Kronecker products
    # G = kron(Iy, Tx) + kron(Ty, Ix) but with correct coefficients
    
    # Start with main diagonal (alpha terms)
    G = alpha * eye(n_total, format='csr')
    
    # Add x-direction coupling: -cx for i±1,j terms
    # This corresponds to ±1 in the flattened indexing
    x_coupling_diag = -cx * np.ones(n_total - 1)
    # But we need to zero out connections across y-boundaries
    for i in range(n_interior_x - 1, n_total - 1, n_interior_x):
        if i < len(x_coupling_diag):
            x_coupling_diag[i] = 0
    
    # Add y-direction coupling: -cy for i,j±1 terms  
    # This corresponds to ±n_interior_x in the flattened indexing
    y_coupling_diag = -cy * np.ones(n_total - n_interior_x)
    
    # Combine all diagonals
    diagonals = [y_coupling_diag, x_coupling_diag, np.full(n_total, alpha), 
                 x_coupling_diag[:n_total-1], y_coupling_diag]
    offsets = [-n_interior_x, -1, 0, 1, n_interior_x]
    
    G = diags(diagonals, offsets=offsets, shape=(n_total, n_total), format='csr')
    
    return G, n_interior_x, n_interior_y

def calculateTemperatureCN(U):
    """
    2D Crank-Nicolson time stepping
    """
    nx, ny = numPointsSpace, numPointsSpace
    
    # Build the system matrix G (this is the LHS matrix)
    G, n_interior_x, n_interior_y = build_2d_crank_nicolson_matrix(nx, ny, cx, cy, alpha)
    
    print("Starting Crank-Nicolson time stepping...")
    
    for tau in range(numPointsTime - 1):
        if tau % 50 == 0:
            print(f"Time step {tau}/{numPointsTime-1}, t = {tau*timeStepSize:.4f}")
        
        # Build RHS vector g
        # g comes from: beta*U[tau] + cx*(neighbors in x) + cy*(neighbors in y) + boundary terms
        
        rhs = np.zeros(n_interior_x * n_interior_y)
        
        # Flatten interior points from previous time step
        u_prev_interior = U[tau, 1:-1, 1:-1].flatten()
        
        # Main contribution: beta * U_prev for interior points
        rhs = beta * u_prev_interior
        
        # Add contributions from previous time step neighbors (explicit part)
        idx = 0
        for j in range(1, ny-1):  # j is y-index
            for i in range(1, nx-1):  # i is x-index
                # Add explicit x-direction terms: cx*(U[i-1,j] + U[i+1,j])
                rhs[idx] += cx * (U[tau, i-1, j] + U[tau, i+1, j])
                
                # Add explicit y-direction terms: cy*(U[i,j-1] + U[i,j+1])  
                rhs[idx] += cy * (U[tau, i, j-1] + U[tau, i, j+1])
                
                # Add boundary contributions for current time step (implicit part)
                # Left boundary contribution
                if i == 1:
                    rhs[idx] += cx * U[tau+1, 0, j]
                # Right boundary contribution  
                if i == nx-2:
                    rhs[idx] += cx * U[tau+1, -1, j]
                # Bottom boundary contribution
                if j == 1:
                    rhs[idx] += cy * U[tau+1, i, 0]
                # Top boundary contribution
                if j == ny-2:
                    rhs[idx] += cy * U[tau+1, i, -1]
                
                idx += 1
        
        # Solve the linear system G * u_next = rhs
        try:
            u_next_interior = spsolve(G, rhs)
            
            # Reshape and assign back to interior points
            U[tau+1, 1:-1, 1:-1] = u_next_interior.reshape((n_interior_x, n_interior_y))
            
            # Check for stability
            max_temp = np.max(U[tau+1])
            min_temp = np.min(U[tau+1])
            
            if max_temp > 1000 or min_temp < -100:
                print(f"WARNING: Extreme temperatures detected at t={tau*timeStepSize:.4f}")
                print(f"Temperature range: [{min_temp:.2f}, {max_temp:.2f}]")
                break
                
        except Exception as e:
            print(f"ERROR: Failed to solve linear system at time step {tau}: {e}")
            break
    
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
    z_min = max(15, np.min(u_k))
    z_max = min(100, np.max(u_k))
    ax.set_zlim(z_min, z_max)
    
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
    
    plt.tight_layout()
    plt.show()
    
    # Optionally save animation
    anim.save("heat_equation_2d_crank_nicolson.gif", writer='pillow', fps=10)